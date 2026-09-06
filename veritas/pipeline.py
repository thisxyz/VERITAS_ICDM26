# -*- coding: utf-8 -*-
"""VERITAS pipeline (Algorithm 3) and the undefended (Baseline) counterpart.

Simulation flow:
  1. load & pre-process the dataset (per-dim scaling, random 50/25/25 split)
  2. inject n_atk = r*|V| fake nodes with crafted features and social links
  3. user side (Stage 1): Alg.2 feature perturbation + KRR verification list
  4. server side: rebuild a noisy adjacency from mutually acknowledged claims
     * Baseline: train GCN directly on the poisoned reports
     * VERITAS : Stage 2 pruning + Stage 3 dual denoising + Stage 4 GCN
"""
import numpy as np
import torch

from . import attack as attack_mod
from . import denoise as denoise_mod
from . import mechanisms as mech_mod
from . import structure as struct_mod


def _undirected(edge_index):
    """Return only one direction per undirected edge (a<b)."""
    a = edge_index[0]
    b = edge_index[1]
    mask = a < b
    return np.stack([a[mask], b[mask]])


def _rebuild_noisy_adjacency(perturbed_levels, claim_edges, n):
    """Server-side adjacency reconstructed from mutually acknowledged
    (both directed reports > 0) KRR claims."""
    lvl = perturbed_levels
    s, d = claim_edges
    seen = {}
    for i in range(lvl.shape[0]):
        if lvl[i] <= 0:
            continue
        a, b = int(s[i]), int(d[i])
        key = (a, b) if a < b else (b, a)
        if a < b:
            seen[key] = seen.get(key, [0, 0]); seen[key][0] = 1
        else:
            seen.setdefault(key, [0, 0])[1] = 1
    src = [k[0] for k, v in seen.items() if v[0] and v[1]]
    dst = [k[1] for k, v in seen.items() if v[0] and v[1]]
    if src:
        return np.array([src, dst], dtype=np.int64)
    return np.empty((2, 0), dtype=np.int64)


def zscore_features(x):
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    return (x - mu) / sd


def simulate(cfg):
    """Run one configuration.  cfg is a dict with keys:
    dataset, mech, defense ('baseline'|'veritas'), seed, eps_x, eps_a,
    kappa, r_star, t, ff_degree, attack_style, delta, gamma, lam,
    nfr_k, hoa_alpha, hoa_hops, input_zscore, hidden, epochs, lr, wd,
    patience, dropout, device.
    Returns a dict of metrics.
    """
    from .data import load_dataset, preprocess

    rng = np.random.default_rng(cfg.get("seed", 0))
    data = load_dataset(cfg["dataset"], cfg.get("data_root", "./data"))
    G = preprocess(data, split_seed=cfg.get("seed", 0))
    ei = _undirected(G["edge_index"])
    n = G["n"]
    d = G["d"]
    y = G["y"]

    # ---------------- attack injection (Alg. 1) ----------------
    style = cfg.get("attack_style", "max")
    direct = style in ("max", "min", "mean")   # attacker crafts post-mechanism
    atk = attack_mod.inject_attack(
        ei, n, r_star=cfg.get("r_star", 0.01), t=cfg.get("t", 8),
        ff_degree=cfg.get("ff_degree", 3),
        fake_style=style, seed=cfg.get("seed", 0))
    poi_ei = atk["edge_index"]              # undirected poisoned edges
    n_total = atk["n_total"]
    fake_ids = atk["fake_ids"]
    if direct:
        fake_x = attack_mod.direct_reports(
            atk["n_atk"], d, style, G["x"], atk["victims_per_fake"],
            seed=cfg.get("seed", 0))
    else:
        fake_x = attack_mod.make_fake_features(
            atk["n_atk"], d, style=style, seed=cfg.get("seed", 0))
    x_all_raw = np.concatenate([G["x"], fake_x], axis=0)

    # ---------------- Stage 1: user-side perturbation ----------------
    if direct:
        # attacker reports bypass the mechanism; only benign users perturb
        x_pert_benign, _ = mech_mod.algorithm2(
            G["x"], cfg.get("eps_x", 0.1), mech=cfg.get("mech", "PM"),
            delta=cfg.get("delta", 30.0), seed=cfg.get("seed", 0), rect=True)
        x_pert = np.concatenate([x_pert_benign, fake_x.astype(np.float32)],
                                axis=0)
    else:
        x_pert, _ = mech_mod.algorithm2(
            x_all_raw, cfg.get("eps_x", 0.1), mech=cfg.get("mech", "PM"),
            delta=cfg.get("delta", 30.0), seed=cfg.get("seed", 0), rect=True)
    lvl_pert, (cs, cd, lvl_true) = struct_mod.build_verification_reports(
        poi_ei, malicious_ids=fake_ids, kappa=cfg.get("kappa", 5),
        eps_a=cfg.get("eps_a", 5.0), seed=cfg.get("seed", 0),
        deterministic_ids=fake_ids if direct else None)

    # ---------------- server side ----------------
    a0 = _rebuild_noisy_adjacency(lvl_pert, (cs, cd), n_total)
    defense = cfg.get("defense", "veritas")

    if defense == "baseline":
        keep_edges = a0
        feat = x_pert
        if cfg.get("input_zscore", True):
            feat = zscore_features(feat)
        removed_fake, removed_benign = 0, 0
    else:  # veritas
        pr = struct_mod.server_prune(
            lvl_pert, (cs, cd), n_total, kappa=cfg.get("kappa", 5),
            eps_a=cfg.get("eps_a", 5.0),
            gamma=cfg.get("gamma"), lam=cfg.get("lam", 1.0))
        keep_edges = pr["keep_edges"]
        # Stage 3: dual denoising (on the pruned node set)
        feat = x_pert
        if cfg.get("use_nfr", True):
            feat = denoise_mod.nfr(x_pert, y, G["train"],
                                   k=cfg.get("nfr_k"))
        if cfg.get("use_hoa", True):
            feat = denoise_mod.hoa(feat, keep_edges, n_total,
                                   alpha=cfg.get("hoa_alpha", 0.5),
                                   hops=cfg.get("hoa_hops", 1))
        if cfg.get("input_zscore", True):
            feat = zscore_features(feat)
        removed_set = set(pr["removed"])
        removed_fake = len([i for i in fake_ids if int(i) in removed_set])
        removed_benign = len(pr["removed"]) - removed_fake

    # ---------------- Stage 4: train GCN, evaluate ----------------
    test_acc = _train_and_eval(G, keep_edges, feat, cfg)
    det_fake = removed_fake / max(1, atk["n_atk"])
    false_rem = removed_benign / max(1, n)
    return {
        "dataset": cfg["dataset"], "mech": cfg["mech"],
        "defense": defense, "seed": cfg.get("seed", 0),
        "test_acc": float(test_acc),
        "det_fake": float(det_fake),
        "false_rem": float(false_rem),
        "n_atk": int(atk["n_atk"]),
    }


def _train_and_eval(G, edge_index, feat, cfg):
    import torch
    import torch.nn.functional as F
    from .models import GCN

    device = cfg.get("device", "cpu")
    edge = torch.tensor(edge_index, dtype=torch.long, device=device)
    x = torch.tensor(feat, dtype=torch.float32, device=device)
    # only original nodes have labels
    n = G["n"]
    y = torch.tensor(G["y"], dtype=torch.long, device=device)
    tr = torch.tensor(G["train"], dtype=torch.bool, device=device)[:n]
    va = torch.tensor(G["val"], dtype=torch.bool, device=device)[:n]
    te = torch.tensor(G["test"], dtype=torch.bool, device=device)[:n]

    model = GCN(in_dim=feat.shape[1], hidden=cfg.get("hidden", 64),
                n_class=G["c"], dropout=cfg.get("dropout", 0.5)).to(device)
    opt = torch.optim.Adam(model.parameters(),
                           lr=cfg.get("lr", 1e-2),
                           weight_decay=cfg.get("wd", 5e-4))
    epochs = cfg.get("epochs", 400)
    patience = cfg.get("patience", 100)
    best_val, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x, edge)
        loss = F.cross_entropy(out[:n][tr], y[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            outv = model(x, edge)[:n]
            val_acc = (outv[va].argmax(1) == y[va]).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        outt = model(x, edge)[:n]
        te_acc = (outt[te].argmax(1) == y[te]).float().mean().item()
    return te_acc
