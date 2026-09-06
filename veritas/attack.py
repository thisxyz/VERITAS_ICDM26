# -*- coding: utf-8 -*-

import numpy as np


def inject_attack(edge_index_orig, n_orig, r_star=0.01, t=8, ff_degree=3,
                  fake_style="ones", seed=0):
    rng = np.random.default_rng(seed)
    n_atk = int(round(r_star * n_orig))
    fake_ids = np.arange(n_orig, n_orig + n_atk, dtype=np.int64)
    fake_set = set(int(i) for i in fake_ids)

    # --- victim edges (fake -> benign) ---
    victims = rng.integers(0, n_orig, size=n_atk * t)
    soc_src = np.repeat(fake_ids, t)
    soc_dst = victims
    victims_per_fake = victims.reshape(n_atk, t)
    # --- fake-fake coordination (each fake links to ff_degree peers) ---
    ff_src, ff_dst = [], []
    for i, fi in enumerate(fake_ids):
        cand = [fake_ids[j] for j in range(n_atk) if j != i]
        if not cand:
            continue
        k = min(ff_degree, len(cand))
        picks = rng.choice(cand, size=k, replace=False)
        for p in picks:
            ff_src.append(fi)
            ff_dst.append(int(p))
    ff_src = np.array(ff_src, dtype=np.int64)
    ff_dst = np.array(ff_dst, dtype=np.int64)

    # concatenate: original | social | fake-fake
    src = np.concatenate([edge_index_orig[0],
                          soc_src,
                          ff_src])
    dst = np.concatenate([edge_index_orig[1],
                          soc_dst,
                          ff_dst])
    e_orig = edge_index_orig.shape[1]
    e_soc = soc_src.shape[0]
    edge_index = np.stack([src, dst], axis=0)
    # category per edge: 0=benign-benign, 1=social(attack), 2=fake-fake
    cat = np.concatenate([np.zeros(e_orig, dtype=np.int64),
                          np.ones(e_soc, dtype=np.int64),
                          np.full(ff_src.shape[0], 2, dtype=np.int64)])
    return {"edge_index": edge_index,
            "n_total": n_orig + n_atk,
            "fake_ids": fake_ids,
            "fake_set": fake_set,
            "cat": cat,
            "n_atk": n_atk,
            "victims_per_fake": victims_per_fake}


def make_fake_features(n_atk, d, style="ones", seed=0):
    """Raw (unperturbed) feature vectors crafted by S_x for the fake nodes."""
    if style == "ones":
        return np.full((n_atk, d), 1.0, dtype=np.float32)
    if style == "zeros":
        return np.zeros((n_atk, d), dtype=np.float32)
    if style == "noise":
        rng = np.random.default_rng(seed)
        return rng.uniform(-1.0, 1.0, size=(n_atk, d)).astype(np.float32)
    raise ValueError(style)


def direct_reports(n_atk, d, style, x_orig, victims_per_fake, seed=0):
    """Feature reports crafted *after* the mechanism (Algorithm 1: the
    attacker fully controls the perturbed reports it uploads, choosing any
    value in Range(M) without running the randomised mechanism).
      style 'max' : all-+1 report (legitimate range value for every mechanism)
      style 'mean': report equal to the mean raw feature of its victims
      style 'min' : all--1 report
    """
    if style == "max":
        return np.full((n_atk, d), 1.0, dtype=np.float32)
    if style == "min":
        return np.full((n_atk, d), -1.0, dtype=np.float32)
    if style == "mean":
        out = np.zeros((n_atk, d), dtype=np.float64)
        for i in range(n_atk):
            vids = victims_per_fake[i]
            out[i] = x_orig[vids].mean(axis=0)
        return out.astype(np.float32)
    raise ValueError(style)
