# -*- coding: utf-8 -*-

import numpy as np

KAPPA_DEFAULT = 5


def build_verification_reports(edge_index, malicious_ids=None, kappa=5,
                               eps_a=5.0, seed=0, deterministic_ids=None):
    """Build unperturbed verification lists (Eq. 10) from the graph and
    perturb every directed claim with KRR (Eq. 11).

    edge_index   : (2,E) undirected edges of the (poisoned) graph
    malicious_ids: ids of the injected fake nodes
    deterministic_ids : ids whose *outgoing* claims bypass KRR (the attacker
                        fully controls its uploaded reports, Alg. 1)
    Returns (perturbed_levels, (src,dst,true_level)) where levels are the KRR
    outputs (length = 2*E). Every undirected edge produces two directed
    claims (a->b, b->a).
    """
    rng = np.random.default_rng(seed)
    src = edge_index[0]
    dst = edge_index[1]
    e = edge_index.shape[1]
    mal = set()
    if malicious_ids is not None and len(malicious_ids):
        mal = set(int(i) for i in malicious_ids)

    # directed claims a->b and b->a
    s = np.concatenate([src, dst])
    d = np.concatenate([dst, src])
    lvl = np.empty(2 * e, dtype=np.int64)
    for i in range(2 * e):
        a, b = int(s[i]), int(d[i])
        if a in mal and b in mal:
            lvl[i] = kappa
        elif a in mal:            # malicious rates its victim at max
            lvl[i] = kappa
        elif b in mal:            # honest victim rates the attacker low
            lvl[i] = 1
        else:
            lvl[i] = kappa

    # KRR (Eq. 11)
    pa = np.exp(eps_a) / (np.exp(eps_a) + kappa)
    out = np.empty_like(lvl)
    same = rng.random(lvl.shape) < pa
    out[same] = lvl[same]
    n_flip = int((~same).sum())
    others = np.arange(kappa + 1)
    flip = np.empty(n_flip, dtype=np.int64)
    for t, i0 in enumerate(lvl[~same]):
        pool = others[others != i0]
        flip[t] = pool[rng.integers(0, kappa)]
    out[~same] = flip
    # attacker-controlled (deterministic) outgoing reports bypass KRR
    if deterministic_ids is not None and len(deterministic_ids):
        det = set(int(i) for i in deterministic_ids)
        for i in range(out.shape[0]):
            if int(s[i]) in det:
                out[i] = lvl[i]
    return out.astype(np.int64), (s, d, lvl)


def server_prune(perturbed_levels, claim_edges, n_nodes, kappa=5,
                 eps_a=5.0, gamma=None, lam=1.0):
    """Stage 2 (server side): assemble bilateral attestation pairs, compute
    edge confidence (Eq. 13) and node suspicion (Eq. 14), prune nodes with
    phi > gamma (Eq. 16) and return the surviving edges.

    The sign convention of Eq. 14 follows the paper's textual description:
    "a positive phi_v indicates that v consistently assigns higher trust to
    others than others assign to it", i.e.  phi_v = mean_u(a~_v,u - a~_u,v).
    """
    lvl = perturbed_levels
    s, d = claim_edges
    # group by unordered pair
    pair = {}
    for i in range(lvl.shape[0]):
        a, b = int(s[i]), int(d[i])
        key = (a, b) if a < b else (b, a)
        entry = pair.setdefault(key, [0, 0])
        if a < b:
            entry[0] = int(lvl[i])     # a->b
        else:
            entry[1] = int(lvl[i])     # b->a

    confidence = {}
    phi_num = np.zeros(n_nodes, dtype=np.float64)
    phi_den = np.zeros(n_nodes, dtype=np.float64)
    for (a, b), (rab, rba) in pair.items():
        if rab == 0 and rba == 0:
            continue
        if rab == 0 or rba == 0:
            continue  # existence gate: both must acknowledge the edge
        scons = min(rab, rba)
        sasym = abs(rab - rba)
        c = (scons / kappa) * np.exp(-lam * sasym)   # Eq. 13 (sexist=1 here)
        if c > 0:
            confidence[(a, b)] = float(c)
        # suspicion contributions (v assigns to others  -  others assign to v)
        phi_num[a] += (rab - rba)
        phi_den[a] += 1
        phi_num[b] += (rba - rab)
        phi_den[b] += 1
    nz = phi_den > 0
    phi = np.zeros(n_nodes)
    phi[nz] = phi_num[nz] / phi_den[nz]

    if gamma is None:
        pa = np.exp(eps_a) / (np.exp(eps_a) + kappa)
        gamma = 0.5 * (2.0 * pa - 1.0) * (kappa - 1.0)   # Prop. 2 midpoint
    # a node is only pruned if it has enough attestation claims for the
    # suspicion statistics to be reliable (avoids killing low-degree nodes
    # whose single noisy claim can be dominated by KRR noise)
    min_claims = 5
    removed = np.where((phi > gamma) & (phi_den >= min_claims))[0]
    removed_set = set(int(i) for i in removed)

    keep_src, keep_dst = [], []
    for (a, b), c in confidence.items():
        if a in removed_set or b in removed_set:
            continue
        keep_src.append(a)
        keep_dst.append(b)
    keep = np.array([keep_src, keep_dst], dtype=np.int64) if keep_src \
        else np.empty((2, 0), dtype=np.int64)
    return {"removed": removed.tolist(),
            "keep_edges": keep,
            "phi": phi,
            "gamma": float(gamma)}
