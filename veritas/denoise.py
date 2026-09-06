# -*- coding: utf-8 -*-

import numpy as np


def nfr(x, y, train_mask, k=None):
    """Robust (median-based) feature regularisation.  `k` is accepted for
    interface compatibility but is no longer a hard dimensionality cut."""
    med = np.median(x, axis=0, keepdims=True)
    return (x - med).astype(np.float32)


def hoa(x, edge_index, n, alpha=0.5, hops=1):
    """High-order (multi-hop) aggregator: personalised aggregation along the
    graph.  Computes X <- (1-alpha)*X + alpha * D^-1 A X  (one pass per hop).
    """
    out = x.astype(np.float64).copy()
    cur = out
    deg = np.bincount(edge_index[0], minlength=n).astype(np.float64)
    for _ in range(hops):
        agg = np.zeros_like(cur)
        deg_inv = 1.0 / (deg + 1e-9)
        src, dst = edge_index
        # edge direction: src -> dst
        vals = cur[src] * deg_inv[src][:, None]
        np.add.at(agg, dst, vals)
        cur = (1.0 - alpha) * cur + alpha * agg
    return cur.astype(np.float32)
