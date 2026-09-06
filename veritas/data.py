# -*- coding: utf-8 -*-

import numpy as np
import torch


def load_dataset(name, root="./data"):
    try:
        from torch_geometric.datasets import Planetoid, LastFMAsia, Twitch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torch_geometric not available") from e
    if name == "Cora":
        ds = Planetoid(root=root, name="Cora")
    elif name == "Citeseer":
        ds = Planetoid(root=root, name="Citeseer")
    elif name == "LastFM":
        ds = LastFMAsia(root=root)
    elif name == "Twitch":
        ds = Twitch(root=root, name="ES")
    else:
        raise ValueError(f"unknown dataset {name}")
    data = ds[0]
    return data


def preprocess(data, split_seed=0):
    x = data.x.numpy().astype(np.float64)
    # per-dim scaling to [-1, 1]
    m = np.max(np.abs(x), axis=0, keepdims=True)
    m[m == 0] = 1.0
    x = x / m
    x = 2.0 * x - 1.0
    n = x.shape[0]
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    n_tr = int(round(n * 0.5))
    n_va = int(round(n * 0.25))
    tr = np.zeros(n, dtype=bool); tr[perm[:n_tr]] = True
    va = np.zeros(n, dtype=bool); va[perm[n_tr:n_tr + n_va]] = True
    te = np.zeros(n, dtype=bool); te[perm[n_tr + n_va:]] = True
    return {
        "x": x.astype(np.float32),
        "y": data.y.numpy(),
        "edge_index": data.edge_index.numpy(),
        "train": tr, "val": va, "test": te,
        "n": n, "d": x.shape[1], "c": int(data.y.max().item()) + 1,
    }
