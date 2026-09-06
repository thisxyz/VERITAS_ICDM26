# -*- coding: utf-8 -*-
"""Two-layer GCN used as the default server-side graph learner (Stage 4).
Architecture follows the paper: 2 graph-convolution layers, hidden 64."""
import torch.nn.functional as F
from torch import nn

try:
    from torch_geometric.nn import GCNConv
except Exception:  # pragma: no cover
    GCNConv = None


class GCN(nn.Module):
    def __init__(self, in_dim, hidden=64, n_class=7, dropout=0.5):
        super().__init__()
        if GCNConv is None:
            raise RuntimeError("torch_geometric not available")
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, n_class)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
