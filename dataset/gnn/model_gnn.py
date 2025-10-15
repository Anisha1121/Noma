# model_gnn.py
# GNN for link prediction (bipartite pairing) using GraphSAGE + edge MLP
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class PairPredictionGNN(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 128,
                 out_channels: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        assert num_layers >= 2, "num_layers must be >= 2"

        convs = []
        convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            convs.append(SAGEConv(hidden_channels, hidden_channels))
        convs.append(SAGEConv(hidden_channels, out_channels))
        self.convs = nn.ModuleList(convs)

        self.dropout = nn.Dropout(dropout)

        # Edge predictor: takes [z_u || z_v] and outputs a logit
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_channels, 1)  # logits (use with BCEWithLogitsLoss)
        )

        # Optional: Node-level projection head if you later want regression tasks
        self.node_head = nn.Identity()

    def encode(self, x, edge_index):
        # x: [N, F], edge_index used for message passing
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:  # last layer no activation/dropout
                x = F.relu(x, inplace=True)
                x = self.dropout(x)
        return x  # node embeddings z

    def decode_edges(self, z, edge_pairs):
        # edge_pairs: [2, E] (src, dst)
        src, dst = edge_pairs
        h = torch.cat([z[src], z[dst]], dim=1)  # [E, 2*out_channels]
        logits = self.edge_mlp(h).view(-1)      # [E]
        return logits

    def forward(self, x, edge_index, pos_edge_index, neg_edge_index=None):
        z = self.encode(x, edge_index)  # use current graph connectivity
        pos_logits = self.decode_edges(z, pos_edge_index)
        if neg_edge_index is not None:
            neg_logits = self.decode_edges(z, neg_edge_index)
            return pos_logits, neg_logits, z
        return pos_logits, None, z
