import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
import os
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = "D:/Developer/NOMA_new/dataset"
USERS_CSV = os.path.join(BASE_DIR, "merged_h_values.csv")
PAIRS_CSV = os.path.join(BASE_DIR, "merged_pairs.csv")

# ============================================================
# LOAD MERGED DATA
# ============================================================
users_df = pd.read_csv(USERS_CSV)
pairs_df = pd.read_csv(PAIRS_CSV)

print(f"Loaded {len(users_df)} user records and {len(pairs_df)} pair records.")

# ============================================================
# NORMALIZE NUMERIC FEATURES
# ============================================================
feature_cols = [
    "distance_m", "path_loss_dB", "shadowing_dB",
    "rayleigh_fading", "h_dB"
]

# Normalize (z-score)
users_df[feature_cols] = (
    users_df[feature_cols] - users_df[feature_cols].mean()
) / users_df[feature_cols].std()

# ============================================================
# GROUP BY GRAPH_ID TO CREATE PER-GRAPH OBJECTS
# ============================================================
graphs = []
for gid, user_group in users_df.groupby("Graph_ID"):
    pair_group = pairs_df[pairs_df["Graph_ID"] == gid]

    # --- node features ---
    x = torch.tensor(
        user_group[feature_cols].values, dtype=torch.float
    )

    # --- edges (NOMA pairs only) ---
    edges = pair_group[pair_group["Mode"] == "NOMA"][["User1_ID", "User2_ID"]].values
    if len(edges) == 0:
        continue
    edge_index = torch.tensor(edges.T, dtype=torch.long)

    # --- edge labels ---
    y = torch.ones(edge_index.shape[1], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.graph_id = gid
    graphs.append(data)

print(f"Created {len(graphs)} graph samples (each with 500 nodes).")

# ============================================================
# SAVE THE DATASET
# ============================================================
torch.save(graphs, os.path.join(BASE_DIR, "bpf_graph_dataset.pt"))
print(f"✅ Saved dataset to {BASE_DIR}/bpf_graph_dataset.pt")
