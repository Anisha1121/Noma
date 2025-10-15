# inference_match.py
# -------------------------------------------------------------
# Use trained GNN to predict NOMA pairings and compute throughput
# -------------------------------------------------------------

import torch
import pandas as pd
import numpy as np
import networkx as nx
import os
import glob
import argparse
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from model_gnn import PairPredictionGNN

# =============================================================
# CONFIGURATION
# =============================================================
DEFAULT_CHECKPOINT = "../checkpoints/best_model.pt"
DEFAULT_RESULTS_DIR = "../results/"
DEFAULT_OUTPUT = "inference_results.csv"

# System parameters (same as in your simulation)
TOTAL_POWER = 1.0
NOISE_POWER = 1e-9
B_TOTAL = 20e6  # 20 MHz
SIC_THRESHOLD_DB = 8

# =============================================================
# HELPER FUNCTIONS
# =============================================================

def calc_pair_rate(h1, h2):
    """Compute power allocation and achievable rates."""
    P1 = total_power * h2 / (h1 + h2)
    P2 = total_power * h1 / (h1 + h2)
    R1 = np.log2(1 + (P1 * h1) / (P2 * h1 + noise_power))
    R2 = np.log2(1 + (P2 * h2) / noise_power)
    return P1, P2, R1, R2, R1 + R2

def sic_satisfied(h1, h2):
    """Check SIC feasibility."""
    return 10 * np.log10(h2 / h1) >= sic_threshold_db

# =============================================================
# LOAD MODEL
# =============================================================
ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
print(f"✅ Loaded trained model from {CHECKPOINT_PATH}")

model = PairPredictionGNN(
    in_channels=ckpt['in_channels'],
    hidden_channels=ckpt['hidden_dim'],
    out_channels=ckpt['out_dim'],
    num_layers=ckpt['num_layers'],
    dropout=ckpt['dropout']
)
model.load_state_dict(ckpt['model_state'])
model.eval()

# =============================================================
# LOAD NEW GRAPH (unseen data)
# =============================================================
print(f"\nLoading unseen graph: {NEW_GRAPH_PATH}")
df = pd.read_csv(NEW_GRAPH_PATH)

# Normalize features same way as training
feature_cols = ["distance_m", "path_loss_dB", "shadowing_dB", "rayleigh_fading", "h_dB"]
df[feature_cols] = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std()

# Node features
x = torch.tensor(df[feature_cols].values, dtype=torch.float)
num_nodes = x.size(0)
node_ids = np.arange(num_nodes)

# Create dummy edges (for message passing, fully-connected approx.)
# For inference, you can just assume no prior edges
edge_index = torch.empty((2, 0), dtype=torch.long)

# Build PyG data object
data = Data(x=x, edge_index=edge_index)

# =============================================================
# GNN EMBEDDING & EDGE SCORING
# =============================================================
print("\nComputing node embeddings and pair scores...")
with torch.no_grad():
    z = model.encode(data.x, to_undirected(data.edge_index, num_nodes=num_nodes))

# Separate strong and weak users by channel gain
h_values = df["h_linear"].values
sorted_idx = np.argsort(h_values)
weak_users = sorted_idx[:num_nodes // 2]
strong_users = sorted_idx[num_nodes // 2:]

# Candidate strong–weak pairs
candidate_pairs = []
for i in weak_users:
    for j in strong_users:
        if sic_satisfied(min(h_values[i], h_values[j]), max(h_values[i], h_values[j])):
            candidate_pairs.append((i, j))

print(f"Total candidate pairs checked: {len(candidate_pairs)}")

# Prepare tensors for scoring
if len(candidate_pairs) == 0:
    raise RuntimeError("No valid candidate pairs found (check SIC threshold or channel data).")

pairs_tensor = torch.tensor(candidate_pairs, dtype=torch.long).T
with torch.no_grad():
    logits = model.decode_edges(z, pairs_tensor).sigmoid().numpy()

# Attach scores
scores_df = pd.DataFrame(candidate_pairs, columns=["User1_ID", "User2_ID"])
scores_df["Pair_Prob"] = logits

# =============================================================
# GRAPH MATCHING (MAX WEIGHT MATCHING)
# =============================================================
print("\nBuilding maximum-weight matching based on GNN scores...")
G = nx.Graph()
for _, row in scores_df.iterrows():
    G.add_edge(int(row["User1_ID"]), int(row["User2_ID"]), weight=row["Pair_Prob"])

matching = nx.max_weight_matching(G, maxcardinality=True)
pairs = list(matching)
print(f"✅ Matched {len(pairs)} NOMA pairs out of {num_nodes} users")

# =============================================================
# THROUGHPUT & RATE CALCULATIONS
# =============================================================
data_out = []
total_rate = 0.0
for (u1, u2) in pairs:
    h1, h2 = sorted([h_values[u1], h_values[u2]])
    P1, P2, R1, R2, R_sum = calc_pair_rate(h1, h2)
    data_out.append([u1, u2, h1, h2, P1, P2, R1, R2, R_sum])
    total_rate += R_sum

noma_pairs = len(pairs)
oma_users = num_nodes - (2 * noma_pairs)
B_unit = B_total / (noma_pairs + oma_users)
throughput_total = total_rate * B_unit / 1e6

print(f"\n📊 Inference Summary:")
print(f"- NOMA pairs formed: {noma_pairs}")
print(f"- OMA users: {oma_users}")
print(f"- System Throughput: {throughput_total:.2f} Mbps")

# =============================================================
# SAVE RESULTS TO CSV
# =============================================================
results_df = pd.DataFrame(data_out, columns=[
    "User1_ID", "User2_ID", "h1_linear", "h2_linear",
    "P1", "P2", "R1_bitsHz", "R2_bitsHz", "R_sum_bitsHz"
])
results_df["Throughput_Mbps"] = results_df["R_sum_bitsHz"] * (B_unit / 1e6)
results_df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"\n✅ Saved inference results to {OUTPUT_CSV_PATH}")
print("Top 5 predicted NOMA pairs:\n", results_df.head())

# =============================================================
# COMMAND LINE INTERFACE
# =============================================================

def main():
    parser = argparse.ArgumentParser(description="GNN-based NOMA Pairing Inference")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to trained model checkpoint")
    parser.add_argument("--test_folder", type=str, default=None,
                        help="Specific test folder path (optional)")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR,
                        help="Directory containing result folders")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help="Output CSV file path")
    
    args = parser.parse_args()
    
    try:
        results_df = run_inference(
            checkpoint_path=args.checkpoint,
            test_folder_path=args.test_folder,
            results_dir=args.results_dir,
            output_path=args.output
        )
        print("\n✅ Inference completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
