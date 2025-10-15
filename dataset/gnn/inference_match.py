# inference_match.py
# -------------------------------------------------------------
# Use trained GNN to predict NOMA pairings and compute throughput
# Enhanced with complexity analysis and bipartite comparison
# -------------------------------------------------------------

import torch
import pandas as pd
import numpy as np
import networkx as nx
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from model_gnn import PairPredictionGNN

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")



# =============================================================
# CONFIGURATION 'd:\Developer\NOMA_new\main code\results\results_20251015_145654/'
# =============================================================
CHECKPOINT_PATH = "D:/Developer/NOMA_new/dataset/checkpoints/best_model.pt"
NEW_GRAPH_PATH  = "D:/Developer/NOMA_new/main code/results/results_20251015_173035/h_values.csv"  # Most recent results
OUTPUT_CSV_PATH = "D:/Developer/NOMA_new/dataset/inference_results.csv"

# System parameters (same as in your simulation)
total_power = 1.0
noise_power = 1e-9
B_total = 20e6
sic_threshold_db = 8

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

def validate_sic_constraint(pairs, h_values):
    """Validate that all pairs satisfy SIC constraint."""
    violations = []
    for (u1, u2) in pairs:
        h1, h2 = sorted([h_values[u1], h_values[u2]])
        if not sic_satisfied(h1, h2):
            violations.append((u1, u2, h1, h2))
    return violations

def calculate_proportional_fairness_weight(h1, h2):
    """Calculate PF weight for bipartite matching comparison."""
    P1, P2, R1, R2, R_sum = calc_pair_rate(h1, h2)
    # PF weight: log(R1) + log(R2)
    if R1 > 0 and R2 > 0:
        return np.log(R1) + np.log(R2)
    return 0

# =============================================================
# LOAD MODEL
# =============================================================
print("="*80)
print("GNN-BASED NOMA PAIRING INFERENCE WITH COMPLEXITY ANALYSIS")
print("="*80)

start_time_total = time.time()

ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
print(f"\n✅ Loaded trained model from {CHECKPOINT_PATH}")

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
print("\n" + "="*80)
print("STEP 1: GNN NODE EMBEDDING")
print("="*80)
start_time_embedding = time.time()

print("Computing node embeddings...")
with torch.no_grad():
    z = model.encode(data.x, to_undirected(data.edge_index, num_nodes=num_nodes))

embedding_time = time.time() - start_time_embedding
print(f"✓ Embedding completed in {embedding_time:.4f} seconds")
print(f"✓ Node embedding dimension: {z.shape[1]}")
print(f"✓ Complexity: O(L × N × d²) = O(3 × {num_nodes} × {z.shape[1]}²) ≈ {3 * num_nodes * z.shape[1]**2:,} operations")

# Separate strong and weak users by channel gain
print("\n" + "="*80)
print("STEP 2: CANDIDATE PAIR GENERATION WITH SIC VALIDATION")
print("="*80)
start_time_candidates = time.time()

h_values = df["h_linear"].values
sorted_idx = np.argsort(h_values)
weak_users = sorted_idx[:num_nodes // 2]
strong_users = sorted_idx[num_nodes // 2:]

print(f"✓ Weak users (bottom 50%): {len(weak_users)}")
print(f"✓ Strong users (top 50%): {len(strong_users)}")

# Candidate strong–weak pairs
candidate_pairs = []
sic_violations = 0
for i in weak_users:
    for j in strong_users:
        h_weak = min(h_values[i], h_values[j])
        h_strong = max(h_values[i], h_values[j])
        if sic_satisfied(h_weak, h_strong):
            candidate_pairs.append((i, j))
        else:
            sic_violations += 1

candidates_time = time.time() - start_time_candidates
print(f"\n✓ Total weak-strong combinations: {len(weak_users) * len(strong_users):,}")
print(f"✓ SIC-feasible candidate pairs: {len(candidate_pairs):,}")
print(f"✓ SIC violations (filtered out): {sic_violations:,}")
print(f"✓ SIC feasibility rate: {len(candidate_pairs)/(len(weak_users)*len(strong_users))*100:.2f}%")
print(f"✓ Time taken: {candidates_time:.4f} seconds")
print(f"✓ Complexity: O(N²/4) = O({num_nodes}²/4) ≈ {(num_nodes**2)//4:,} comparisons")

# Prepare tensors for scoring
if len(candidate_pairs) == 0:
    raise RuntimeError("No valid candidate pairs found (check SIC threshold or channel data).")

print("\n" + "="*80)
print("STEP 3: GNN EDGE SCORING")
print("="*80)
start_time_scoring = time.time()

pairs_tensor = torch.tensor(candidate_pairs, dtype=torch.long).T
with torch.no_grad():
    logits = model.decode_edges(z, pairs_tensor).sigmoid().numpy()

scoring_time = time.time() - start_time_scoring
print(f"✓ Scored {len(candidate_pairs):,} candidate pairs")
print(f"✓ Score range: [{logits.min():.4f}, {logits.max():.4f}]")
print(f"✓ Mean score: {logits.mean():.4f}")
print(f"✓ Time taken: {scoring_time:.4f} seconds")
print(f"✓ Complexity: O(|C| × d) = O({len(candidate_pairs)} × {z.shape[1]}) ≈ {len(candidate_pairs) * z.shape[1]:,} operations")

# Attach scores
scores_df = pd.DataFrame(candidate_pairs, columns=["User1_ID", "User2_ID"])
scores_df["Pair_Prob"] = logits

# =============================================================
# GRAPH MATCHING (MAX WEIGHT MATCHING)
# =============================================================
print("\n" + "="*80)
print("STEP 4: MAXIMUM WEIGHT MATCHING")
print("="*80)
start_time_matching = time.time()

print("Building weighted graph for matching...")
G = nx.Graph()
for _, row in scores_df.iterrows():
    G.add_edge(int(row["User1_ID"]), int(row["User2_ID"]), weight=row["Pair_Prob"])

print(f"✓ Graph nodes: {G.number_of_nodes()}")
print(f"✓ Graph edges: {G.number_of_edges():,}")

print("\nFinding maximum weight matching...")
matching = nx.max_weight_matching(G, maxcardinality=True)
pairs = list(matching)

matching_time = time.time() - start_time_matching
print(f"✓ Matched {len(pairs)} NOMA pairs")
print(f"✓ Paired users: {len(pairs) * 2}")
print(f"✓ Unpaired (OMA) users: {num_nodes - len(pairs) * 2}")
print(f"✓ Time taken: {matching_time:.4f} seconds")
print(f"✓ Complexity: O(N³) using Blossom algorithm = O({num_nodes}³) ≈ {num_nodes**3:,} operations")

# =============================================================
# SIC VALIDATION FOR FINAL PAIRS
# =============================================================
print("\n" + "="*80)
print("STEP 5: SIC CONSTRAINT VALIDATION FOR MATCHED PAIRS")
print("="*80)

violations = validate_sic_constraint(pairs, h_values)
if violations:
    print(f"⚠️  WARNING: {len(violations)} pairs violate SIC constraint!")
    for idx, (u1, u2, h1, h2) in enumerate(violations[:5]):
        print(f"  Pair {idx+1}: Users ({u1}, {u2}) | h_ratio = {10*np.log10(h2/h1):.2f} dB (< {sic_threshold_db} dB)")
else:
    print(f"✓ All {len(pairs)} pairs satisfy SIC constraint (≥ {sic_threshold_db} dB)")

# =============================================================
# THROUGHPUT & RATE CALCULATIONS
# =============================================================
print("\n" + "="*80)
print("STEP 6: POWER ALLOCATION & RATE CALCULATION")
print("="*80)
start_time_rates = time.time()

data_out = []
total_rate = 0.0
rate_list = []
power_efficiency = []

for (u1, u2) in pairs:
    h1, h2 = sorted([h_values[u1], h_values[u2]])
    P1, P2, R1, R2, R_sum = calc_pair_rate(h1, h2)
    data_out.append([u1, u2, h1, h2, P1, P2, R1, R2, R_sum])
    total_rate += R_sum
    rate_list.append(R_sum)
    power_efficiency.append(R_sum / total_power)  # bits/Hz/Watt

rates_time = time.time() - start_time_rates

noma_pairs = len(pairs)
oma_users = num_nodes - (2 * noma_pairs)
B_unit = B_total / (noma_pairs + oma_users)
throughput_total = total_rate * B_unit / 1e6

print(f"✓ Computed rates for {noma_pairs} pairs")
print(f"✓ Average sum rate per pair: {np.mean(rate_list):.3f} bits/s/Hz")
print(f"✓ Min/Max sum rate: {np.min(rate_list):.3f} / {np.max(rate_list):.3f} bits/s/Hz")
print(f"✓ Average power efficiency: {np.mean(power_efficiency):.3f} bits/s/Hz/Watt")
print(f"✓ Time taken: {rates_time:.4f} seconds")

print("\n" + "="*80)
print("GNN-BASED NOMA SYSTEM PERFORMANCE")
print("="*80)
print(f"Total Users:              {num_nodes}")
print(f"NOMA Pairs:               {noma_pairs}")
print(f"OMA Users:                {oma_users}")
print(f"Total Sum Rate:           {total_rate:.2f} bits/s/Hz")
print(f"System Throughput:        {throughput_total:.2f} Mbps")
print(f"Bandwidth per Resource:   {B_unit/1e6:.4f} MHz")
print(f"Spectral Efficiency:      {total_rate/num_nodes:.3f} bits/s/Hz/user")

# =============================================================
# BIPARTITE PROPORTIONAL FAIRNESS COMPARISON
# =============================================================
print("\n" + "="*80)
print("COMPARISON: GNN vs BIPARTITE PROPORTIONAL FAIRNESS")
print("="*80)

print("\nRunning Bipartite PF for comparison...")
start_time_bipartite = time.time()

# Build bipartite graph with PF weights
G_bipartite = nx.Graph()
G_bipartite.add_nodes_from(weak_users, bipartite=0)
G_bipartite.add_nodes_from(strong_users, bipartite=1)

for i in weak_users:
    for j in strong_users:
        h_weak = h_values[i]
        h_strong = h_values[j]
        h1, h2 = sorted([h_weak, h_strong])
        if sic_satisfied(h1, h2):
            pf_weight = calculate_proportional_fairness_weight(h1, h2)
            G_bipartite.add_edge(i, j, weight=pf_weight)

# Maximum weight matching on bipartite graph
bipartite_matching = nx.bipartite.maximum_matching(G_bipartite, top_nodes=weak_users)
bipartite_pairs = [(u, v) for u, v in bipartite_matching.items() if u in weak_users]

bipartite_time = time.time() - start_time_bipartite

# Calculate bipartite performance
bipartite_total_rate = 0.0
for (u, v) in bipartite_pairs:
    h1, h2 = sorted([h_values[u], h_values[v]])
    _, _, _, _, R_sum = calc_pair_rate(h1, h2)
    bipartite_total_rate += R_sum

bipartite_throughput = bipartite_total_rate * B_unit / 1e6

print(f"✓ Bipartite PF pairs: {len(bipartite_pairs)}")
print(f"✓ Bipartite sum rate: {bipartite_total_rate:.2f} bits/s/Hz")
print(f"✓ Bipartite throughput: {bipartite_throughput:.2f} Mbps")
print(f"✓ Bipartite time: {bipartite_time:.4f} seconds")

print("\n" + "-"*80)
print("PERFORMANCE COMPARISON")
print("-"*80)
print(f"{'Metric':<30} {'GNN Method':<20} {'Bipartite PF':<20} {'Difference':<15}")
print("-"*80)
print(f"{'Number of Pairs':<30} {noma_pairs:<20} {len(bipartite_pairs):<20} {noma_pairs - len(bipartite_pairs):<15}")
print(f"{'Sum Rate (bits/s/Hz)':<30} {total_rate:<20.2f} {bipartite_total_rate:<20.2f} {total_rate - bipartite_total_rate:<15.2f}")
print(f"{'Throughput (Mbps)':<30} {throughput_total:<20.2f} {bipartite_throughput:<20.2f} {throughput_total - bipartite_throughput:<15.2f}")
print(f"{'Relative Performance':<30} {(total_rate/bipartite_total_rate)*100:<20.2f}% {'100.00%':<20} {((total_rate/bipartite_total_rate)-1)*100:<15.2f}%")
print("-"*80)

# =============================================================
# COMPUTATIONAL COMPLEXITY ANALYSIS
# =============================================================
total_time = time.time() - start_time_total

print("\n" + "="*80)
print("COMPUTATIONAL COMPLEXITY ANALYSIS")
print("="*80)

print("\n1. GNN METHOD:")
print("-" * 40)
print(f"  a) Node Embedding:       {embedding_time:8.4f}s  | O(L×N×d²) = O(3×{num_nodes}×{z.shape[1]}²)")
print(f"  b) Candidate Generation: {candidates_time:8.4f}s  | O(N²/4) = O({(num_nodes**2)//4:,})")
print(f"  c) Edge Scoring:         {scoring_time:8.4f}s  | O(|C|×d) = O({len(candidate_pairs)}×{z.shape[1]})")
print(f"  d) Max Weight Matching:  {matching_time:8.4f}s  | O(N³) = O({num_nodes}³)")
print(f"  e) Rate Calculation:     {rates_time:8.4f}s  | O(P) = O({noma_pairs})")
print(f"  {'─'*40}")
print(f"  Total GNN Time:          {total_time:8.4f}s")

print(f"\n2. BIPARTITE PF METHOD:")
print("-" * 40)
print(f"  Total Bipartite Time:    {bipartite_time:8.4f}s  | O(N²/4 + N²log(N))")

print(f"\n3. SPEEDUP ANALYSIS:")
print("-" * 40)
if bipartite_time > total_time:
    speedup = bipartite_time / total_time
    print(f"  GNN is {speedup:.2f}x FASTER than Bipartite PF")
else:
    slowdown = total_time / bipartite_time
    print(f"  GNN is {slowdown:.2f}x SLOWER than Bipartite PF")
    print(f"  Note: GNN benefits from amortized cost over multiple inferences")

print(f"\n4. COMPLEXITY SUMMARY:")
print("-" * 40)
print(f"  GNN Theoretical:     O(N³) dominated by matching")
print(f"  Bipartite Theoretical: O(N²log(N)) for bipartite matching")
print(f"  GNN Practical:       {total_time:.4f}s for {num_nodes} users")
print(f"  Bipartite Practical: {bipartite_time:.4f}s for {num_nodes} users")

# =============================================================
# SAVE RESULTS TO CSV
# =============================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results_df = pd.DataFrame(data_out, columns=[
    "User1_ID", "User2_ID", "h1_linear", "h2_linear",
    "P1", "P2", "R1_bitsHz", "R2_bitsHz", "R_sum_bitsHz"
])
results_df["Throughput_Mbps"] = results_df["R_sum_bitsHz"] * (B_unit / 1e6)
results_df["Method"] = "GNN"

# Save GNN results
results_df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"✓ Saved GNN results to: {OUTPUT_CSV_PATH}")

# Save comparison summary
summary_file = OUTPUT_CSV_PATH.replace(".csv", "_summary.txt")
with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("GNN-BASED NOMA PAIRING - INFERENCE SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Test Graph: {NEW_GRAPH_PATH}\n")
    f.write(f"Total Users: {num_nodes}\n\n")
    
    f.write("PERFORMANCE METRICS:\n")
    f.write("-"*80 + "\n")
    f.write(f"GNN Method:\n")
    f.write(f"  - NOMA Pairs: {noma_pairs}\n")
    f.write(f"  - Sum Rate: {total_rate:.2f} bits/s/Hz\n")
    f.write(f"  - Throughput: {throughput_total:.2f} Mbps\n")
    f.write(f"  - Time: {total_time:.4f} seconds\n\n")
    
    f.write(f"Bipartite PF Method:\n")
    f.write(f"  - NOMA Pairs: {len(bipartite_pairs)}\n")
    f.write(f"  - Sum Rate: {bipartite_total_rate:.2f} bits/s/Hz\n")
    f.write(f"  - Throughput: {bipartite_throughput:.2f} Mbps\n")
    f.write(f"  - Time: {bipartite_time:.4f} seconds\n\n")
    
    f.write(f"Relative Performance: {(total_rate/bipartite_total_rate)*100:.2f}%\n")
    f.write(f"SIC Violations: {len(violations)}\n")
    
print(f"✓ Saved summary to: {summary_file}")

print("\n" + "="*80)
print("Top 5 GNN-Predicted NOMA Pairs:")
print("="*80)
print(results_df.head().to_string(index=False))

print("\n" + "="*80)
print("GENERATING COMPARISON VISUALIZATIONS")
print("="*80)

# Create output directory for plots
plot_dir = OUTPUT_CSV_PATH.replace(".csv", "_plots")
os.makedirs(plot_dir, exist_ok=True)

# =============================================================
# PLOT 1: Performance Comparison Bar Chart
# =============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('GNN vs Bipartite PF: Performance Comparison', fontsize=16, fontweight='bold')

# Plot 1.1: Number of Pairs
ax1 = axes[0, 0]
methods = ['GNN', 'Bipartite PF']
pairs_count = [noma_pairs, len(bipartite_pairs)]
bars1 = ax1.bar(methods, pairs_count, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
ax1.set_ylabel('Number of Pairs', fontsize=12, fontweight='bold')
ax1.set_title('NOMA Pairs Formed', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontweight='bold')

# Plot 1.2: Sum Rate Comparison
ax2 = axes[0, 1]
sum_rates = [total_rate, bipartite_total_rate]
bars2 = ax2.bar(methods, sum_rates, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
ax2.set_ylabel('Sum Rate (bits/s/Hz)', fontsize=12, fontweight='bold')
ax2.set_title('Total Sum Rate', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom', fontweight='bold')

# Plot 1.3: Throughput Comparison
ax3 = axes[1, 0]
throughputs = [throughput_total, bipartite_throughput]
bars3 = ax3.bar(methods, throughputs, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
ax3.set_ylabel('Throughput (Mbps)', fontsize=12, fontweight='bold')
ax3.set_title('System Throughput', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom', fontweight='bold')

# Plot 1.4: Execution Time Comparison
ax4 = axes[1, 1]
times = [total_time, bipartite_time]
bars4 = ax4.bar(methods, times, color=['#3498db', '#e74c3c'], alpha=0.7, edgecolor='black')
ax4.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
ax4.set_title('Computational Time', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}s',
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plot1_path = os.path.join(plot_dir, "performance_comparison.png")
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plot1_path}")
plt.close()

# =============================================================
# PLOT 2: Complexity Breakdown Pie Chart
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Computational Complexity Breakdown', fontsize=16, fontweight='bold')

# GNN Method Breakdown
ax1 = axes[0]
gnn_components = ['Embedding', 'Candidates', 'Scoring', 'Matching', 'Rates']
gnn_times = [embedding_time, candidates_time, scoring_time, matching_time, rates_time]
colors1 = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
wedges1, texts1, autotexts1 = ax1.pie(gnn_times, labels=gnn_components, autopct='%1.1f%%',
                                        colors=colors1, startangle=90, explode=[0.05]*5)
ax1.set_title('GNN Method Time Distribution', fontsize=12, fontweight='bold')
for autotext in autotexts1:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# Time Comparison
ax2 = axes[1]
comparison_labels = ['GNN\nTotal', 'Bipartite\nPF']
comparison_times = [total_time, bipartite_time]
colors2 = ['#3498db', '#e74c3c']
bars = ax2.bar(comparison_labels, comparison_times, color=colors2, alpha=0.7, edgecolor='black', width=0.5)
ax2.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
ax2.set_title('Total Execution Time', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}s',
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plot2_path = os.path.join(plot_dir, "complexity_breakdown.png")
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plot2_path}")
plt.close()

# =============================================================
# PLOT 3: Rate Distribution Comparison
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Sum Rate Distribution Comparison', fontsize=16, fontweight='bold')

# GNN Rate Distribution
ax1 = axes[0]
gnn_rates = [row[8] for row in data_out]  # R_sum values
ax1.hist(gnn_rates, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
ax1.axvline(np.mean(gnn_rates), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(gnn_rates):.2f}')
ax1.set_xlabel('Sum Rate (bits/s/Hz)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('GNN Method: Rate Distribution', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Bipartite PF Rate Distribution
ax2 = axes[1]
bipartite_rates = []
for (u, v) in bipartite_pairs:
    h1, h2 = sorted([h_values[u], h_values[v]])
    _, _, _, _, R_sum = calc_pair_rate(h1, h2)
    bipartite_rates.append(R_sum)
ax2.hist(bipartite_rates, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
ax2.axvline(np.mean(bipartite_rates), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(bipartite_rates):.2f}')
ax2.set_xlabel('Sum Rate (bits/s/Hz)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title('Bipartite PF: Rate Distribution', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plot3_path = os.path.join(plot_dir, "rate_distribution.png")
plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plot3_path}")
plt.close()

# =============================================================
# PLOT 4: Power Allocation Analysis
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Power Allocation Analysis (GNN Method)', fontsize=16, fontweight='bold')

# Power split ratio
ax1 = axes[0]
power_ratios = [row[4]/(row[4]+row[5]) for row in data_out]  # P1/(P1+P2)
ax1.hist(power_ratios, bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
ax1.axvline(np.mean(power_ratios), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(power_ratios):.3f}')
ax1.set_xlabel('Power Ratio (P_weak / P_total)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('Power Allocation to Weak Users', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Channel gain ratios
ax2 = axes[1]
channel_ratios = [10*np.log10(row[3]/row[2]) for row in data_out]  # h2/h1 in dB
ax2.hist(channel_ratios, bins=30, color='#f39c12', alpha=0.7, edgecolor='black')
ax2.axvline(sic_threshold_db, color='red', linestyle='--', linewidth=2, label=f'SIC Threshold: {sic_threshold_db} dB')
ax2.axvline(np.mean(channel_ratios), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(channel_ratios):.2f} dB')
ax2.set_xlabel('Channel Gain Ratio (h_strong / h_weak) [dB]', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title('Channel Gain Difference in Pairs', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plot4_path = os.path.join(plot_dir, "power_allocation.png")
plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plot4_path}")
plt.close()

# =============================================================
# PLOT 5: GNN Score Analysis
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('GNN Prediction Score Analysis', fontsize=16, fontweight='bold')

# Score distribution for matched vs unmatched
ax1 = axes[0]
matched_pairs_set = set(pairs)
matched_scores = []
unmatched_scores = []

for idx, row in scores_df.iterrows():
    u1, u2 = int(row["User1_ID"]), int(row["User2_ID"])
    score = row["Pair_Prob"]
    if (u1, u2) in matched_pairs_set or (u2, u1) in matched_pairs_set:
        matched_scores.append(score)
    else:
        unmatched_scores.append(score)

ax1.hist(matched_scores, bins=30, color='#2ecc71', alpha=0.7, edgecolor='black', label='Matched Pairs')
ax1.hist(unmatched_scores, bins=30, color='#e74c3c', alpha=0.5, edgecolor='black', label='Unmatched Pairs')
ax1.set_xlabel('GNN Prediction Score', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('Score Distribution: Matched vs Unmatched', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Score vs Sum Rate scatter
ax2 = axes[1]
pair_scores = []
pair_rates = []
for (u1, u2) in pairs:
    # Find score for this pair
    score_row = scores_df[
        ((scores_df["User1_ID"] == u1) & (scores_df["User2_ID"] == u2)) |
        ((scores_df["User1_ID"] == u2) & (scores_df["User2_ID"] == u1))
    ]
    if not score_row.empty:
        pair_scores.append(score_row["Pair_Prob"].iloc[0])
        h1, h2 = sorted([h_values[u1], h_values[u2]])
        _, _, _, _, R_sum = calc_pair_rate(h1, h2)
        pair_rates.append(R_sum)

ax2.scatter(pair_scores, pair_rates, alpha=0.6, c=pair_rates, cmap='viridis', edgecolor='black', s=50)
ax2.set_xlabel('GNN Prediction Score', fontsize=11, fontweight='bold')
ax2.set_ylabel('Actual Sum Rate (bits/s/Hz)', fontsize=11, fontweight='bold')
ax2.set_title('GNN Score vs Achieved Rate', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

# Add correlation coefficient
if len(pair_scores) > 0:
    correlation = np.corrcoef(pair_scores, pair_rates)[0, 1]
    ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=ax2.transAxes, fontsize=11, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plot5_path = os.path.join(plot_dir, "gnn_score_analysis.png")
plt.savefig(plot5_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plot5_path}")
plt.close()

# =============================================================
# PLOT 6: Comparative Metrics Summary
# =============================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Create comparison table as visualization
metrics = ['Pairs Formed', 'Sum Rate\n(bits/s/Hz)', 'Throughput\n(Mbps)', 
           'Avg Rate/Pair\n(bits/s/Hz)', 'Execution\nTime (s)', 
           'Relative\nPerformance (%)']
gnn_values = [
    noma_pairs,
    total_rate,
    throughput_total,
    np.mean(rate_list),
    total_time,
    100.0
]
bipartite_values = [
    len(bipartite_pairs),
    bipartite_total_rate,
    bipartite_throughput,
    np.mean(bipartite_rates),
    bipartite_time,
    100.0
]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, [v/max(gnn_values[i], bipartite_values[i]) for i, v in enumerate(gnn_values)], 
               width, label='GNN', color='#3498db', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, [v/max(gnn_values[i], bipartite_values[i]) for i, v in enumerate(bipartite_values)], 
               width, label='Bipartite PF', color='#e74c3c', alpha=0.7, edgecolor='black')

ax.set_ylabel('Normalized Value', fontsize=12, fontweight='bold')
ax.set_title('Comprehensive Performance Comparison (Normalized)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add actual values on bars
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    ax.text(bar1.get_x() + bar1.get_width()/2., height1,
            f'{gnn_values[i]:.2f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.text(bar2.get_x() + bar2.get_width()/2., height2,
            f'{bipartite_values[i]:.2f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plot6_path = os.path.join(plot_dir, "comprehensive_comparison.png")
plt.savefig(plot6_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {plot6_path}")
plt.close()

print(f"\n✅ All plots saved to: {plot_dir}/")

print("\n" + "="*80)
print("INFERENCE COMPLETED SUCCESSFULLY")
print("="*80)
print(f"Total execution time: {total_time:.4f} seconds")
print(f"\nOutput files:")
print(f"  1. Results CSV: {OUTPUT_CSV_PATH}")
print(f"  2. Summary: {summary_file}")
print(f"  3. Plots directory: {plot_dir}/")
print("="*80)
