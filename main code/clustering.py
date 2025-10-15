# -------------------- Parameters & Initialization --------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from tqdm import tqdm
import os
import time
from datetime import datetime
import seaborn as sns
from matplotlib.patches import Circle

# Parameters (Referenced: 3GPP TR 38.901, Section 7.4.1 & 7.4.2, UMa scenario)
# European Telecommunications Standards Institute
# https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/16.01.00_60/tr_138901v160100p.pdf?

np.random.seed(int(time.time()))
N, radius = 2000, 5000  # Users, cell radius (m)
fc, c = 3.5e9, 3e8  # Carrier frequency (Hz), speed of light (m/s)
lambda_c = c / fc  # Wavelength (m)
h_BS = 25  # Base station height (m)
path_loss_exp, shadow_std_db = 3.5, 8
noise_power, total_power = 1e-9, 1.0
sic_threshold_db, B_total = 8, 20e6

# ==================== NEW: Matching/Power Optimization Knobs ====================
THETA_MIN_DEG = 25          # Angular guard for pairing (bipartite PF) in degrees
PF_EPS = 1e-12              # Small epsilon for PF logs to avoid -inf
POWER_OPT_TOL = 1e-4        # Tolerance for 1-D power optimizer
POWER_OPT_MAXIT = 80        # Maximum iterations for 1-D power optimizer
CHANNEL_GAIN_EPS = 1e-12    # Small epsilon for channel gain log calculation
# ================================================================================

# Create timestamped results directory in the same folder as this script
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
script_dir = os.path.dirname(os.path.abspath(__file__))
base_results_dir = os.path.join(script_dir, "results")
results_dir = os.path.join(base_results_dir, f"results_{timestamp}")
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")

# Path loss reference (1m reference distance)
pl_1m_db = 20 * np.log10(4 * np.pi / lambda_c)

# -------------------- User Placement --------------------
r = np.sqrt(np.random.uniform(0, radius**2, N))  # 2D distance from BS
theta = np.random.uniform(0, 2*np.pi, N)
x_coords, y_coords = r * np.cos(theta), r * np.sin(theta)

# Random UE heights within 3GPP UMa limits (1.5m – 22.5m)
h_UTs = np.random.uniform(1.5, 22.5, N)

# -------------------- LOS Probability (3GPP Eq. 7.4.2-1) --------------------
def C_hUT(h_UT):
    if h_UT <= 13:
        return 0
    elif h_UT < 23:
        return ((h_UT - 13) / 10) ** 1.5
    else:
        return ((23 - 13) / 10) ** 1.5

def prob_LOS_UMa(d_2D_out, h_UT):
    P_LOS = np.zeros_like(d_2D_out, dtype=float)
    mask1 = d_2D_out <= 18
    P_LOS[mask1] = 1.0
    mask2 = ~mask1
    C_val = np.array([C_hUT(h) for h in h_UT])
    P_LOS[mask2] = (
        (18 / d_2D_out[mask2]) +
        np.exp(-d_2D_out[mask2] / 63) * (1 - 18 / d_2D_out[mask2])
    ) * (
        1 + C_val[mask2] * (5/4) * ((d_2D_out[mask2] / 100) ** 3) *
        np.exp(-d_2D_out[mask2] / 150)
    )
    return np.clip(P_LOS, 0, 1)

# -------------------- Path Loss Models --------------------
def PL_UMa_LOS(d_3D, fc):
    return 28.0 + 22 * np.log10(d_3D) + 20 * np.log10(fc / 1e9)

def PL_UMa_NLOS(d_3D, fc, h_UT):
    return (13.54 + 39.08 * np.log10(d_3D) + 20 * np.log10(fc / 1e9) - 
            0.6 * (h_UT - 1.5))

# -------------------- Generate Path Loss for Each User --------------------
d_3D = np.sqrt(r*2 + (h_BS - h_UTs)*2)
P_LOS_users = prob_LOS_UMa(r, h_UTs)

PL_dB = np.zeros(N)
is_LOS = np.zeros(N, dtype=bool)

for i in range(N):
    if np.random.rand() <= P_LOS_users[i]:
        PL_dB[i] = PL_UMa_LOS(d_3D[i], fc)
        is_LOS[i] = True
    else:
        # FIXED: Use only NLOS model for NLOS condition
        PL_dB[i] = PL_UMa_NLOS(d_3D[i], fc, h_UTs[i])
        is_LOS[i] = False

# Convert PL to linear scale
pl_linear = 10 ** (-PL_dB / 10)

# -------------------- Shadowing --------------------
shadowing = np.random.normal(0, shadow_std_db, N)  # Log-normal shadowing in dB

# -------------------- Small-Scale Fading --------------------
fading = np.zeros(N)
K_factor_LOS = 9  # Typical UMa LOS Rician K-factor in dB
K_linear = 10 ** (K_factor_LOS / 10)

for i in range(N):
    if is_LOS[i]:
        # Rician fading for LOS users
        s = np.sqrt(K_linear / (K_linear + 1))
        sigma = np.sqrt(1 / (2 * (K_linear + 1)))
        complex_fading = np.random.normal(s, sigma) + 1j * np.random.normal(0, sigma)
        fading[i] = np.abs(complex_fading)
    else:
        # Rayleigh fading for NLOS users - using standard scale
        fading[i] = np.random.rayleigh(scale=1/np.sqrt(2))

# -------------------- Channel Gain --------------------
# Combine path loss, shadowing, and fading components correctly
channel_gain_linear = fading * np.sqrt(pl_linear * 10**(-shadowing/10))
h_values = channel_gain_linear
h_db = 20 * np.log10(h_values + CHANNEL_GAIN_EPS)  # Convert to dB with small epsilon

# Debug prints
print(f"\nChannel Statistics:")
print(f"3D Distance range: {np.min(d_3D):.1f} - {np.max(d_3D):.1f} m")
print(f"Path Loss range: {np.min(PL_dB):.1f} - {np.max(PL_dB):.1f} dB")
print(f"Fading range: {np.min(fading):.3f} - {np.max(fading):.3f}")
print(f"Channel Gain range: {np.min(h_db):.1f} - {np.max(h_db):.1f} dB")

# Verify with theoretical values
d_test = 1000  # Test at 1km
pl_theoretical = 28 + 22*np.log10(d_test) + 20*np.log10(fc/1e9)
print(f"Theoretical PL at 1km: {pl_theoretical:.1f} dB")

# Create consistent path loss variable name for visualization functions
path_loss_db = PL_dB

# Channel values are now ready for clustering

# Sorted indices for clustering
sorted_indices = np.argsort(h_values)

# Save comprehensive user data including coordinates and channel components
user_data = pd.DataFrame({
    "User_ID": np.arange(N),
    "x_coord_m": x_coords,
    "y_coord_m": y_coords,
    "distance_m": r,
    "angle_rad": theta,
    "path_loss_dB": path_loss_db,
    "path_loss_linear": pl_linear,
    "shadowing_dB": shadowing,
    "rayleigh_fading": fading,
    "h_linear": h_values,
    "h_dB": h_db
})
user_data.to_csv(f"{results_dir}/h_values.csv", index=False)

# -------------------- Visualization Functions --------------------
def plot_user_distribution():
    """Plot user distribution in the cell"""
    plt.figure(figsize=(10, 8))
    circle = plt.Circle((0, 0), radius, fill=False, color='black', linewidth=2, linestyle='--')
    plt.gca().add_patch(circle)
    scatter = plt.scatter(x_coords, y_coords, c=h_db, cmap='viridis',
                         s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Channel Gain (dB)')
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.title('User Distribution in Cell with Channel Gains', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/user_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_channel_characteristics():
    """Plot channel gain distributions and characteristics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    ax1.hist(h_db, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Channel Gain (dB)')
    ax1.set_ylabel('Number of Users')
    ax1.set_title('Channel Gain Distribution')
    ax1.grid(True, alpha=0.3)

    ax2.scatter(r, path_loss_db, alpha=0.6, color='orange', s=30)
    ax2.set_xlabel('Distance from BS (m)')
    ax2.set_ylabel('Path Loss (dB)')
    ax2.set_title('Path Loss vs Distance')
    ax2.grid(True, alpha=0.3)

    ax3.hist(fading, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_xlabel('Rayleigh Fading Coefficient')
    ax3.set_ylabel('Number of Users')
    ax3.set_title('Rayleigh Fading Distribution')
    ax3.grid(True, alpha=0.3)

    ax4.plot(range(N), np.sort(h_db), linewidth=2, color='red')
    ax4.set_xlabel('User Index (Sorted)')
    ax4.set_ylabel('Channel Gain (dB)')
    ax4.set_title('Sorted Channel Gains')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{results_dir}/channel_characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_pairing_visualization(pairs_indices, name, clustering_data):
    """Visualize user pairing for each clustering method"""
    plt.figure(figsize=(12, 10))
    circle = plt.Circle((0, 0), radius, fill=False, color='black', linewidth=2, linestyle='--')
    plt.gca().add_patch(circle)

    plt.scatter(x_coords, y_coords, c='lightgray', s=30, alpha=0.5, label='Unpaired Users')

    colors = plt.cm.Set3(np.linspace(0, 1, len(pairs_indices)))

    for idx, (u1, u2) in enumerate(pairs_indices):
        h1, h2 = h_values[u1], h_values[u2]
        if sic_satisfied(min(h1, h2), max(h1, h2)):
            plt.plot([x_coords[u1], x_coords[u2]], [y_coords[u1], y_coords[u2]],
                     color=colors[idx], linewidth=1.5, alpha=0.7)
            plt.scatter([x_coords[u1], x_coords[u2]], [y_coords[u1], y_coords[u2]],
                        c=[colors[idx]], s=50, edgecolors='black', linewidth=0.5)

    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.title(f'{name.title()} Clustering - User Pairing Visualization', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/{name}_pairing.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_channel_components_analysis():
    """Plot detailed analysis of individual channel components"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    ax1.scatter(r, path_loss_db, alpha=0.6, s=20, color='red')
    ax1.scatter(r, path_loss_db, alpha=0.6, s=20, color='red')
    dist_theory = np.linspace(100, radius, 100)
    pl_theory = pl_1m_db + 10 * path_loss_exp * np.log10(dist_theory)
    ax1.plot(dist_theory, pl_theory, 'k--', linewidth=2, label='Theoretical Path Loss')
    ax1.set_xlabel('Distance from BS (m)')
    ax1.set_ylabel('Path Loss (dB)')
    ax1.set_title('Path Loss vs Distance (3GPP Model)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    scatter2 = ax2.scatter(x_coords, y_coords,
                           c=shadowing, cmap='RdBu_r', s=30, alpha=0.7)
    circle2 = plt.Circle((0, 0), radius, fill=False, color='black', linewidth=2, linestyle='--')
    ax2.add_patch(circle2)
    plt.colorbar(scatter2, ax=ax2, label='Shadowing (dB)')
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Spatial Distribution of Shadowing Effects')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    components = ['Path Loss', 'Shadowing', 'Rayleigh Fading', 'Overall Channel']
    pl_contribution = path_loss_db
    shadowing_contribution = shadowing
    fading_contribution = 20 * np.log10(fading)
    overall_gain = h_db

    ax3.boxplot([pl_contribution, shadowing_contribution, fading_contribution, overall_gain],
                labels=components)
    ax3.set_ylabel('Magnitude (dB)')
    ax3.set_title('Channel Component Contributions')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

    ax4.scatter(r, h_db, alpha=0.6, s=20, color='green')
    ax4.set_xlabel('Distance from BS (m)')
    ax4.set_ylabel('Overall Channel Gain (dB)')
    ax4.set_title('Distance vs Overall Channel Gain')
    ax4.grid(True, alpha=0.3)

    stats_text = f"""
    Statistics:
    Mean distance: {r.mean():.0f} m
    Mean path loss: {path_loss_db.mean():.1f} dB
    Shadowing std: {shadowing.std():.1f} dB
    Channel gain range: {h_db.min():.1f} to {h_db.max():.1f} dB
    """
    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{results_dir}/channel_components_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_user_positioning_analysis():
    """Plot analysis of user positioning and spatial characteristics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    ax1.hist(r, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Distance from BS (m)')
    ax1.set_ylabel('Number of Users')
    ax1.set_title('Radial Distribution of Users')
    ax1.grid(True, alpha=0.3)

    ax2.hist(theta, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Angle (radians)')
    ax2.set_ylabel('Number of Users')
    ax2.set_title('Angular Distribution of Users')
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 2, 3, projection='polar')
    scatter3 = ax3.scatter(theta, r, c=h_db, cmap='viridis', s=20, alpha=0.7)
    ax3.set_title('Polar View of User Distribution\n(Color = Channel Gain)')
    plt.colorbar(scatter3, ax=ax3, label='Channel Gain (dB)', shrink=0.8)

    scatter4 = ax4.scatter(theta * 180/np.pi, r, c=h_db, cmap='viridis', s=30, alpha=0.7)
    ax4.set_xlabel('Angle (degrees)')
    ax4.set_ylabel('Distance from BS (m)')
    ax4.set_title('User Positions (Angle vs Distance)')
    plt.colorbar(scatter4, ax=ax4, label='Channel Gain (dB)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{results_dir}/user_positioning_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_clustering_comparison():
    """Compare performance of all methods that wrote *_clustering.csv in results_dir."""
    method_files = [f for f in os.listdir(results_dir) if f.endswith('_clustering.csv')]
    if not method_files:
        print("Warning: No clustering CSV files found for comparison.")
        return
    methods = [os.path.splitext(f)[0].replace('_clustering', '') for f in method_files]

    results = {}
    for method, file in zip(methods, method_files):
        df = pd.read_csv(os.path.join(results_dir, file))
        noma_pairs = df[df['Mode'] == 'NOMA']
        oma_users = df[df['Mode'] == 'OMA']
        results[method] = {
            'total_throughput': df['Throughput_Mbps'].sum(),
            'noma_pairs': len(noma_pairs),
            'oma_users': len(oma_users),
            'avg_noma_rate': noma_pairs['R_sum_bitsHz'].mean() if len(noma_pairs) > 0 else 0,
            'avg_oma_rate': oma_users['R_sum_bitsHz'].mean() if len(oma_users) > 0 else 0
        }

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    methods_list = list(results.keys())
    throughputs = [results[m]['total_throughput'] for m in methods_list]
    bars1 = ax1.bar(methods_list, throughputs)
    ax1.set_ylabel('Total Throughput (Mbps)')
    ax1.set_title('Total System Throughput Comparison')
    ax1.grid(True, alpha=0.3)
    for bar, value in zip(bars1, throughputs):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01, f'{value:.1f}',
                 ha='center', va='bottom', fontweight='bold')

    noma_counts = [results[m]['noma_pairs'] for m in methods_list]
    oma_counts = [results[m]['oma_users'] for m in methods_list]
    x = np.arange(len(methods_list)); width = 0.35
    ax2.bar(x - width/2, noma_counts, width, label='NOMA Pairs')
    ax2.bar(x + width/2, oma_counts, width, label='OMA Users')
    ax2.set_xlabel('Clustering Method'); ax2.set_ylabel('Number of Users/Pairs')
    ax2.set_title('NOMA vs OMA User Distribution')
    ax2.set_xticks(x); ax2.set_xticklabels(methods_list, rotation=15)
    ax2.legend(); ax2.grid(True, alpha=0.3)

    avg_noma_rates = [results[m]['avg_noma_rate'] for m in methods_list]
    avg_oma_rates  = [results[m]['avg_oma_rate'] for m in methods_list]
    ax3.bar(x - width/2, avg_noma_rates, width, label='Avg NOMA Rate')
    ax3.bar(x + width/2, avg_oma_rates,  width, label='Avg OMA Rate')
    ax3.set_xlabel('Clustering Method'); ax3.set_ylabel('Average Rate (bits/Hz)')
    ax3.set_title('Average Data Rates Comparison')
    ax3.set_xticks(x); ax3.set_xticklabels(methods_list, rotation=15)
    ax3.legend(); ax3.grid(True, alpha=0.3)

    ax4.pie(throughputs, labels=methods_list, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Throughput Share by Method')

    plt.tight_layout()
    plt.savefig(f'{results_dir}/clustering_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    summary_data = []
    for m in methods_list:
        summary_data.append({
            'Method': m,
            'Total_Throughput_Mbps': results[m]['total_throughput'],
            'NOMA_Pairs': results[m]['noma_pairs'],
            'OMA_Users': results[m]['oma_users'],
            'Avg_NOMA_Rate_bitsHz': results[m]['avg_noma_rate'],
            'Avg_OMA_Rate_bitsHz': results[m]['avg_oma_rate'],
            'NOMA_Percentage': (results[m]['noma_pairs'] * 2) / N * 100
        })
    pd.DataFrame(summary_data).to_csv(f'{results_dir}/clustering_summary.csv', index=False)

# -------------------- SIC Rate Calculation --------------------
def calc_pair_rate(h1, h2):
    P1, P2 = total_power * h2 / (h1 + h2), total_power * h1 / (h1 + h2)
    R1 = np.log2(1 + (P1 * h1) / (P2 * h1 + noise_power))
    R2 = np.log2(1 + (P2 * h2) / noise_power)
    return P1, P2, R1, R2, R1 + R2

# -------------------- SIC Condition Check --------------------
def sic_satisfied(h1, h2):
    return 10 * np.log10(h2 / h1) >= sic_threshold_db

# ==================== NEW: Angle helper + 1-D power optimizer ====================
def angle_diff_rad(a, b):
    """Smallest absolute angle difference on circle [0, 2π)."""
    d = np.abs(a - b) % (2*np.pi)
    return np.minimum(d, 2*np.pi - d)

def _pair_rates_given_P1(h1, h2, P1):
    """Rates for a given split P1 (P2 = total_power - P1). h1<=h2 assumed."""
    P1 = np.clip(P1, 0.0, total_power)
    P2 = total_power - P1
    R1 = np.log2(1.0 + (P1 * h1) / (P2 * h1 + noise_power))
    R2 = np.log2(1.0 + (P2 * h2) / noise_power)
    return R1, R2, R1 + R2, P1, P2

def optimize_pair_power(h1, h2, objective='sum', w1=1.0, w2=1.0,
                        tol=POWER_OPT_TOL, max_iter=POWER_OPT_MAXIT):
    """
    1-D golden-section search to maximize:
      - 'sum' : R1 + R2
      - 'pf'  : log(R1) + log(R2)  (PF; uses PF_EPS)
    Assumes h1 <= h2 and P1 in (0, total_power).
    """
    lo = 1e-9
    hi = total_power - 1e-9
    gr = (np.sqrt(5) - 1) / 2  # ~0.618

    def U(P1):
        R1, R2, Rsum, _, _ = _pair_rates_given_P1(h1, h2, P1)
        if objective == 'pf':
            return np.log(R1 + PF_EPS) + np.log(R2 + PF_EPS)
        else:
            return Rsum

    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    uc, ud = U(c), U(d)

    it = 0
    while (hi - lo) > tol and it < max_iter:
        if uc < ud:
            lo = c
            c = d
            uc = ud
            d = lo + gr * (hi - lo)
            ud = U(d)
        else:
            hi = d
            d = c
            ud = uc
            c = hi - gr * (hi - lo)
            uc = U(c)
        it += 1

    P1_opt = 0.5 * (lo + hi)
    R1, R2, Rsum, P1_opt, P2_opt = _pair_rates_given_P1(h1, h2, P1_opt)
    return P1_opt, P2_opt, R1, R2, Rsum
# ================================================================================

# -------------------- Clustering Common Function (UPDATED) --------------------
def perform_clustering(pairs_indices, name, power_opt=False, objective='sum', w1=1.0, w2=1.0):
    """
    Perform clustering with given pair indices and save results with visualizations

    Args:
        pairs_indices: List of tuples representing user pairs
        name: String identifier for the clustering method
        power_opt: If True, run 1-D power optimization per NOMA pair
        objective: 'sum' or 'pf' (used when power_opt=True)
        w1, w2: utility weights (usually 1,1)

    Returns:
        Dictionary containing performance metrics
    """
    data, used = [], np.zeros(N, bool)
    total_rate = 0
    noma_pairs_count = 0

    print(f"\n--- {name.title()} Clustering ---")
    print(f"Evaluating {len(pairs_indices)} potential pairs...")

    for u1, u2 in tqdm(pairs_indices, desc=f"Processing {name} pairs"):
        h1u, h2u = h_values[u1], h_values[u2]
        if h1u <= h2u:
            a, b = u1, u2
            h1, h2 = h1u, h2u
        else:
            a, b = u2, u1
            h1, h2 = h2u, h1u

        if not sic_satisfied(h1, h2):
            continue

        if power_opt:
            P1, P2, R1, R2, R_sum = optimize_pair_power(h1, h2, objective=objective, w1=w1, w2=w2)
        else:
            P1, P2, R1, R2, R_sum = calc_pair_rate(h1, h2)

        used[[a, b]] = True
        data.append([a, b, h1, h2, P1, P2, R1, R2, R_sum, "NOMA"])
        total_rate += R_sum
        noma_pairs_count += 1

    num_pairs = noma_pairs_count
    num_oma = int(np.sum(~used))
    B_unit = B_total / (num_pairs + num_oma) if (num_pairs + num_oma) > 0 else 0.0
    throughput_total = 0.0

    # NOMA throughput
    for row in data:
        row.append(row[8] * B_unit / 1e6)
        throughput_total += row[-1]

    # OMA users
    oma_users_count = 0
    for u in range(N):
        if not used[u]:
            h = h_values[u]
            R1_oma = np.log2(1 + total_power * h / noise_power)
            throughput = R1_oma * B_unit / 1e6
            throughput_total += throughput
            data.append([u, -1, h, 0, total_power, 0, R1_oma, 0, R1_oma, "OMA", throughput])
            oma_users_count += 1

    # Save results
    save_name = f"{results_dir}/{name}_clustering.csv"
    cols = ["User1_ID", "User2_ID", "h1", "h2", "P1", "P2", "R1_bitsHz", "R2_bitsHz", "R_sum_bitsHz", "Mode", "Throughput_Mbps"]
    pd.DataFrame(data, columns=cols).to_csv(save_name, index=False)

    # Print performance summary
    print(f"NOMA Pairs: {noma_pairs_count}")
    print(f"OMA Users: {oma_users_count}")
    print(f"Total Throughput: {throughput_total:.2f} Mbps")
    print(f"NOMA Coverage: {(noma_pairs_count * 2)/N*100:.1f}%")

    # Generate pairing visualization
    plot_pairing_visualization(pairs_indices, name, data)

    return {
        'noma_pairs': noma_pairs_count,
        'oma_users': oma_users_count,
        'total_throughput': throughput_total,
        'noma_coverage': (noma_pairs_count * 2)/N*100
    }

# ==================== PF-weighted strong–weak bipartite matching ====================
def build_bipartite_pf_matching(theta_min_deg=THETA_MIN_DEG):
    """
    Build a bipartite graph (weak ↔ strong) with PF weights and angular guard,
    then run max-weight matching. Uses fixed half-half split.
    Returns list of (u, v) matched indices.
    """
    theta_min_rad = np.deg2rad(theta_min_deg)

    weak = list(sorted_indices[:N//2])
    strong = list(sorted_indices[N//2:])

    G = nx.Graph()
    for i in weak:
        for j in strong:
            # Angular guard
            if angle_diff_rad(theta[i], theta[j]) < theta_min_rad:
                continue
            # SIC guard
            h1, h2 = (h_values[i], h_values[j]) if h_values[i] <= h_values[j] else (h_values[j], h_values[i])
            if not sic_satisfied(h1, h2):
                continue
            # PF weight using seed rates from baseline split
            _, _, R1, R2, _ = calc_pair_rate(h1, h2)
            w = np.log(R1 + PF_EPS) + np.log(R2 + PF_EPS)
            if np.isfinite(w):
                G.add_edge(i, j, weight=w)

    matching = nx.max_weight_matching(G, maxcardinality=True)
    return list(matching)



# -------------------- Main Execution with Visualizations --------------------
print("="*60)
print("NOMA CLUSTERING SIMULATION AND ANALYSIS")
print("="*60)
print(f"System Parameters:")
print(f"- Number of Users: {N}")
print(f"- Cell Radius: {radius} m")
print(f"- Carrier Frequency: {fc/1e9:.1f} GHz")
print(f"- SIC Threshold: {sic_threshold_db} dB")
print(f"- Total Bandwidth: {B_total/1e6:.0f} MHz")

# Generate initial visualizations
print("\nGenerating channel analysis plots...")
plot_user_distribution()
plot_channel_characteristics()
plot_channel_components_analysis()
plot_user_positioning_analysis()

# Store results for comparison
clustering_results = {}

# -------------------- Static Clustering --------------------
print("\n" + "="*50)
static_indices = [(sorted_indices[i], sorted_indices[N-1-i]) for i in range(N//2)]
clustering_results['static'] = perform_clustering(static_indices, "static")

# -------------------- Balanced Clustering --------------------
print("\n" + "="*50)
balanced_indices = [(sorted_indices[i], sorted_indices[i + N//2]) for i in range(N//2)]
clustering_results['balanced'] = perform_clustering(balanced_indices, "balanced")

# -------------------- Blossom Clustering --------------------
print("\n" + "="*50)
print("Blossom Clustering")
print("Building maximum weight matching graph...")
G = nx.Graph()
edge_count = 0

for i in tqdm(range(N), desc="Building graph"):
    for j in range(i+1, N):
        if sic_satisfied(min(h_values[i], h_values[j]), max(h_values[i], h_values[j])):
            _, _, _, _, R_sum = calc_pair_rate(min(h_values[i], h_values[j]), max(h_values[i], h_values[j]))
            G.add_edge(i, j, weight=R_sum)
            edge_count += 1

print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

blossom_indices = list(nx.max_weight_matching(G, maxcardinality=True))
clustering_results['blossom'] = perform_clustering(blossom_indices, "blossom")

# -------------------- PF Bipartite (strong-weak) - Original --------------------
print("\n" + "="*50)
print("PF Bipartite (strong-weak) Matching with Angular Guard - Original")
bp_pairs = build_bipartite_pf_matching(theta_min_deg=THETA_MIN_DEG)
print(f"PF Bipartite candidate matches: {len(bp_pairs)}")
clustering_results['bipartite_pf'] = perform_clustering(
    bp_pairs, "bipartite_pf", power_opt=True, objective='pf'
)



# -------------------- Generate Comparison Plots --------------------
print("\n" + "="*50)
print("Generating comparison analysis...")
plot_clustering_comparison()

# -------------------- Final Summary --------------------
print("\n" + "="*60)
print("SIMULATION SUMMARY")
print("="*60)

for method, results in clustering_results.items():
    print(f"\n{method.upper().replace('_', ' ')} CLUSTERING:")
    print(f"  - NOMA Pairs: {results['noma_pairs']}")
    print(f"  - OMA Users: {results['oma_users']}")
    print(f"  - Total Throughput: {results['total_throughput']:.2f} Mbps")
    print(f"  - NOMA Coverage: {results['noma_coverage']:.1f}%")


# Find best performing method
best_method = max(clustering_results.keys(),
                 key=lambda x: clustering_results[x]['total_throughput'])
print(f"\nBEST PERFORMING METHOD: {best_method.upper().replace('_', ' ')}")
print(f"Throughput: {clustering_results[best_method]['total_throughput']:.2f} Mbps")



print(f"\nAll results and visualizations saved to '{results_dir}/' directory")
print("Simulation completed successfully!")
print("="*60)