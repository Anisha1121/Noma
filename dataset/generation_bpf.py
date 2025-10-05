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
N, radius = 500, 5000  # Users, cell radius (m)
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
print("BIPARTITE PF MATCHING - STANDALONE")
print("="*60)
print(f"System Parameters:")
print(f"- Number of Users: {N}")
print(f"- Cell Radius: {radius} m")
print(f"- Carrier Frequency: {fc/1e9:.1f} GHz")
print(f"- SIC Threshold: {sic_threshold_db} dB")
print(f"- Total Bandwidth: {B_total/1e6:.0f} MHz")



# Store results for comparison
clustering_results = {}







# -------------------- PF Bipartite (strong-weak) - Original --------------------
print("\n" + "="*50)
print("PF Bipartite (strong-weak) Matching with Angular Guard - Original")
bp_pairs = build_bipartite_pf_matching(theta_min_deg=THETA_MIN_DEG)
print(f"PF Bipartite candidate matches: {len(bp_pairs)}")
clustering_results['bipartite_pf'] = perform_clustering(
    bp_pairs, "bipartite_pf", power_opt=True, objective='pf'
)





# -------------------- Final Summary --------------------
print(f"\n{'='*60}")
print("BIPARTITE MATCHING RESULTS")
print("="*60)
print(f"NOMA Pairs: {clustering_results['bipartite_pf']['noma_pairs']}")
print(f"OMA Users: {clustering_results['bipartite_pf']['oma_users']}")
print(f"Total Throughput: {clustering_results['bipartite_pf']['total_throughput']:.2f} Mbps")
print(f"NOMA Coverage: {clustering_results['bipartite_pf']['noma_coverage']:.1f}%")



print(f"\nResults saved to: {results_dir}/bipartite_pf_clustering.csv")
print(f"User data saved to: {results_dir}/h_values.csv")
print("Bipartite matching completed successfully!")
print("="*60)