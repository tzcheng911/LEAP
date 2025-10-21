import numpy as np
from scipy.stats import pearsonr
from scipy.ndimage import label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# --- Step 1: Prep the data
np.random.seed(42)
n_subjects, n_sensors = np.shape(MEG)[0], np.shape(MEG)[1]
neural = MEG
behavior = CDI.to_numpy()

print("Computing adjacency.")
stc1 = mne.read_source_estimate('/media/tzcheng/storage/BabyRhythm/br_03/sss_fif/br_03_01_stc_mne_morph_mag6pT-vl.stc')
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')
adjacency = mne.spatial_src_adjacency(src)
adjacency_sparse = csr_matrix(adjacency)
# --- Step 2: Compute observed correlations
r_vals = np.array([pearsonr(neural[:, i], behavior)[0] for i in range(n_sensors)])
t_vals = r_vals * np.sqrt((n_subjects - 2) / (1 - r_vals**2))

# --- Step 3: Threshold for cluster formation (e.g., |t| > critical t for p<0.05, df=n-2)
from scipy.stats import t
t_crit = t.ppf(1 - 0.025, df=n_subjects - 2)  # two-tailed
supra_thresh = np.abs(t_vals) > t_crit

# --- Step 4: Define clusters (1D case — adjacency = neighboring sensors)
# label contiguous True values as clusters
sub_adj = adjacency_sparse[supra_thresh][:, supra_thresh]
n_clusters, labels = connected_components(sub_adj, directed=False)
cluster_labels = np.zeros(n_sensors, dtype=int)
cluster_labels[supra_thresh] = labels + 1  # +1 to start cluster IDs at 1

# --- Step 5: Compute cluster statistics (sum of t-values within each cluster)
cluster_stats = np.array([np.sum(np.abs(t_vals)[cluster_labels == c + 1]) for c in range(n_clusters)])

# --- Step 6: Build null distribution with permutation
n_permutations = 500
max_cluster_stats = np.zeros(n_permutations)
for p in range(n_permutations):
    print("iteration " + str(p))
    y_perm = np.random.permutation(behavior)
    r_perm = np.array([pearsonr(neural[:, i], y_perm)[0] for i in range(n_sensors)])
    t_perm = r_perm * np.sqrt((n_subjects - 2) / (1 - r_perm**2))
    supra_perm = np.abs(t_perm) > t_crit
    sub_adj_perm = adjacency_sparse[supra_perm][:, supra_perm]
    perm_n_clusters, perm_labels = connected_components(sub_adj_perm, directed=False)
    cluster_labels_perm = np.zeros(n_sensors, dtype=int)
    cluster_labels_perm[supra_perm] = perm_labels + 1  # +1 to start cluster IDs at 1
    if perm_n_clusters > 0:
        perm_stats = np.array([np.sum(np.abs(t_perm)[cluster_labels_perm == c + 1]) for c in range(perm_n_clusters)])
        max_cluster_stats[p] = np.max(perm_stats)
    else:
        max_cluster_stats[p] = 0
        
# --- Step 7: Compare observed cluster stats to null (family-wise correction)
p_values = np.array([np.mean(max_cluster_stats >= s) for s in cluster_stats])

# --- Step 8: Print results
for i, (s, p) in enumerate(zip(cluster_stats, p_values)):
    print(f"Cluster {i+1}: sum(|t|)={s:.2f}, p={p:.4f}")

# --- Optional: visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.hist(max_cluster_stats, bins=50, color="gray", alpha=0.7)
plt.axvline(np.percentile(max_cluster_stats, 95))
for stat in cluster_stats:
    plt.axvline(stat, color="black", linestyle="--")
plt.xlabel("Cluster statistic")
plt.ylabel("Frequency")
plt.title("Permutation null distribution")
plt.show()