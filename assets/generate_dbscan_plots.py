import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN

# 1. Generate Dataset (Nested moons where K-Means fails)
X, y = make_moons(n_samples=500, noise=0.05, random_state=42)

# 2. Setup Figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plt.subplots_adjust(hspace=0.3)

# ==========================================
# Plot 1: K-Means Failure
# ==========================================
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

ax1 = axes[0, 0]
ax1.scatter(X[kmeans_labels == 0, 0], X[kmeans_labels == 0, 1], c='blue', s=20, label='Cluster 1')
ax1.scatter(X[kmeans_labels == 1, 0], X[kmeans_labels == 1, 1], c='orange', s=20, label='Cluster 2')
ax1.set_title("1. The K-Means Failure\n(Assumes spherical clusters)")
ax1.legend()
ax1.set_xticks([])
ax1.set_yticks([])

# ==========================================
# Run DBSCAN
# ==========================================
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# Identify point types mathematically
core_mask = np.zeros_like(dbscan_labels, dtype=bool)
core_mask[dbscan.core_sample_indices_] = True
outlier_mask = dbscan_labels == -1
border_mask = ~(core_mask | outlier_mask)

# ==========================================
# Plot 2: Point Classification (Setup)
# ==========================================
ax2 = axes[0, 1]
# Plot order: Outliers -> Borders -> Cores (to layer cores on top)
ax2.scatter(X[outlier_mask, 0], X[outlier_mask, 1], c='red', s=40, label='Noise (Outliers)', marker='x')
ax2.scatter(X[border_mask, 0], X[border_mask, 1], c='yellow', s=40, edgecolor='k', label='Border Points', linewidth=0.5)
ax2.scatter(X[core_mask, 0], X[core_mask, 1], c='green', s=20, label='Core Points')
ax2.set_title("2. Point Classification (Setup)\neps=0.2, min_samples=5")
ax2.legend()
ax2.set_xticks([])
ax2.set_yticks([])

# ==========================================
# Plot 3: Cluster Expansion (The Core)
# ==========================================
ax3 = axes[1, 0]
# Gray out non-core points (borders + noise)
ax3.scatter(X[~core_mask, 0], X[~core_mask, 1], c='lightgray', s=15, alpha=0.5, label='Non-Core Points')
# Plot core points colored by cluster
ax3.scatter(X[(core_mask) & (dbscan_labels == 0), 0], X[(core_mask) & (dbscan_labels == 0), 1], c='blue', s=30, label='Core Cluster 1')
ax3.scatter(X[(core_mask) & (dbscan_labels == 1), 0], X[(core_mask) & (dbscan_labels == 1), 1], c='orange', s=30, label='Core Cluster 2')
ax3.set_title("3. Cluster Expansion\n(Connecting dense Core Points)")
ax3.legend()
ax3.set_xticks([])
ax3.set_yticks([])

# ==========================================
# Plot 4: Final DBSCAN Result
# ==========================================
ax4 = axes[1, 1]
# Plot outliers as tiny black dots
ax4.scatter(X[outlier_mask, 0], X[outlier_mask, 1], c='black', s=15, label='Outliers')
# Plot fully formed clusters (Core + Border)
ax4.scatter(X[dbscan_labels == 0, 0], X[dbscan_labels == 0, 1], c='blue', s=30, label='Cluster 1')
ax4.scatter(X[dbscan_labels == 1, 0], X[dbscan_labels == 1, 1], c='orange', s=30, label='Cluster 2')
ax4.set_title("4. Final DBSCAN Result\n(Borders joined, Outliers ignored)")
ax4.legend()
ax4.set_xticks([])
ax4.set_yticks([])

plt.tight_layout()
plt.savefig("assets/dbscan_step_by_step.png", dpi=300, bbox_inches='tight')
print("Successfully generated: assets/dbscan_step_by_step.png")
