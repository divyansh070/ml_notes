import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN

# 1. Generate Dataset
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

# Manually append 15 obvious outliers
np.random.seed(42)
outliers = np.random.uniform(low=-1.5, high=2.5, size=(15, 2))
X = np.vstack([X, outliers])

# 2. Setup Figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.3)

# ==========================================
# Plot 1: K-Means Failure
# ==========================================
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

ax1 = axes[0, 0]
ax1.scatter(X[kmeans_labels == 0, 0], X[kmeans_labels == 0, 1], c='blue', s=20)
ax1.scatter(X[kmeans_labels == 1, 0], X[kmeans_labels == 1, 1], c='orange', s=20)
ax1.set_title("1. K-Means Failure")
ax1.set_xticks([])
ax1.set_yticks([])

# Add annotation pointing near the middle
ax1.annotate('Fails on nested/non-spherical shapes', 
             xy=(0.5, 0.25), xytext=(0.5, 1.2),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=11, ha='center')

# ==========================================
# Run DBSCAN
# ==========================================
eps = 0.15
dbscan = DBSCAN(eps=eps, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

core_mask = np.zeros_like(dbscan_labels, dtype=bool)
core_mask[dbscan.core_sample_indices_] = True
outlier_mask = dbscan_labels == -1
border_mask = ~(core_mask | outlier_mask)

# ==========================================
# Plot 2: Point Classification (Setup)
# ==========================================
ax2 = axes[0, 1]
ax2.scatter(X[outlier_mask, 0], X[outlier_mask, 1], c='red', s=40, label='Noise (Outliers)', marker='x')
ax2.scatter(X[border_mask, 0], X[border_mask, 1], c='yellow', s=40, edgecolor='k', label='Border Points', linewidth=0.5)
ax2.scatter(X[core_mask, 0], X[core_mask, 1], c='green', s=20, label='Core Points')
ax2.set_title(f"2. The Anatomy of DBSCAN (eps={eps})")
ax2.legend(loc='lower left')
ax2.set_xticks([])
ax2.set_yticks([])

# Pick specific points to annotate
noise_idx = np.where(outlier_mask)[0][0]
border_idx = np.where(border_mask)[0][3]
core_idx = np.where(core_mask)[0][10]

# Annotate Noise
ax2.annotate('Noise: 0 neighbors', 
             xy=(X[noise_idx, 0], X[noise_idx, 1]), xytext=(X[noise_idx, 0] - 0.5, X[noise_idx, 1] + 0.4),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
             fontsize=10)

# Annotate Border
ax2.annotate('Border: <5 neighbors,\nbut touches Core', 
             xy=(X[border_idx, 0], X[border_idx, 1]), xytext=(X[border_idx, 0] + 0.3, X[border_idx, 1] - 0.5),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
             fontsize=10)

# Annotate Core
ax2.annotate('Core: >=5 neighbors', 
             xy=(X[core_idx, 0], X[core_idx, 1]), xytext=(X[core_idx, 0] - 0.8, X[core_idx, 1] - 0.5),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
             fontsize=10)

# Draw eps circles
circle_core = Circle((X[core_idx, 0], X[core_idx, 1]), eps, color='green', fill=True, alpha=0.2)
ax2.add_patch(circle_core)

circle_border = Circle((X[border_idx, 0], X[border_idx, 1]), eps, color='yellow', fill=True, alpha=0.3)
ax2.add_patch(circle_border)

# ==========================================
# Plot 3: The Chain Reaction (Overlapping Radii)
# ==========================================
ax3 = axes[1, 0]
cluster_0_cores = X[(core_mask) & (dbscan_labels == 0)]
zoom_cores = cluster_0_cores[10:16] 

ax3.scatter(zoom_cores[:, 0], zoom_cores[:, 1], c='green', s=60, label='Core Points')
for pt in zoom_cores:
    circle = Circle((pt[0], pt[1]), eps, color='green', fill=True, alpha=0.15, edgecolor='black', linewidth=1)
    ax3.add_patch(circle)

x_min, x_max = zoom_cores[:, 0].min() - 0.2, zoom_cores[:, 0].max() + 0.2
y_min, y_max = zoom_cores[:, 1].min() - 0.2, zoom_cores[:, 1].max() + 0.2
ax3.set_xlim(x_min, x_max)
ax3.set_ylim(y_min, y_max)
ax3.set_title("3. The Chain Reaction (Zoomed In)")
ax3.set_xticks([])
ax3.set_yticks([])

ax3.annotate('Overlapping radii link core points', 
             xy=(zoom_cores[2, 0], zoom_cores[2, 1]), xytext=(x_min + 0.02, y_max - 0.05),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
             fontsize=11)

# ==========================================
# Plot 4: Final DBSCAN Result
# ==========================================
ax4 = axes[1, 1]
ax4.scatter(X[outlier_mask, 0], X[outlier_mask, 1], c='black', s=15, label='Outliers')
ax4.scatter(X[dbscan_labels == 0, 0], X[dbscan_labels == 0, 1], c='blue', s=30, label='Cluster 1')
ax4.scatter(X[dbscan_labels == 1, 0], X[dbscan_labels == 1, 1], c='orange', s=30, label='Cluster 2')
ax4.set_title("4. Final Result")
ax4.legend(loc='lower left')
ax4.set_xticks([])
ax4.set_yticks([])

ax4.annotate('Outliers isolated and ignored', 
             xy=(X[noise_idx, 0], X[noise_idx, 1]), xytext=(X[noise_idx, 0] - 0.8, X[noise_idx, 1] + 0.5),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
             fontsize=11)

plt.tight_layout()
plt.savefig("assets/dbscan_step_by_step.png", dpi=300, bbox_inches='tight')
print("Successfully generated highly annotated plot: assets/dbscan_step_by_step.png")
