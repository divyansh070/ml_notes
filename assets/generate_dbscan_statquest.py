import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

# --- STATQUEST COLORS ---
C_GREEN = '#2ca02c'
C_BLUE = '#3498db'
C_RED = '#e74c3c'
C_ORANGE = '#f39c12'
C_GREY = '#bdc3c7'
C_BLACK = '#2c3e50'

# HARDCODED TOY DATASET (Nested Clusters + Outliers)
# Cluster 1: Outer C-shape
g_pts = np.array([
    [1, 5], [1, 4], [1, 3], [1, 2], [1, 1], 
    [2, 1], [3, 1], [4, 1], [5, 1],
    [2, 5], [3, 5], [4, 5], [5, 5],
    [1.5, 4.5], [1.5, 1.5], [4.5, 1.5], [4.5, 4.5] 
])
# Cluster 2: Inner Blob
b_pts = np.array([
    [3, 3], [4, 3], [3, 2.5], [4, 2.5], [3.5, 2.75]
])
# Outliers
o_pts = np.array([
    [7, 7], [6, -1], [-1, 3]
])
all_pts = np.vstack((g_pts, b_pts, o_pts))

# Figure Setup
fig, axs = plt.subplots(2, 2, figsize=(14, 11))
fig.patch.set_facecolor('#f8f9fa')

for ax in axs.flat:
    ax.set_xlim(-2, 8)
    ax.set_ylim(-2, 8)
    ax.axis('off')

# --- Panel 1: The Orange Circle ---
ax = axs[0, 0]
ax.scatter(all_pts[:, 0], all_pts[:, 1], color=C_GREY, s=100, zorder=2)
target = np.array([1, 3])
# Draw orange circle (radius = 1.5)
circle = Circle(target, 1.5, facecolor=C_ORANGE, edgecolor=C_ORANGE, alpha=0.3, zorder=1)
ax.add_patch(circle)
ax.scatter(target[0], target[1], color=C_RED, s=120, edgecolor='black', zorder=3)
# Highlight neighbors
neighbors = np.array([[1, 4], [1, 2], [1.5, 4.5], [1.5, 1.5]]) 
ax.scatter(neighbors[:, 0], neighbors[:, 1], color='#c0392b', s=100, zorder=3)
ax.text(3.5, 6, "Step 1: The Orange Circle", fontsize=16, fontweight='bold')
ax.text(3.5, 4.5, "Draw an Orange Circle around a point.\nCount how many points it overlaps.\nThis red point is close to 4 other points.", fontsize=12, bbox=dict(facecolor='white', edgecolor=C_ORANGE, alpha=0.9))

# --- Panel 2: Core vs Non-Core ---
ax = axs[0, 1]
ax.text(3.5, 6, "Step 2: Core Points vs. Non-Core", fontsize=16, fontweight='bold')
ax.text(3.5, 4.5, "Core Point: Overlaps $\geq 4$ points.\nNon-Core Point: Overlaps $< 4$ points.\nRed = Core, Black = Non-Core", fontsize=12, bbox=dict(facecolor='white', edgecolor=C_RED, alpha=0.9))
# Visually designate core vs non-core
core_pts = np.vstack((g_pts[:-4], b_pts))
non_core = np.vstack((g_pts[-4:], o_pts))
ax.scatter(core_pts[:, 0], core_pts[:, 1], color=C_RED, s=100, edgecolor='black', zorder=3)
ax.scatter(non_core[:, 0], non_core[:, 1], color=C_BLACK, s=80, zorder=2)

# --- Panel 3: Growing the Cluster ---
ax = axs[1, 0]
ax.text(3.5, 6, "Step 3: Extending the Cluster", fontsize=16, fontweight='bold')
ax.text(3.5, 4.5, "Pick a Core Point to start.\nNeighboring Core Points JOIN and EXTEND.\nNon-Core points JOIN but CANNOT extend.", fontsize=12, bbox=dict(facecolor='white', edgecolor=C_GREEN, alpha=0.9))
ax.scatter(core_pts[:, 0], core_pts[:, 1], color=C_RED, s=100, edgecolor='black', zorder=3)
ax.scatter(non_core[:, 0], non_core[:, 1], color=C_BLACK, s=80, zorder=2)
# Highlight the growing green cluster
green_grown = g_pts[:9]
ax.scatter(green_grown[:, 0], green_grown[:, 1], color=C_GREEN, s=120, edgecolor='black', zorder=4)
# Point out a non-core point joining
joining_non_core = np.array([[1.5, 1.5]])
ax.scatter(joining_non_core[:, 0], joining_non_core[:, 1], color=C_GREEN, s=80, marker='s', edgecolor='black', zorder=4)
ax.annotate('Non-Core\nPoint joins!', xy=(1.5, 1.5), xytext=(2.5, 0),
            arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
            fontsize=12, fontweight='bold')

# --- Panel 4: Final Clusters ---
ax = axs[1, 1]
ax.text(3.5, 6, "Step 4: Final Clusters & Outliers", fontsize=16, fontweight='bold')
ax.text(3.5, 4.5, "BAM! Nested clusters correctly identified.\nRemaining Non-Core points are Outliers.", fontsize=12, bbox=dict(facecolor='white', edgecolor=C_BLUE, alpha=0.9))
ax.scatter(g_pts[:, 0], g_pts[:, 1], color=C_GREEN, s=120, edgecolor='black', zorder=3, label='Cluster 1')
ax.scatter(b_pts[:, 0], b_pts[:, 1], color=C_BLUE, s=120, edgecolor='black', zorder=3, label='Cluster 2')
ax.scatter(o_pts[:, 0], o_pts[:, 1], color=C_BLACK, s=80, zorder=3, label='Outliers')
ax.legend(loc='upper right', fontsize=12)

plt.suptitle("StatQuest Style: Clustering with DBSCAN", fontsize=22, fontweight='bold', y=0.95)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig('assets/statquest_dbscan.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated assets/statquest_dbscan.png successfully!")
