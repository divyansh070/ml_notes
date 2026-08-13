import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Input Feature Map (4x4)
input_matrix = np.array([
    [0, 0, 0, 8],
    [0, 2, 0, 0],
    [0, 0, 5, 0],
    [1, 0, 0, 0]
], dtype=float)

# 2. Operations (Stride=2, Pool Size=2x2)
max_pooled = np.zeros((2, 2))
avg_pooled = np.zeros((2, 2))

for i in range(2):
    for j in range(2):
        window = input_matrix[i*2:i*2+2, j*2:j*2+2]
        max_pooled[i, j] = np.max(window)
        avg_pooled[i, j] = np.mean(window)

# 3. Visualization Setup
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Common heatmap settings
cmap = 'Blues'
annot_settings = {"size": 20, "color": "black", "weight": "bold"}

# Subplot 1: Input Matrix
sns.heatmap(input_matrix, annot=True, cmap=cmap, cbar=False, ax=axes[0], 
            annot_kws=annot_settings, fmt='g', linewidths=1, linecolor='gray', vmin=0, vmax=8)
axes[0].set_title("Original Feature Map (4x4)", pad=20, fontsize=16, fontweight='bold')
# Overlay faint 2x2 grid to show windows
axes[0].axhline(2, color='red', linewidth=4, linestyle='--')
axes[0].axvline(2, color='red', linewidth=4, linestyle='--')
axes[0].set_xticks([])
axes[0].set_yticks([])

# Subplot 2: Max Pooled
sns.heatmap(max_pooled, annot=True, cmap=cmap, cbar=False, ax=axes[1], 
            annot_kws=annot_settings, fmt='g', linewidths=1, linecolor='gray', vmin=0, vmax=8)
axes[1].set_title("Max Pooling (2x2, Stride 2)\nIsolates the Strongest Activations", pad=20, fontsize=16, fontweight='bold')
axes[1].set_xticks([])
axes[1].set_yticks([])

# Subplot 3: Average Pooled
sns.heatmap(avg_pooled, annot=True, cmap=cmap, cbar=False, ax=axes[2], 
            annot_kws=annot_settings, fmt='g', linewidths=1, linecolor='gray', vmin=0, vmax=8)
axes[2].set_title("Average Pooling (2x2, Stride 2)\nDilutes the Signal", pad=20, fontsize=16, fontweight='bold')
axes[2].set_xticks([])
axes[2].set_yticks([])

plt.tight_layout()
plt.suptitle("Max Pooling vs. Average Pooling", fontsize=22, fontweight='bold', y=1.08)
plt.savefig("assets/pooling_visual.png", dpi=300, bbox_inches='tight')
print("Successfully generated assets/pooling_visual.png")
