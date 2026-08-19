import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create assets dir if not exists
os.makedirs('/Users/divyanshverma/Desktop/ml_interview_questions/assets', exist_ok=True)

# Setup data
tokens = ["<BOS>", "Je", "suis", "un"]
raw_scores = np.array([
    [15, 4, 1, 0],
    [4, 12, 3, 2],
    [1, 3, 14, 0],
    [0, 2, 0, 13]
])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#f8f9fa')

# Colors
cmap1 = sns.light_palette("#3498db", as_cmap=True)

# Raw scores
sns.heatmap(raw_scores, annot=True, fmt="d", cmap=cmap1, cbar=False, ax=ax1, 
            xticklabels=tokens, yticklabels=tokens, annot_kws={"size": 16, "weight": "bold"})
ax1.set_title("1. Raw Attention Scores ($Q \cdot K^T$)", fontsize=18, fontweight='bold', pad=15)
ax1.tick_params(axis='both', labelsize=14)
ax1.set_ylabel("Query (Current Word)", fontsize=14, fontweight='bold', labelpad=10)
ax1.set_xlabel("Key (Attending To)", fontsize=14, fontweight='bold', labelpad=10)

# Masked scores
# Create a custom mask matrix for visualization: numbers where valid, NaN where -inf
masked_scores = raw_scores.astype(float)
mask = np.triu(np.ones_like(masked_scores, dtype=bool), k=1)
masked_scores[mask] = np.nan

# Plot masked heatmap
# We use a custom color map for the valid part, and gray for the NaN part
sns.heatmap(masked_scores, annot=True, fmt=".0f", cmap=cmap1, cbar=False, ax=ax2, 
            xticklabels=tokens, yticklabels=tokens, annot_kws={"size": 16, "weight": "bold"})
ax2.set_facecolor('#2c3e50') # Dark background for the masked out regions
ax2.set_title("2. Apply Causal Mask (Upper-Triangle = $-\infty$)", fontsize=18, fontweight='bold', pad=15)
ax2.tick_params(axis='both', labelsize=14)
ax2.set_ylabel("Query (Current Word)", fontsize=14, fontweight='bold', labelpad=10)
ax2.set_xlabel("Key (Attending To)", fontsize=14, fontweight='bold', labelpad=10)

# Manually add the -\infty text to the masked cells
for i in range(4):
    for j in range(4):
        if j > i:
            ax2.text(j + 0.5, i + 0.5, r'$-\infty$', ha='center', va='center', 
                     color='#e74c3c', fontsize=22, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/divyanshverma/Desktop/ml_interview_questions/assets/causal_mask_visualization.png', dpi=300, bbox_inches='tight')
print("Saved visualization to assets/causal_mask_visualization.png")
