import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('/Users/divyanshverma/Desktop/ml_interview_questions/assets', exist_ok=True)

# English Source
encoder_tokens = ["European", "Economic", "Area"]
# French Target
decoder_tokens = ["Zone", "Économique", "Européenne"]

# The cross attention alignment matrix (Decoder Q dot Encoder K)
# French "Zone" aligns with "Area"
# French "Économique" aligns with "Economic"
# French "Européenne" aligns with "European"
attention_weights = np.array([
    [0.05, 0.10, 0.85], # Zone -> Area
    [0.10, 0.80, 0.10], # Économique -> Economic
    [0.90, 0.05, 0.05], # Européenne -> European
])

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#f8f9fa')

cmap = sns.light_palette("#2ecc71", as_cmap=True)

sns.heatmap(attention_weights, annot=True, fmt=".2f", cmap=cmap, cbar=False, ax=ax,
            xticklabels=encoder_tokens, yticklabels=decoder_tokens, 
            annot_kws={"size": 16, "weight": "bold"})

ax.set_title("Cross-Attention Alignment Matrix", fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel("Encoder Keys (English Source)", fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel("Decoder Queries (French Target)", fontsize=14, fontweight='bold', labelpad=10)
ax.tick_params(axis='both', labelsize=14)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('/Users/divyanshverma/Desktop/ml_interview_questions/assets/cross_attention_heatmap.png', dpi=300, bbox_inches='tight')
