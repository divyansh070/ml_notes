import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 1. Define the 3D Semantic Space Coordinates
# [X (Financial), Y (Nature), Z (Neutral Dimension)]
bank_raw = np.array([0.9, 0.1, 0.4])   # High Finance, Low Nature
river_raw = np.array([0.1, 0.9, 0.5])  # Low Finance, High Nature

# Attention Math: 10% self, 90% River
bank_contextualized = (0.1 * bank_raw) + (0.9 * river_raw) 
# Results in roughly [0.18, 0.82, 0.49] -> Pulled away from Finance, towards Nature!

# 2. Plot the Points
ax.scatter(*bank_raw, color='red', s=150, edgecolor='black', label='Raw "Bank"', zorder=5)
ax.scatter(*river_raw, color='green', s=150, edgecolor='black', label='Raw "River"', zorder=5)
ax.scatter(*bank_contextualized, color='blue', s=150, edgecolor='black', label='New Contextualized "Bank"', zorder=5)

# 3. Draw the "Pull" of Self-Attention (Quiver Arrow)
# Calculate the direction vector from Raw Bank to Contextualized Bank
u, v, w = bank_contextualized - bank_raw
ax.quiver(*bank_raw, u, v, w, color='blue', arrow_length_ratio=0.1, lw=2.5, ls='--')

# 4. Text Annotations next to dots
ax.text(bank_raw[0], bank_raw[1], bank_raw[2] + 0.05, 'Raw "Bank"\n(Financial)', color='red', fontweight='bold', ha='center')
ax.text(river_raw[0], river_raw[1], river_raw[2] + 0.05, 'Raw "River"\n(Nature)', color='green', fontweight='bold', ha='center')
ax.text(bank_contextualized[0], bank_contextualized[1], bank_contextualized[2] + 0.05, 'Contextualized\n"Bank"', color='blue', fontweight='bold', ha='center')

# 5. Axis Labels to Define the Space
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xlabel('Financial Meaning Axis', fontweight='bold', labelpad=10)
ax.set_ylabel('Nature Meaning Axis', fontweight='bold', labelpad=10)
ax.set_zlabel('Neutral/Other Dimension', fontweight='bold', labelpad=10)

# 6. Explanatory Box
explanation = (
    "Self-Attention as Geometric Movement:\n"
    "1. The raw word 'Bank' starts in the 'Financial' corner of the embedding space.\n"
    "2. The Attention matrix calculates a 90% correlation with the word 'River'.\n"
    "3. The Matrix Multiplication physically pulls the 'Bank' vector across the 3D space.\n"
    "4. The new Contextualized 'Bank' vector now sits closely next to 'River' in the Nature space!"
)
ax.text2D(0.5, -0.1, explanation, transform=ax.transAxes, ha='center', va='top', 
          fontsize=12, fontweight='bold', bbox=dict(facecolor='#e8f8f5', edgecolor='teal', pad=1))

plt.title("CampusX Style: Self-Attention in 3D Semantic Space", fontsize=18, fontweight='bold', y=1.05)
plt.legend(loc='upper right')
plt.savefig('assets/vectors_before_after_attention.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated assets/vectors_before_after_attention.png successfully!")
