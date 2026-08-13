import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# 1. Input Image (5x5)
# Vertical line of 1s in a background of 0s
input_image = np.array([
    [0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0]
])

# 2. Filter (3x3 Vertical Edge Detector)
filter_matrix = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# 3. Output Feature Map (3x3)
output_map = np.zeros((3, 3))

for i in range(3):
    for j in range(3):
        patch = input_image[i:i+3, j:j+3]
        output_map[i, j] = np.sum(patch * filter_matrix)

# 4. Visualization Setup
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Subplot 1: Input Image
sns.heatmap(input_image, annot=True, cmap='Blues', cbar=False, ax=axes[0], 
            annot_kws={"size": 16, "color": "black"}, fmt='g', linewidths=1, linecolor='gray')
axes[0].set_title("Input Image (5x5)", pad=20, fontsize=16, fontweight='bold')
# In seaborn, xy=(0,0) is the top left corner.
rect1 = patches.Rectangle(xy=(0, 0), width=3, height=3, fill=False, edgecolor='red', lw=4, zorder=10)
axes[0].add_patch(rect1)
axes[0].set_xticks([])
axes[0].set_yticks([])

# Subplot 2: Filter
sns.heatmap(filter_matrix, annot=True, cmap='coolwarm', cbar=False, ax=axes[1], 
            annot_kws={"size": 16, "color": "black"}, fmt='g', linewidths=1, linecolor='gray')
axes[1].set_title("Filter (3x3)\nVertical Edge Detector", pad=20, fontsize=16, fontweight='bold')
axes[1].set_xticks([])
axes[1].set_yticks([])

# Subplot 3: Output Feature Map
sns.heatmap(output_map, annot=True, cmap='Greens', cbar=False, ax=axes[2], 
            annot_kws={"size": 16, "color": "black"}, fmt='g', linewidths=1, linecolor='gray')
axes[2].set_title("Output Feature Map (3x3)", pad=20, fontsize=16, fontweight='bold')
rect3 = patches.Rectangle(xy=(0, 0), width=1, height=1, fill=False, edgecolor='red', lw=4, zorder=10)
axes[2].add_patch(rect3)
axes[2].set_xticks([])
axes[2].set_yticks([])

# 5. Text Annotation below plots
patch_0_0 = input_image[0:3, 0:3]
calculation_str = "Calculation for the Top-Left Patch (Red Box):\n\n"
for i in range(3):
    row_str = " + ".join([f"({patch_0_0[i,j]} × {filter_matrix[i,j]})" for j in range(3)])
    calculation_str += f"Row {i+1}: {row_str}\n"

calculation_str += f"\nTotal Sum = {int(output_map[0,0])}  -->  Placed at position (0,0) in the Output Feature Map."

plt.figtext(0.5, -0.2, calculation_str, ha="center", fontsize=15, 
            bbox={"facecolor": "white", "alpha": 1, "pad": 15, "edgecolor": "red", "linewidth": 2})

plt.tight_layout()
plt.savefig("assets/convolution_visual.png", dpi=300, bbox_inches='tight')
print("Successfully generated assets/convolution_visual.png")
