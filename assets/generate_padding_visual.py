import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# 1. Create the matrices
# A tiny 3x3 input image of all 1s
original_image = np.ones((3, 3), dtype=int)
# Apply p=1 zero padding
padded_image = np.pad(original_image, pad_width=1, mode='constant', constant_values=0)

# 2. Setup the custom colormap (0 = gray, 1 = blue)
# This perfectly visually separates the artificial border from the true image
cmap = ListedColormap(['#e0e0e0', '#63b8ff']) # Light gray and soft blue

# 3. Visualization Setup
fig, ax = plt.subplots(figsize=(7, 7))

# Plot the 5x5 padded matrix
sns.heatmap(padded_image, annot=True, cmap=cmap, cbar=False, ax=ax,
            annot_kws={"size": 28, "color": "black", "weight": "bold"}, 
            fmt='d', linewidths=3, linecolor='white', square=True)

# Formatting to make it clean
ax.set_xticks([])
ax.set_yticks([])

# 4. Titles and Subtitles
title = "SAME Padding (p=1): Preserving Spatial Dimensions"
subtitle = "The artificial border of 0s allows a 3x3 filter to scan the extreme edges and corners\nof the true image (blue) without mathematically altering the feature detection.\nThis prevents the output feature map from shrinking."

plt.suptitle(title, fontsize=18, fontweight='bold', y=1.08)
plt.title(subtitle, fontsize=13, pad=15, color='#333333')

plt.tight_layout()
plt.savefig("assets/padding_visual.png", dpi=300, bbox_inches='tight')
print("Successfully generated assets/padding_visual.png")
