import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Matrices
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

dY = np.array([[2, 0],
               [0, -1]])

dW = np.array([[-3, -2],
               [0, 1]])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Common heatmap settings
kwargs = {'annot': True, 'fmt': 'd', 'cbar': False, 
          'annot_kws': {'size': 20, 'weight': 'bold'}, 
          'linewidths': 2, 'linecolor': 'black', 'square': True}

# Plot X
sns.heatmap(X, ax=axes[0], cmap='Blues', **kwargs)
axes[0].set_title('Input Image (X)\n3x3', fontsize=18, fontweight='bold', pad=15)
axes[0].set_xticks([])
axes[0].set_yticks([])

# Plot dY
sns.heatmap(dY, ax=axes[1], cmap='Reds', **kwargs)
axes[1].set_title('Error Gradient (dY)\n2x2 Filter', fontsize=18, fontweight='bold', pad=15)
axes[1].set_xticks([])
axes[1].set_yticks([])

# Plot dW
sns.heatmap(dW, ax=axes[2], cmap='Greens', **kwargs)
axes[2].set_title('Weight Gradient (dW)\n2x2 Result', fontsize=18, fontweight='bold', pad=15)
axes[2].set_xticks([])
axes[2].set_yticks([])

plt.suptitle('CNN Backprop: Updating Weights via Convolution', fontsize=24, fontweight='bold', y=1.05)

text_str = "Because the filter shares weights across the image, the gradient for each weight is the sum of the errors it caused.\n"
text_str += "Mathematically, this is just convolving the Input with the Error! (dW = X * dY)"

plt.figtext(0.5, -0.05, text_str, ha="center", fontsize=16, 
            bbox={"facecolor": "white", "alpha": 1, "pad": 15, "edgecolor": "black", "linewidth": 2})

plt.tight_layout()
plt.savefig("assets/cnn_backprop_visual.png", dpi=300, bbox_inches='tight')
print("Saved assets/cnn_backprop_visual.png")
