import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# HARDCODED MATRICES FROM USER
X = np.array([[1, 1, 1], [1, 1, 0], [0, 1, 1]])
W1 = np.array([[1, -1], [-1, 1]])
Z1 = np.array([[0, -1], [2, 1]])
A1 = np.array([[0, 0], [2, 1]])
F = np.array([[0, 0, 2, 1]])
W2 = np.array([[0.1, 0.2, 0.3, -0.4]])
pred = 0.55
error = -0.45
dF = np.array([[-0.045, -0.09, -0.135, 0.18]])
dA1 = np.array([[-0.045, -0.09], [-0.135, 0.18]])
dY = np.array([[0, 0], [-0.135, 0.18]])
dW1 = np.array([[0.045, -0.135], [0.18, 0.045]])

def plot_heatmap(ax, data, title, cmap, fmt='.2g'):
    sns.heatmap(data, ax=ax, annot=True, fmt=fmt, cmap=cmap, cbar=False, 
                annot_kws={'size': 16, 'weight': 'bold'}, linewidths=2, linecolor='black', square=True)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks([])
    ax.set_yticks([])

# --- FORWARD PASS INFOGRAPHIC ---
fig, axes = plt.subplots(1, 6, figsize=(24, 4), gridspec_kw={'width_ratios': [3, 2, 2, 4, 4, 2]})

plot_heatmap(axes[0], X, '1. Input (X)\n3x3', 'Blues', fmt='d')
plot_heatmap(axes[1], W1, '2. Conv Filter (W1)\n2x2', 'Oranges', fmt='d')
plot_heatmap(axes[2], A1, '3. ReLU Output (A1)\n2x2', 'Greens', fmt='d')
plot_heatmap(axes[3], F, '4. Flatten (F)\n1x4', 'Purples', fmt='d')
plot_heatmap(axes[4], W2, '5. Dense Weights (W2)\n1x4', 'Oranges', fmt='.1f')

axes[5].axis('off')
axes[5].text(0.5, 0.5, f"6. Prediction\n{pred}", ha='center', va='center', fontsize=20, fontweight='bold',
             bbox=dict(facecolor='#e6f5c9', edgecolor='black', boxstyle='round,pad=1', lw=2))

# Add arrows between subplots
for i in range(5):
    pos = axes[i].get_position()
    fig.text(pos.x1 + 0.015, pos.y0 + pos.height/2, '➔', fontsize=30, ha='center', va='center', fontweight='bold')

plt.suptitle('CNN Forward Pass: From Pixels to Prediction', fontsize=26, fontweight='bold', y=1.2)
plt.subplots_adjust(wspace=0.3)
plt.savefig('assets/cnn_forward_pass.png', dpi=300, bbox_inches='tight')

# --- BACKWARD PASS INFOGRAPHIC ---
fig, axes = plt.subplots(1, 6, figsize=(24, 4), gridspec_kw={'width_ratios': [2, 4, 2, 2, 3, 2]})
axes[0].axis('off')
axes[0].text(0.5, 0.5, f"1. Error (Loss grad)\n{error}", ha='center', va='center', fontsize=18, fontweight='bold',
             bbox=dict(facecolor='#fbb4ae', edgecolor='black', boxstyle='round,pad=1', lw=2))

plot_heatmap(axes[1], dF, '2. Dense Gradient (dF)\n1x4', 'Reds', fmt='.3f')
plot_heatmap(axes[2], dA1, '3. Unflatten (dA1)\n2x2', 'Reds', fmt='.3f')
plot_heatmap(axes[3], dY, '4. ReLU Mask (dY)\n2x2', 'Reds', fmt='.3f')
plot_heatmap(axes[4], X, '5. Input (X)\n3x3', 'Blues', fmt='d')
plot_heatmap(axes[5], dW1, '6. Conv Gradient (dW1)\n2x2', 'Greens', fmt='.3f')

for i in range(5):
    pos = axes[i].get_position()
    fig.text(pos.x1 + 0.015, pos.y0 + pos.height/2, '➔', fontsize=30, ha='center', va='center', fontweight='bold')

plt.suptitle('CNN Backward Pass: Propagating Error to Weights', fontsize=26, fontweight='bold', y=1.2)
plt.subplots_adjust(wspace=0.3)
plt.savefig('assets/cnn_backward_pass.png', dpi=300, bbox_inches='tight')
print("Saved assets/cnn_forward_pass.png and assets/cnn_backward_pass.png")
