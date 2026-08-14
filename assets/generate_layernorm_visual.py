import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def draw_tensor_slice(ax, slice_type, title, color):
    # Base 3D Tensor dimensions: Batch(Z)=4, Seq(Y)=5, Feature(X)=6
    # We will draw the wireframe of the full tensor
    for z in range(5):
        ax.plot([0, 6], [0, 5], [z, z], color='lightgray', lw=1, zorder=1)
    for y in range(6):
        ax.plot([0, 6], [y, y], [0, 4], color='lightgray', lw=1, zorder=1)
    for x in range(7):
        ax.plot([x, x], [0, 5], [0, 4], color='lightgray', lw=1, zorder=1)

    # Draw the highlighted slice based on normalization type
    if slice_type == 'batch':
        # BatchNorm: Normalizes one Feature across the entire Batch and Sequence
        # Slice at X=2 (Feature 2)
        x = 2
        y, z = np.meshgrid(range(6), range(5))
        ax.plot_surface(np.full_like(y, x), y, z, color=color, alpha=0.6, zorder=3)
        ax.text(x, 2.5, 4.5, "Norm across\nBatch & Seq", color='red', fontsize=10, fontweight='bold', ha='center')
        
    elif slice_type == 'layer':
        # LayerNorm: Normalizes all Features for ONE Word in ONE Batch
        # Slice at Z=3 (Batch 3), Y=2 (Word 2)
        z = 3
        y = 2
        x, z_mesh = np.meshgrid(range(7), [z, z+1]) # Small thickness for visibility
        ax.plot_surface(x, np.full_like(x, y), np.full_like(x, z), color=color, alpha=0.8, zorder=3)
        ax.text(3, y, z+0.5, "Norm across\nFeatures (1 Word)", color='blue', fontsize=10, fontweight='bold', ha='center')

    # Labels
    ax.set_xlabel('Feature / Embedding ($d_{model}$)', fontweight='bold')
    ax.set_ylabel('Sequence (Words)', fontweight='bold')
    ax.set_zlabel('Batch Size', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Clean up axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)

# ==========================================
# GENERATE THE 3D COMPARISON
# ==========================================
fig = plt.figure(figsize=(14, 6))

# Left: BatchNorm
ax1 = fig.add_subplot(121, projection='3d')
draw_tensor_slice(ax1, 'batch', "Batch Normalization\n(Fails on variable length text)", '#ff9999')
ax1.view_init(elev=20, azim=-45)

# Right: LayerNorm
ax2 = fig.add_subplot(122, projection='3d')
draw_tensor_slice(ax2, 'layer', "Layer Normalization\n(Perfect for independent tokens)", '#99ccff')
ax2.view_init(elev=20, azim=-45)

plt.subplots_adjust(wspace=0.1)
plt.savefig('assets/layernorm_vs_batchnorm.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated assets/layernorm_vs_batchnorm.png successfully!")
