import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_tensor_slice(ax, slice_type, title, color):
    # Draw wireframe cube (Batch=4, Seq=5, Features=6)
    for z in range(5):
        ax.plot([0, 6], [0, 5], [z, z], color='lightgray', lw=1, zorder=1)
    for y in range(6):
        ax.plot([0, 6], [y, y], [0, 4], color='lightgray', lw=1, zorder=1)
    for x in range(7):
        ax.plot([x, x], [0, 5], [0, 4], color='lightgray', lw=1, zorder=1)

    if slice_type == 'batch':
        # BatchNorm: Slice at X=2 (One feature, across all words and batches)
        # Vertices: (x, y, z)
        verts = [[(2, 0, 0), (2, 5, 0), (2, 5, 4), (2, 0, 4)]]
        poly = Poly3DCollection(verts, alpha=0.7, facecolors=color, edgecolors='darkred', zorder=3)
        ax.add_collection3d(poly)
        ax.text(2.5, 2.5, 4.5, "Norm across\nBatch & Seq", color='red', fontsize=11, fontweight='bold', ha='center')
        
    elif slice_type == 'layer':
        # LayerNorm: Slice at Z=3, Y=2 (All features for ONE word in ONE batch)
        # We give it a slight thickness in Z so it renders as a visible strip
        verts = [[(0, 2, 3), (6, 2, 3), (6, 2, 3.5), (0, 2, 3.5)]]
        poly = Poly3DCollection(verts, alpha=0.8, facecolors=color, edgecolors='darkblue', zorder=3)
        ax.add_collection3d(poly)
        ax.text(3, 1, 3.8, "Norm across\nFeatures (1 Word)", color='blue', fontsize=11, fontweight='bold', ha='center')

    # Labels & Limits
    ax.set_xlim(0, 6); ax.set_ylim(0, 5); ax.set_zlim(0, 4)
    ax.set_xlabel('Feature / Embedding ($d_{model}$)', fontweight='bold', labelpad=10)
    ax.set_ylabel('Sequence (Words)', fontweight='bold', labelpad=10)
    ax.set_zlabel('Batch Size', fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)

# Generate Figure
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
draw_tensor_slice(ax1, 'batch', "Batch Normalization\n(Fails on variable length text)", '#ff9999')
ax1.view_init(elev=20, azim=-45)

ax2 = fig.add_subplot(122, projection='3d')
draw_tensor_slice(ax2, 'layer', "Layer Normalization\n(Perfect for independent tokens)", '#99ccff')
ax2.view_init(elev=20, azim=-45)

plt.subplots_adjust(wspace=0.1)
plt.savefig('assets/layernorm_vs_batchnorm.png', dpi=300, bbox_inches='tight')
plt.close()
print("Generated Fixed assets/layernorm_vs_batchnorm.png successfully!")
