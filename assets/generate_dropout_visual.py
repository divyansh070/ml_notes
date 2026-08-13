import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# Create the Feature Map (3 channels, 8x8 spatial)
tensor = np.full((3, 8, 8), 0.2)

# Add a distinct 3x3 active feature across all channels
tensor[:, 2:5, 2:5] = 1.0

# --- Standard Dropout (p=0.5) ---
# Randomly zero out 50% of the individual pixels independently
standard_dropout_mask = np.random.binomial(1, 0.5, size=tensor.shape)
standard_dropout_tensor = tensor * standard_dropout_mask

# --- Spatial Dropout (p=0.33) ---
# Randomly zero out 1 out of the 3 channels entirely
spatial_dropout_tensor = tensor.copy()
spatial_dropout_tensor[1, :, :] = 0.0 # Force drop Channel 2 (index 1) for reliable visual

# --- Visualization ---
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

kwargs = {'vmin': 0.0, 'vmax': 1.0, 'cmap': 'magma', 'cbar': False, 
          'linewidths': 1, 'linecolor': 'black', 'square': True}

# Top Row: Standard Dropout
for i in range(3):
    sns.heatmap(standard_dropout_tensor[i], ax=axes[0, i], **kwargs)
    axes[0, i].set_title(f'Channel {i+1}', fontsize=16, fontweight='bold', pad=10)
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])

# Bottom Row: Spatial Dropout
for i in range(3):
    sns.heatmap(spatial_dropout_tensor[i], ax=axes[1, i], **kwargs)
    axes[1, i].set_title(f'Channel {i+1}', fontsize=16, fontweight='bold', pad=10)
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])

# Row Titles
fig.text(0.5, 0.95, "Standard Dropout: Fails to drop the feature (pixels co-adapt)", 
         ha='center', va='center', fontsize=20, fontweight='bold', bbox=dict(facecolor='#ffcccc', edgecolor='black', pad=10))
fig.text(0.5, 0.48, "Spatial Dropout: Drops entire feature maps to force independent learning", 
         ha='center', va='center', fontsize=20, fontweight='bold', bbox=dict(facecolor='#cce5ff', edgecolor='black', pad=10))

plt.subplots_adjust(hspace=0.4)
plt.savefig('assets/spatial_dropout_visual.png', dpi=300, bbox_inches='tight')
print("Saved assets/spatial_dropout_visual.png")
