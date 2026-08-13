import numpy as np
import matplotlib.pyplot as plt

def draw_neural_net(ax, layer_sizes, dropout_rate=0.0):
    # Setup coordinates for nodes
    v_spacing = 1.0
    h_spacing = 2.0
    
    nodes = []
    # Create nodes coordinates
    for i, size in enumerate(layer_sizes):
        layer_nodes = []
        # Center the layer vertically
        y_offset = (max(layer_sizes) - size) * v_spacing / 2.0
        for j in range(size):
            x = i * h_spacing
            y = y_offset + j * v_spacing
            layer_nodes.append((x, y))
        nodes.append(layer_nodes)
    
    # Decide which hidden nodes are dropped
    dropped_nodes = []
    if dropout_rate > 0:
        for i in range(1, len(layer_sizes) - 1): # Only hidden layers
            size = layer_sizes[i]
            # Randomly select nodes to drop
            num_drop = int(size * dropout_rate)
            drop_indices = np.random.choice(size, num_drop, replace=False)
            dropped_nodes.append(drop_indices)
    else:
        for i in range(1, len(layer_sizes) - 1):
            dropped_nodes.append([])

    # Draw Edges (Synapses)
    for i in range(len(nodes) - 1):
        layer1 = nodes[i]
        layer2 = nodes[i+1]
        
        # Are we looking at dropped nodes?
        drop1 = []
        drop2 = []
        if i > 0: drop1 = dropped_nodes[i-1]
        if i < len(layer_sizes) - 2: drop2 = dropped_nodes[i]
            
        for n1_idx, n1 in enumerate(layer1):
            for n2_idx, n2 in enumerate(layer2):
                # If either node is dropped, draw a faint dashed red line to show the dead connection
                is_dropped = (n1_idx in drop1) or (n2_idx in drop2)
                if is_dropped:
                    ax.plot([n1[0], n2[0]], [n1[1], n2[1]], 'r--', alpha=0.15, linewidth=1)
                else:
                    ax.plot([n1[0], n2[0]], [n1[1], n2[1]], 'gray', alpha=0.6, linewidth=1.5)

    # Draw Nodes (Neurons)
    for i, layer_nodes in enumerate(nodes):
        drop_indices = []
        if 0 < i < len(layer_sizes) - 1:
            drop_indices = dropped_nodes[i-1]
            
        for j, node in enumerate(layer_nodes):
            if j in drop_indices:
                # Dropped Node
                circle = plt.Circle(node, radius=0.2, color='red', alpha=0.2, zorder=4)
                ax.add_patch(circle)
                ax.text(node[0], node[1], 'X', ha='center', va='center', color='darkred', fontsize=16, fontweight='bold', zorder=5)
            else:
                # Active Node
                color = 'dodgerblue' if i == 0 else ('mediumseagreen' if i == len(layer_sizes)-1 else 'orange')
                circle = plt.Circle(node, radius=0.2, color=color, zorder=4, edgecolor='black', linewidth=1.5)
                ax.add_patch(circle)

    ax.axis('off')

# Setup figure
np.random.seed(42) # For reproducible "random" dropout
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Define a simple network architecture: 4 Input -> 6 Hidden -> 6 Hidden -> 2 Output
layer_sizes = [4, 6, 6, 2]

# --- Left Plot: Standard Network ---
draw_neural_net(axes[0], layer_sizes, dropout_rate=0.0)
axes[0].set_title("Standard Neural Network\n(100% of neurons active during training)", fontsize=16, fontweight='bold', pad=15)
axes[0].text(0.5, -0.05, "The Danger: Neurons can become highly dependent on specific pathways\n(Co-adaptation), leading to memorization of the training data.", 
             ha='center', va='top', transform=axes[0].transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

# --- Right Plot: Network with Dropout ---
draw_neural_net(axes[1], layer_sizes, dropout_rate=0.5) # 50% dropout
axes[1].set_title("Network with 50% Dropout\n(Random neurons deactivated per batch)", fontsize=16, fontweight='bold', pad=15)
axes[1].text(0.5, -0.05, "The Solution: Forces the network to spread information across all neurons.\nNo single neuron can be relied upon exclusively.", 
             ha='center', va='top', transform=axes[1].transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.suptitle("How Dropout Prevents Co-adaptation and Overfitting", fontsize=22, fontweight='bold', y=1.1)
plt.savefig("assets/dropout_visualization.png", dpi=300, bbox_inches='tight')
print("Successfully generated assets/dropout_visualization.png")
