import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set seed for reproducible matrix numbers
np.random.seed(42)

# 1. Tensors
# Input Tensor: 3x3x3
input_tensor = np.random.randint(1, 6, size=(3, 3, 3))
# 1x1x3 Filter Weights
weights = np.array([2, -1, 3])

# The target pixel: Row 1, Col 1 (center pixel of a 3x3 matrix)
target_row, target_col = 1, 1
# Extract channels for the center pixel
c1, c2, c3 = input_tensor[target_row, target_col, :]
w1, w2, w3 = weights

# 2. Output
output_val = (c1 * w1) + (c2 * w2) + (c3 * w3)
output_tensor = np.zeros((3, 3), dtype=int)
for r in range(3):
    for c in range(3):
        output_tensor[r, c] = np.sum(input_tensor[r, c, :] * weights)

# 3. Visualization Setup
fig, axes = plt.subplots(1, 3, figsize=(18, 7))

# Helper to draw a matrix layer in pseudo-3D
def draw_layer(ax, matrix, z_offset_x, z_offset_y, color, alpha, highlight_rc=None, highlight_color='red'):
    rows, cols = matrix.shape
    for r in range(rows):
        for c in range(cols):
            x = c + z_offset_x
            y = (rows - r - 1) + z_offset_y
            
            # Highlight specific pixel
            is_highlight = highlight_rc is not None and r == highlight_rc[0] and c == highlight_rc[1]
            edgecolor = highlight_color if is_highlight else 'black'
            linewidth = 5 if is_highlight else 1
            facecolor = 'white' if is_highlight else color
            text_color = 'red' if is_highlight else 'black'
            text_weight = 'bold' if is_highlight else 'normal'

            rect = patches.Rectangle((x, y), 1, 1, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, zorder=z_offset_x*10)
            ax.add_patch(rect)
            ax.text(x + 0.5, y + 0.5, str(matrix[r, c]), ha='center', va='center', color=text_color, fontweight=text_weight, fontsize=16, zorder=z_offset_x*10 + 1)

# --- Subplot 1: 3x3x3 Input Tensor ---
ax1 = axes[0]
ax1.set_title("Input Tensor (3x3x3)\n(Exploded Channels)", pad=20, fontsize=18, fontweight='bold')
ax1.set_xlim(-0.5, 6)
ax1.set_ylim(-0.5, 6)
ax1.axis('off')

# Draw the 3 channels (back to front)
draw_layer(ax1, input_tensor[:, :, 2], 1.6, 1.6, '#b3e2cd', 0.8, highlight_rc=(1,1))
draw_layer(ax1, input_tensor[:, :, 1], 0.8, 0.8, '#fdcdac', 0.8, highlight_rc=(1,1))
draw_layer(ax1, input_tensor[:, :, 0], 0.0, 0.0, '#cbd5e8', 0.8, highlight_rc=(1,1))

# Draw a piercing line through the highlighted pixel
ax1.plot([1.5, 1.5+0.8, 1.5+1.6], [1.5, 1.5+0.8, 1.5+1.6], color='red', linestyle='--', linewidth=3, zorder=100)

# --- Subplot 2: 1x1x3 Filter ---
ax2 = axes[1]
ax2.set_title("1x1x3 Filter Weights\n(Pointwise Convolution)", pad=20, fontsize=18, fontweight='bold')
ax2.set_xlim(-1, 3.5)
ax2.set_ylim(0, 4.5)
ax2.axis('off')

# Draw the 1x1 filters (back to front)
def draw_1x1(ax, val, z_offset_x, z_offset_y, color):
    x = 0.5 + z_offset_x
    y = 1.5 + z_offset_y
    rect = patches.Rectangle((x, y), 1, 1, facecolor='white', edgecolor='red', linewidth=5, zorder=z_offset_x*10)
    ax.add_patch(rect)
    ax.text(x + 0.5, y + 0.5, str(val), ha='center', va='center', color='red', fontweight='bold', fontsize=20, zorder=z_offset_x*10 + 1)

draw_1x1(ax2, w3, 1.6, 1.6, '#b3e2cd')
draw_1x1(ax2, w2, 0.8, 0.8, '#fdcdac')
draw_1x1(ax2, w1, 0.0, 0.0, '#cbd5e8')
ax2.plot([1.0, 1.0+0.8, 1.0+1.6], [2.0, 2.0+0.8, 2.0+1.6], color='red', linestyle='--', linewidth=3, zorder=100)

# --- Subplot 3: Output Map ---
ax3 = axes[2]
ax3.set_title("Output Feature Map (3x3x1)\nCross-Channel Dot Product", pad=20, fontsize=18, fontweight='bold')
ax3.set_xlim(-0.5, 3.5)
ax3.set_ylim(-0.5, 3.5)
ax3.axis('off')

draw_layer(ax3, output_tensor, 0, 0, '#e6f5c9', 0.9, highlight_rc=(1,1))

# --- 4. Text Annotation ---
math_str = f"Mathematical Dot Product for Center Pixel:\n\n"
math_str += f"(Input_C1 × Weight_1) + (Input_C2 × Weight_2) + (Input_C3 × Weight_3) = Output\n"
math_str += f"({c1} × {w1}) + ({c2} × {w2}) + ({c3} × {w3})  =  {c1*w1} + {c2*w2} + {c3*w3}  =  {output_val}"

plt.figtext(0.5, -0.05, math_str, ha="center", fontsize=18, 
            bbox={"facecolor": "white", "alpha": 1, "pad": 20, "edgecolor": "red", "linewidth": 3})

plt.suptitle("1x1 Convolution: Cross-Channel Parametric Pooling", fontsize=24, fontweight='bold', y=1.05)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig("assets/pointwise_visual.png", dpi=300, bbox_inches='tight')
print("Successfully generated assets/pointwise_visual.png")
