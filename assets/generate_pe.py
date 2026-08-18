import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- CONFIGURATION ---
C_EMBED = '#d5f5e3'
C_POS = '#d6eaf8'
C_FINAL = '#fcf3cf'

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#f8f9fa')

# -----------------------------------------------------
# TOP LEFT: Dimensions 0 and 1 (High Frequency)
# -----------------------------------------------------
ax1 = fig.add_axes([0.1, 0.55, 0.35, 0.35])
x = np.linspace(0, 5, 200)

# Plot sine and cosine waves
ax1.plot(x, np.sin(x), color='blue', label='Dim 0: $\sin(pos/1)$', lw=2)
ax1.plot(x, np.cos(x), color='red', label='Dim 1: $\cos(pos/1)$', linestyle='--', lw=2)

# Sample the wave exactly at pos=1
ax1.axvline(1, color='gray', linestyle=':', linewidth=2, zorder=1)
ax1.scatter([1, 1], [np.sin(1), np.cos(1)], color=['blue', 'red'], s=150, zorder=5, edgecolor='black')

# Annotate the extracted values
ax1.text(1.1, np.sin(1), ' 0.84', color='blue', fontweight='bold', fontsize=14, va='center')
ax1.text(1.1, np.cos(1), ' 0.54', color='red', fontweight='bold', fontsize=14, va='center')

ax1.set_title("Lower Dimensions (Fast Oscillations)\nActing like the 'Seconds Hand'", fontweight='bold', fontsize=14)
ax1.set_xlabel("Position in Sequence (pos)", fontweight='bold')
ax1.set_ylabel("Encoding Value", fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# -----------------------------------------------------
# TOP RIGHT: Dimensions 2 and 3 (Low Frequency)
# -----------------------------------------------------
ax2 = fig.add_axes([0.55, 0.55, 0.35, 0.35])

# Plot the slower sine and cosine waves
ax2.plot(x, np.sin(x/100), color='blue', label='Dim 2: $\sin(pos/100)$', lw=2)
ax2.plot(x, np.cos(x/100), color='red', label='Dim 3: $\cos(pos/100)$', linestyle='--', lw=2)

# Sample the wave exactly at pos=1
ax2.axvline(1, color='gray', linestyle=':', linewidth=2, zorder=1)
ax2.scatter([1, 1], [np.sin(1/100), np.cos(1/100)], color=['blue', 'red'], s=150, zorder=5, edgecolor='black')

# Annotate the extracted values
ax2.text(1.1, np.sin(1/100)+0.05, ' 0.01', color='blue', fontweight='bold', fontsize=14, va='bottom')
ax2.text(1.1, np.cos(1/100)-0.05, ' 1.00', color='red', fontweight='bold', fontsize=14, va='top')

ax2.set_title("Higher Dimensions (Slow Oscillations)\nActing like the 'Hours Hand'", fontweight='bold', fontsize=14)
ax2.set_xlabel("Position in Sequence (pos)", fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

# -----------------------------------------------------
# BOTTOM: Vector Addition for Word 1 (pos = 1)
# -----------------------------------------------------
ax3 = fig.add_axes([0.1, 0.0, 0.8, 0.45])
ax3.axis('off')

def draw_vector(ax, start_x, start_y, values, title, bg_color):
    ax.text(start_x + 2, start_y + 1.4, title, ha='center', va='center', fontsize=14, fontweight='bold')
    for i, val in enumerate(values):
        rect = Rectangle((start_x + i, start_y), 1, 1, facecolor=bg_color, edgecolor='black', lw=2)
        ax.add_patch(rect)
        ax.text(start_x + i + 0.5, start_y + 0.5, str(val), ha='center', va='center', fontsize=14, fontweight='bold')

ax3.text(8, 4.5, "Step 2: Extracting Values and Adding to the Word Embedding (pos=1)", ha='center', fontsize=16, fontweight='bold', style='italic', color='gray')

# Draw the Word Label
ax3.text(1, 2.5, 'Word 1\n("Dog")', ha='center', va='center', fontsize=16, fontweight='bold', color='gray')

# Draw Semantic Embedding
draw_vector(ax3, 2.5, 2, [-0.1, 0.9, 0.4, -0.5], "Semantic Embedding", C_EMBED)

# Plus Sign
ax3.text(7, 2.5, "+", ha='center', va='center', fontsize=32, fontweight='bold')

# Draw Positional Encoding (The sampled values)
draw_vector(ax3, 7.5, 2, [0.84, 0.54, 0.01, 1.0], "Positional Encoding Vector", C_POS)

# Annotations showing where the numbers came from
ax3.annotate('Sampled from\nDim 0 & 1', xy=(8.5, 3.1), xytext=(8.5, 3.8),
             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
             ha='center', va='bottom', fontsize=12, fontweight='bold')

ax3.annotate('Sampled from\nDim 2 & 3', xy=(10.5, 3.1), xytext=(10.5, 3.8),
             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# Equals Sign
ax3.text(12, 2.5, "=", ha='center', va='center', fontsize=32, fontweight='bold')

# Draw Final Output
draw_vector(ax3, 12.5, 2, [0.74, 1.44, 0.41, 0.5], "Final Transformer Input", C_FINAL)

# Title
fig.suptitle("Visualizing Transformer Positional Encoding (Generation & Addition)", fontsize=24, fontweight='bold', y=0.98)

plt.savefig('assets/generating_positional_encoding.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated assets/generating_positional_encoding.png successfully!")
