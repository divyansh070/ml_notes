import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
import numpy as np

# --- CONFIGURATION & HELPERS ---
C_BG = '#f8f9fa'
C_MATH = '#f1c40f'
C_ATTN = '#fcf3cf'
C_MASK = '#fadbd8' # Red for Causal Mask
C_CROSS = '#e8daef' # Purple for Cross Context
C_TEXT = 'black'

def draw_node(ax, x, y, text, radius=0.4, color=C_MATH, text_color='black', fontsize=16):
    circle = Circle((x, y), radius, facecolor=color, edgecolor='black', lw=2, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=text_color, zorder=4)

def draw_wire(ax, start, end, text=None, color='#333', text_color='black', text_y_offset=0, bold=True, rad=0.0):
    arrow = FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", arrowstyle='-|>', mutation_scale=15, color=color, lw=2, zorder=1)
    ax.add_patch(arrow)
    if text:
        mid_x = (start[0] + end[0]) / 2
        mid_y = ((start[1] + end[1]) / 2) + text_y_offset
        f_weight = 'bold' if bold else 'normal'
        ax.text(mid_x, mid_y, text, ha='center', va='center', fontsize=10, fontweight=f_weight, color=text_color,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9), zorder=5)

def draw_arrow(ax, start, end, text=None, rad=0.0):
    draw_wire(ax, start, end, text=text, rad=rad)

def draw_matrix(ax, x, y, matrix, title, color='gray'):
    rows, cols = matrix.shape
    height, width = rows * 0.8, cols * 1.5
    
    # Title
    ax.text(x + width/2, y + height + 0.3, title, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    for r in range(rows):
        for c in range(cols):
            val = matrix[rows-1-r, c]
            rect = Rectangle((x + c*1.5, y + r*0.8), 1.5, 0.8, facecolor='white', edgecolor='black', lw=1, zorder=2)
            ax.add_patch(rect)
            ax.text(x + c*1.5 + 0.75, y + r*0.8 + 0.4, f"{val:.2f}", ha='center', va='center', fontsize=10, fontweight='bold')
    return x + width, y + height

# ==========================================
# 2. DECODER FORWARD PASS: COMP. GRAPH
# ==========================================
fig, ax = plt.subplots(figsize=(14, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis('off')

# Data Setup (Decoder input <Start>)
Y_in = np.array([[1.0, 1.0]]) # Decoder Embedding for <Start>
H_enc = np.array([[1.1, -0.9], [-1.1, 1.3]]) # FINAL Encoder Output H

# 1. Masked Self-Attention Track
draw_matrix(ax, 0.5, 9, Y_in, "Decoder Input (Y)\n[<Start>]")
draw_wire(ax, (3.5, 9.5), (5, 9.5), text=r"$W_Q, W_K, W_V$")
draw_node(ax, 5, 9.5, r"$\otimes, \sigma$", color=C_MASK)
draw_arrow(ax, (5.4, 9.5), (7, 9.5))
mask = np.array([[1.0]]) # 1x1 mask for 1 token is trivial
draw_matrix(ax, 7.5, 9, mask, "Masked Attn Weights")
Y_context = np.array([[1.5, 1.5]]) # Hardcoded result, fixed syntax error here
draw_matrix(ax, 10.5, 9, Y_context, "Masked Z_dec")

# Residual Path 1
draw_wire(ax, (3.5, 9.5), (6, 7.5), text=r"+ Residual", rad=0.2, color='red')
draw_wire(ax, (10.5, 9.4), (9, 7.5))
draw_matrix(ax, 9, 6, Y_in + Y_context, "Addition 1")
draw_node(ax, 9, 6, "Layer\nNorm", color='white', radius=0.3)
# Decoded Z_prime
Z_dec_prime = np.array([[1.0, 1.0]]) 
draw_matrix(ax, 9, 4, Z_dec_prime, "Z'_dec (Input to Cross)")

# 2. CROSS-ATTENTION TRACK (Bridge to Encoder)
ax.text(10, 1, "EXTERNAL Encoder Input (H)", ha='center', fontsize=12, color='purple', bbox=dict(facecolor='#e8daef', edgecolor='purple'))
draw_matrix(ax, 11, 1, H_enc, "Encoder Blueprint H")

# Decoder Query (Z_prime) X Encoder Context (H)
draw_wire(ax, (11.5, 1.9), (12.5, 3.5), text=r"$\times$ $W_Q, W_K, W_V$") # H and Z merging
draw_wire(ax, (9, 4.4), (12.5, 3.5))

ax.text(12.8, 3.8, r"CROSS-ATTN Node", fontweight='bold', fontsize=12, color='purple')
ax.text(12.8, 3.5, r"Q $\in Z'_{dec}$" + "\n" + r"K, V $\in H_{enc}$", color='purple', fontsize=9)
draw_arrow(ax, (13.5, 3.5), (15, 3.5))
Cross_Weights = np.array([[0.3, 0.7]]) # Hardcoded example blend
draw_matrix(ax, 15.5, 3, Cross_Weights, "Cross-Attn Weights")

# Arrow to context multiplication
draw_wire(ax, (15.5, 3.5), (13.5, 1.5))
# Final Cross Context
Z_cross = np.array([[-0.1, 0.5]]) # Example: Heavy on Word 2 of Encoder
draw_matrix(ax, 12, 0.5, Z_cross, "Cross-Contextual Z_c")

plt.suptitle("Transformer synthesis 2: Decoder Forward Prop (Cross-Attention)", fontsize=20, fontweight='bold', y=0.98)
plt.savefig('assets/decoder_forward_scalar.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated assets/decoder_forward_scalar.png successfully!")
