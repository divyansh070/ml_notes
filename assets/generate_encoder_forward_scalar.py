import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
import numpy as np

# --- CONFIGURATION & HELPERS ---
C_BG = '#f8f9fa'
C_INPUT = '#d5f5e3'
C_MATH = '#f1c40f'
C_QKV = '#d6eaf8'
C_ATTN = '#fcf3cf'
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

def draw_math_box(ax, x, y, text, bg_color='white'):
    ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold', bbox=dict(facecolor=bg_color, edgecolor='none', alpha=0.9))

# ==========================================
# 1. ENCODER FORWARD PASS: COMP. GRAPH
# ==========================================
fig, ax = plt.subplots(figsize=(14, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis('off')

# Data Setup (Scalar Model, d=2)
X = np.array([[0.8, 0.8], [0.74, 1.44]]) # Input
W_Q = np.array([[1.0, 0.0], [0.0, 1.0]]) # Identity Weights for simplicity
W_K = np.array([[0.5, 0.5], [0.2, 0.8]])
W_V = np.array([[2.0, 0.0], [0.0, 2.0]])

# 1. Linear Projections (Q, K, V)
draw_matrix(ax, 0.5, 9, X, "Input X\n[Good, morning]")
draw_wire(ax, (3.5, 9.5), (5, 10.5), text=r"$W_Q$ $\times$")
draw_wire(ax, (3.5, 9.5), (5, 9.5), text=r"$W_K$ $\times$")
draw_wire(ax, (3.5, 9.5), (5, 8.5), text=r"$W_V$ $\times$")

draw_matrix(ax, 5.5, 10.1, X @ W_Q, "Query Q")
draw_matrix(ax, 5.5, 9.1, X @ W_K, "Key K")
draw_matrix(ax, 5.5, 8.1, X @ W_V, "Value V")

# 2. Attention: Raw Scores (QK^T)
KT = (X @ W_K).T
draw_wire(ax, (8.5, 10.5), (10, 10))
draw_wire(ax, (8.5, 9.5), (10, 10))
draw_matrix(ax, 10.5, 9.1, (X @ W_Q) @ KT, r"Raw Attention Scores" + "\n" + r"$Q \cdot K^T$")
draw_math_box(ax, 12.5, 11, r"$d_k=2$, $\sqrt{d_k} \approx 1.41$", 'lightgray')

# 3. Attention: Scale & Softmax
draw_wire(ax, (13.5, 9.5), (15, 9.5), text=r"$\div 1.41$, Softmax")
# Hardcoded softmax result for clarity
Softmax_Weights = np.array([[0.8, 0.2], [0.1, 0.9]]) 
draw_matrix(ax, 15.5, 9.1, Softmax_Weights, "Attention Weights")

# 4. Attention: Multiply by V
draw_wire(ax, (17.5, 9.5), (15.5, 6)) # Back towards center
draw_wire(ax, (8.5, 8.5), (12.5, 6)) # V back towards center
# Result: V contextualized
X_context = Softmax_Weights @ (X @ W_V)
draw_matrix(ax, 13.5, 5, X_context, r"Contextualized" + "\n" + r"Outputs ($Z$)")

# 5. Add & Norm 1 (Residual Connection)
draw_wire(ax, (3.5, 9.5), (7, 6), text=r"+ Residual", rad=0.3, color='red') # From input
draw_wire(ax, (13.5, 5.5), (10.5, 5.5), color='#333') # From Attention

draw_matrix(ax, 10.5, 3.5, X + X_context, "Residual Addition")
draw_wire(ax, (10.5, 3.9), (9, 3.9), text="LayerNorm")
# Final normalized output Z_prime
Z_prime = np.array([[1.0, -1.0], [-1.0, 1.0]]) # Hardcoded example
draw_matrix(ax, 8, 2.5, Z_prime, "Normalized Z'")

# 6. Feed Forward Network (Expansion 2->4->2)
draw_wire(ax, (8, 2.9), (6, 2.9), text=r"$W_1 \times$, ReLU")
draw_wire(ax, (6, 2.9), (4, 2.9), text=r"$W_2 \times$")
X_ffn = np.array([[3.0, 3.0], [-1.0, 2.0]]) # FFN Output example
draw_matrix(ax, 4, 1.5, X_ffn, "FFN Outputs")

# 7. Add & Norm 2 (Final Encoder Output)
draw_wire(ax, (6.5, 1.9), (8.5, 1.9), color='red', text="+ Residual", rad=-0.2) # Residual Z' to FFN out
draw_wire(ax, (4, 1.9), (5.5, 1.9))
# Final context matrix H
ax.text(10, 0.5, "FINAL Encoder Output (Blueprint H)", ha='center', fontsize=14, fontweight='bold', color='purple', bbox=dict(facecolor='#e8daef', edgecolor='purple'))

plt.suptitle("Transformer synthesis 1: Complete Encoder Forward Prop Computational Graph (d=2)", fontsize=20, fontweight='bold', y=0.98)
plt.savefig('assets/encoder_forward_scalar.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated assets/encoder_forward_scalar.png successfully!")
