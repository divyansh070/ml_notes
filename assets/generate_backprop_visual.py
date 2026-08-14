import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import numpy as np

# --- HELPERS ---
def draw_matrix(ax, x, y, matrix, title, cell_w=1.2, cell_h=0.6, fontsize=11, title_fontsize=12, bg='white'):
    rows, cols = matrix.shape
    w, h = cols * cell_w, rows * cell_h
    ax.text(x + w/2, y + h + 0.25, title, ha='center', va='bottom', fontsize=title_fontsize, fontweight='bold')
    for r in range(rows):
        for c in range(cols):
            val = matrix[r, c]
            rect = Rectangle((x + c*cell_w, y + (rows-1-r)*cell_h), cell_w, cell_h,
                              facecolor=bg, edgecolor='black', lw=1.5, zorder=2)
            ax.add_patch(rect)
            ax.text(x + c*cell_w + cell_w/2, y + (rows-1-r)*cell_h + cell_h/2,
                    f"{val:.2f}", ha='center', va='center', fontsize=fontsize, fontweight='bold', zorder=3)

def draw_arrow(ax, start, end, color='#555', lw=2, text=None, text_offset=(0, 0.2), ls='-'):
    arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=18, color=color, lw=lw, linestyle=ls, zorder=1)
    ax.add_patch(arrow)
    if text:
        mx, my = (start[0]+end[0])/2 + text_offset[0], (start[1]+end[1])/2 + text_offset[1]
        ax.text(mx, my, text, ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9), zorder=4)

def draw_box(ax, x, y, w, h, text, bg, fontsize=12, text_color='black'):
    rect = Rectangle((x, y), w, h, facecolor=bg, edgecolor='black', lw=2, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=text_color, zorder=3)

def draw_gradient_box(ax, x, y, text, fontsize=11):
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color='red',
            bbox=dict(facecolor='#fff0f0', edgecolor='red', boxstyle='round,pad=0.4', lw=2), zorder=5)

# --- DATA ---
Q0 = np.array([[0.80, 0.80]])
K0 = np.array([[0.56, 1.04]])
Weights = np.array([[0.42, 0.58], [0.18, 0.82]])
dWk_word0 = np.array([[0.64, 0.64], [0.64, 0.64]])

# --- FIGURE: Clean 3-tier layout ---
fig, ax = plt.subplots(figsize=(18, 16))
ax.set_xlim(-1, 19)
ax.set_ylim(-1, 16)
ax.axis('off')
fig.patch.set_facecolor('white')

# ==========================================
# TIER 1 (TOP): Forward pass structure (faded)
# ==========================================
ax.text(9, 15.5, "— Forward Pass (faded reference) —", ha='center', fontsize=14, color='gray', fontstyle='italic')

draw_matrix(ax, 0, 13.5, Q0, "Query Q₀", bg='#f0f0f0')
draw_matrix(ax, 5, 13.5, K0, "Key K₀", bg='#f0f0f0')

draw_arrow(ax, (2.5, 14), (8, 14), color='gray', text="dot product →")
draw_box(ax, 8, 13.2, 3.5, 1.3, "Score Node\nQ₀·K₀ᵀ = 1.28", '#fef9e7', fontsize=11)

draw_arrow(ax, (11.5, 14), (13, 14), color='gray', text="softmax")
draw_matrix(ax, 13.5, 13.5, Weights, "Attn Weights", bg='#f0f0f0')

# ==========================================
# TIER 2 (MIDDLE): Gradient split at Score node
# ==========================================
ax.text(9, 11.5, "— Backward Pass: Error flows RIGHT → LEFT —", ha='center', fontsize=14, color='red', fontweight='bold')

# Error arrives
draw_gradient_box(ax, 16, 10.5, "Total Loss\ndL = 2.0")
draw_arrow(ax, (14.5, 10.5), (12, 10.5), color='red', lw=3, ls='--', text="Error signal →")

# Score node receives error
draw_box(ax, 8.5, 9.5, 3, 1.5, "Score Node\nReceives dL = 1.0", '#ffcccc', fontsize=11, text_color='red')

# SPLIT: Product Rule creates two gradient paths
ax.text(9.8, 9, "Product Rule splits the gradient:", ha='center', fontsize=11, color='red', fontstyle='italic')

# Path 1: dK₀ (goes LEFT to Key)
draw_arrow(ax, (8.5, 10), (6.5, 8.5), color='red', lw=3, ls='--')
draw_gradient_box(ax, 5, 8.5, r"Path → Key: $\frac{\partial S}{\partial K_0} = Q_0^T$")
draw_arrow(ax, (5, 7.8), (5, 7))
draw_gradient_box(ax, 5, 6.5, r"$dK_0 = dL \times Q_0^T$" + "\n" + r"$= 1.0 \times [0.8, 0.8]$" + "\n" + "= [0.8, 0.8]")

# Path 2: dQ₀ (goes LEFT to Query)
draw_arrow(ax, (8.5, 10.5), (3, 8.5), color='red', lw=3, ls='--')
draw_gradient_box(ax, 1.5, 8.5, r"Path → Query: $\frac{\partial S}{\partial Q_0} = K_0^T$")
draw_arrow(ax, (1.5, 7.8), (1.5, 7))
draw_gradient_box(ax, 1.5, 6.5, r"$dQ_0 = dL \times K_0^T$" + "\n" + r"$= 1.0 \times [0.56, 1.04]$" + "\n" + "= [0.56, 1.04]")

# ==========================================
# TIER 3 (BOTTOM): Propagate dK back to shared W_K
# ==========================================
ax.text(9, 4.5, "— Propagate dK₀ back to shared weights W_K —", ha='center', fontsize=14, color='red', fontweight='bold')

# Chain rule: K = X · W_K, so dW_K = X^T · dK
draw_arrow(ax, (5, 5.8), (5, 4.2), color='red', lw=3, ls='--', text=r"Chain: $K = X \cdot W_K$")

draw_gradient_box(ax, 5, 3.5, r"$dW_{K(word0)} = X_0^T \cdot dK_0$")

draw_arrow(ax, (5, 2.8), (5, 2.2), color='red', lw=2, ls='--')
draw_matrix(ax, 3.8, 0.5, dWk_word0, r"$dW_{K(word0)}$ (Word 1's gradient)", bg='#ffcccc')

# Word 2 path (from the right side)
draw_gradient_box(ax, 13, 3.5, r"$dW_{K(word1)} = X_1^T \cdot dK_1$")
draw_arrow(ax, (13, 2.8), (13, 2.2), color='red', lw=2, ls='--')
ax.text(13, 1.5, "(Same process\nfor Word 2)", ha='center', fontsize=11, color='gray', fontstyle='italic')

# Final accumulation
draw_arrow(ax, (6.2, 0.8), (8, 0), color='red', lw=3, ls='--')
draw_arrow(ax, (13, 1), (10.5, 0), color='red', lw=3, ls='--')
draw_gradient_box(ax, 9.2, -0.3, r"$\sum dW_K = dW_{K(word0)} + dW_{K(word1)}$" + "\n→ Update shared weights!")

plt.suptitle("Transformer Synthesis 3: Backpropagation Chain Rule Trace\n(Allocating gradients back to shared Key weights)",
             fontsize=20, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.savefig('assets/backprop_explicit_chain_rule.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Generated assets/backprop_explicit_chain_rule.png successfully!")
