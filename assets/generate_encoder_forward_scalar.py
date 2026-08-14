import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
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
    return x + w/2, y  # bottom center

def draw_arrow(ax, start, end, color='#555', lw=2, text=None, text_offset=(0, 0.2)):
    arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=18, color=color, lw=lw, zorder=1)
    ax.add_patch(arrow)
    if text:
        mx, my = (start[0]+end[0])/2 + text_offset[0], (start[1]+end[1])/2 + text_offset[1]
        ax.text(mx, my, text, ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9), zorder=4)

def draw_box(ax, x, y, w, h, text, bg, fontsize=12):
    rect = Rectangle((x, y), w, h, facecolor=bg, edgecolor='black', lw=2, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', zorder=3)

# --- DATA ---
X = np.array([[0.80, 0.80], [0.74, 1.44]])
W_K = np.array([[0.5, 0.5], [0.2, 0.8]])
W_V = np.array([[2.0, 0.0], [0.0, 2.0]])
Q = X.copy(); K = X @ W_K; V = X @ W_V
Scores = Q @ K.T
Weights = np.array([[0.42, 0.58], [0.18, 0.82]])
Z = Weights @ V
Z_res = X + Z
Z_norm = np.array([[1.0, -1.0], [-1.0, 1.0]])
FFN_out = np.array([[3.0, 3.0], [-1.0, 2.0]])
H = np.array([[1.1, -0.9], [-1.1, 1.3]])

# --- FIGURE ---
fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(-1, 21)
ax.set_ylim(-1, 14)
ax.axis('off')
fig.patch.set_facecolor('white')

# ============ ROW 1: Input → Q, K, V (y=11) ============
draw_matrix(ax, 0, 11, X, "Input X\n[Good, morning]", bg='#d5f5e3')
draw_arrow(ax, (2.5, 12), (4.5, 13), text=r"$\times W_Q = I$")
draw_arrow(ax, (2.5, 11.8), (4.5, 11.8), text=r"$\times W_K$")
draw_arrow(ax, (2.5, 11.5), (4.5, 10.5), text=r"$\times W_V$")
draw_matrix(ax, 5, 12.5, Q, "Query Q", bg='#d6eaf8')
draw_matrix(ax, 5, 11, K, "Key K", bg='#fadbd8')
draw_matrix(ax, 5, 9.5, V, "Value V", bg='#fcf3cf')

# ============ ROW 2: Scores → Softmax → Weights (y=11, right side) ============
draw_arrow(ax, (7.5, 13.2), (9, 12.5), text="Q")
draw_arrow(ax, (7.5, 11.8), (9, 12.2), text=r"$K^T$")
draw_matrix(ax, 9.5, 11.5, Scores, r"Raw Scores ($Q \cdot K^T$)", bg='#fef9e7')
ax.text(12, 11.3, r"$\div \sqrt{d_k}=1.41$", fontsize=10, fontweight='bold', color='gray')

draw_arrow(ax, (12, 12.2), (13.5, 12.2), text="Softmax →")
draw_matrix(ax, 14, 11.5, Weights, "Attention Weights", bg='#e8daef')

# ============ ROW 3: Weighted sum → Z (y=9) ============
draw_arrow(ax, (16.5, 11.5), (16.5, 10.2), text="× V ↓")
draw_arrow(ax, (7.5, 10), (14.5, 9.5), text="V feeds in", text_offset=(0, 0.3))
draw_matrix(ax, 14.5, 8.5, Z, "Contextualized Z", bg='#d6eaf8')

# ============ ROW 4: Residual + LayerNorm (y=6.5) ============
draw_arrow(ax, (16, 8.5), (11, 7.2), text="")
draw_arrow(ax, (1.2, 11), (1.2, 7.2), color='red', text="Residual\n(skip)", text_offset=(0.8, 0))
draw_arrow(ax, (1.2, 7.2), (8.5, 7.2), color='red')
draw_box(ax, 8.5, 6.5, 4, 1.2, "Add & LayerNorm\nX + Z → Normalize", '#d5f5e3', fontsize=11)
draw_matrix(ax, 13.5, 6.2, Z_norm, "Normalized Z'", bg='#d5f5e3')

# ============ ROW 5: FFN (y=4) ============
draw_arrow(ax, (14.5, 6.2), (14.5, 5.3))
draw_box(ax, 12.5, 4, 4, 1.2, "Feed-Forward Network\n(2→8→2, ReLU)", '#d6eaf8', fontsize=11)
draw_arrow(ax, (14.5, 4), (14.5, 3.2))
draw_matrix(ax, 13.5, 2, FFN_out, "FFN Output", bg='#d6eaf8')

# ============ ROW 6: Final Add & Norm → Blueprint H (y=0) ============
draw_arrow(ax, (14.5, 2), (14.5, 1.2))
draw_arrow(ax, (13.5, 6.5), (13, 1), color='red', text="Residual\n(skip)", text_offset=(-1.2, 0))
draw_box(ax, 12.5, 0, 4, 1, "Add & LayerNorm", '#d5f5e3', fontsize=11)
draw_arrow(ax, (12.5, 0.5), (10.5, 0.5))
draw_matrix(ax, 7.5, -0.2, H, "FINAL Encoder Output\n(Blueprint H)", bg='#e8daef')

plt.suptitle("Transformer Synthesis 1: Complete Encoder Forward Pass (d=2, h=1)",
             fontsize=20, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('assets/encoder_forward_scalar.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Generated assets/encoder_forward_scalar.png successfully!")
