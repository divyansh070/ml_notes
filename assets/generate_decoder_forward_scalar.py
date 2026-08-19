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
Y_in = np.array([[1.00, 2.00]])
H_enc = np.array([[1.10, -0.90], [-1.10, 1.30]])
Masked_W = np.array([[1.00]])
Z_masked = np.array([[2.00, 4.00]])
Z_norm1 = np.array([[-1.00, 1.00]])
Cross_W = np.array([[0.04, 0.96]])
Z_cross = np.array([[-1.01, 1.21]])
Prediction = np.array([[0.01, 0.98]])

# --- FIGURE: Clean vertical flow ---
fig, ax = plt.subplots(figsize=(16, 20))
ax.set_xlim(-1, 17)
ax.set_ylim(-1, 20)
ax.axis('off')
fig.patch.set_facecolor('white')

cx = 6  # center x for main flow

# ============ ROW 1: Decoder Input (y=18) ============
draw_matrix(ax, cx, 18, Y_in, 'Decoder Input (Y)\n"<Start>" + Pos. Encoding', bg='#d5f5e3')

# ============ ROW 2: Masked Self-Attention (y=15.5) ============
draw_arrow(ax, (cx+1.2, 18), (cx+1.2, 17))
draw_box(ax, cx-1.5, 15.5, 5.5, 1.3, "Masked Self-Attention\n(1 token → trivial mask)", '#fadbd8')
draw_arrow(ax, (cx+1.2, 15.5), (cx+1.2, 14.5))
# Mask matrix (small, to the side)
draw_matrix(ax, 12, 15.5, Masked_W, "Mask\nWeights", bg='#fadbd8', cell_w=1, cell_h=0.6)
draw_arrow(ax, (11.5, 16.2), (cx+4, 16.2), text="trivial 1×1")
# Output
draw_matrix(ax, cx, 13.5, Z_masked, "Masked Attn Output", bg='#fadbd8')

# ============ ROW 3: Add & Norm 1 (y=12) ============
draw_arrow(ax, (cx+1.2, 13.5), (cx+1.2, 12.8))
# Residual skip from input
draw_arrow(ax, (cx-0.3, 18.2), (cx-1.5, 18.2), color='red')
draw_arrow(ax, (cx-1.5, 18.2), (cx-1.5, 12.3), color='red', text="Residual", text_offset=(0.8, 0))
draw_arrow(ax, (cx-1.5, 12.3), (cx-0.5, 12.3), color='red')
draw_box(ax, cx-0.5, 11.7, 4, 1, "Add & LayerNorm", '#d5f5e3')
draw_arrow(ax, (cx+1.2, 11.7), (cx+1.2, 10.8))
draw_matrix(ax, cx, 10, Z_norm1, "Normalized Z' (Query source)", bg='#d5f5e3')

# ============ ROW 4: Cross-Attention (y=7.5) ============
draw_arrow(ax, (cx+1.2, 10), (cx+1.2, 9))
ax.text(cx+1.2, 9.3, "Q (from Decoder)", ha='center', fontsize=10, fontweight='bold', color='blue')

# Encoder Blueprint H (from the side)
draw_matrix(ax, 12, 8.5, H_enc, "Encoder Blueprint H\n(EXTERNAL)", bg='#e8daef')
draw_arrow(ax, (12, 9), (cx+4.3, 8.5), color='purple', text="K, V (from Encoder)", text_offset=(0, 0.4))

draw_box(ax, cx-1.5, 7.5, 5.5, 1.3, "Cross-Attention\n(Q=Decoder, K,V=Encoder)", '#fcf3cf')

# Cross-Attention Weights
draw_arrow(ax, (cx+4, 8.0), (12, 7.5), text="")
draw_matrix(ax, 12, 6.5, Cross_W, "Cross-Attn\nWeights", bg='#fcf3cf')
ax.text(13.2, 6.3, "4% Good, 96% morning", fontsize=9, ha='center', color='gray')

draw_arrow(ax, (cx+1.2, 7.5), (cx+1.2, 6.5))
draw_matrix(ax, cx, 5.5, Z_cross, "Cross-Attn Output", bg='#fcf3cf')

# ============ ROW 5: Add & Norm 2 (y=4) ============
draw_arrow(ax, (cx+1.2, 5.5), (cx+1.2, 4.8))
# Residual skip from Z'
draw_arrow(ax, (cx-0.3, 10.3), (cx-2.5, 10.3), color='red')
draw_arrow(ax, (cx-2.5, 10.3), (cx-2.5, 4.3), color='red', text="Residual", text_offset=(0.8, 0))
draw_arrow(ax, (cx-2.5, 4.3), (cx-0.5, 4.3), color='red')
draw_box(ax, cx-0.5, 3.7, 4, 1, "Add & LayerNorm", '#d5f5e3')

# ============ ROW 6: FFN (y=2) ============
draw_arrow(ax, (cx+1.2, 3.7), (cx+1.2, 3))
draw_box(ax, cx-0.5, 1.8, 4, 1, "Feed-Forward Network\n(Expand → ReLU → Compress)", '#d6eaf8')

# ============ ROW 7: Add & Norm 3 + Output (y=0) ============
draw_arrow(ax, (cx+1.2, 1.8), (cx+1.2, 1.2))
draw_box(ax, cx-0.5, 0, 4, 1, "Add & LayerNorm → Linear → Softmax", '#d5f5e3')
draw_arrow(ax, (cx+1.2, 0), (cx+1.2, -0.8))

draw_matrix(ax, cx-0.5, -2, Prediction, 'Prediction: P("Bonjour") = 0.98', bg='#e8daef')

plt.suptitle("Transformer Synthesis 2: Complete Decoder Forward Pass (d=2)",
             fontsize=20, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('assets/decoder_forward_scalar.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Generated assets/decoder_forward_scalar.png successfully!")
