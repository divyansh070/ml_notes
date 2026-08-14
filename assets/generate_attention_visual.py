import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

# --- CONFIGURATION ---
C_MAT = '#f8f9fa'    
C_HEAD = '#e5e7e9'
C_HIGHLIGHT = '#d5f5e3'
C_QKV = '#d6eaf8'

def draw_matrix(ax, x, y, matrix, title, highlight_row=None):
    rows, cols = matrix.shape
    width, height = cols * 1.2, rows * 1.2
    
    # Title
    ax.text(x + width/2, y + height + 0.3, title, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    for r in range(rows):
        for c in range(cols):
            val = matrix[r, c]
            bg_color = C_HIGHLIGHT if r == highlight_row else 'white'
            rect = Rectangle((x + c*1.2, y + (rows-1-r)*1.2), 1.2, 1.2, facecolor=bg_color, edgecolor='black', lw=1)
            ax.add_patch(rect)
            ax.text(x + c*1.2 + 0.6, y + (rows-1-r)*1.2 + 0.6, f"{val:.1f}", ha='center', va='center', fontsize=11, fontweight='bold')
    return x + width, y + height

def draw_arrow(ax, start, end, text=None):
    arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=15, color='gray', lw=2)
    ax.add_patch(arrow)
    if text:
        mid_x, mid_y = (start[0]+end[0])/2, (start[1]+end[1])/2
        ax.text(mid_x, mid_y+0.2, text, ha='center', va='bottom', fontsize=10, fontweight='bold', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.2'))

# ==========================================
# SCALED DOT-PRODUCT ATTENTION GRAPH
# ==========================================
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 24)
ax.set_ylim(0, 10)
ax.axis('off')

# Data setup: Sequence length = 2 (e.g., "Bank", "River"), d_k = 3
Q = np.array([[2, 0, 1], [0, 2, 0]]) # Query Matrix
KT = np.array([[2, 0], [0, 2], [1, 0]]) # Key Matrix (Transposed)
V = np.array([[10, 0], [0, 10]]) # Value Matrix

Raw_Scores = np.dot(Q, KT) # [[5, 0], [0, 4]]
Scale_Factor = 2 # Assuming sqrt(d_k) roughly = 2 for easy math
Scaled_Scores = Raw_Scores / Scale_Factor # [[2.5, 0], [0, 2.0]]

# Softmax (Simulated for clear percentages)
Attention_Weights = np.array([[0.9, 0.1], [0.1, 0.9]]) 
Final_Output = np.dot(Attention_Weights, V) # [[9, 1], [1, 9]]

# 1. Q and K^T
draw_matrix(ax, 1, 6, Q, "Query (Q)\n[Words x $d_k$]", highlight_row=0)
ax.text(5.5, 7.2, "X", ha='center', va='center', fontsize=20, fontweight='bold')
draw_matrix(ax, 6.5, 5.4, KT, "Key Transposed ($K^T$)\n[$d_k$ x Words]")

# Arrow to Dot Product
draw_arrow(ax, (9.5, 7.2), (11, 7.2), "Dot Product")

# 2. Raw Scores
draw_matrix(ax, 11.5, 6, Raw_Scores, "Raw Scores\n($Q \cdot K^T$)", highlight_row=0)

# Arrow to Scaling
draw_arrow(ax, (14.5, 7.2), (16, 7.2), "Scale\n($\div \sqrt{d_k}$)")

# 3. Scaled Scores
draw_matrix(ax, 16.5, 6, Scaled_Scores, "Scaled Scores", highlight_row=0)

# Arrow to Softmax (Downwards)
draw_arrow(ax, (17.7, 5.8), (17.7, 4.5), "Softmax\n(Row-wise)")

# 4. Attention Weights
draw_matrix(ax, 16.5, 1.5, Attention_Weights, "Attention Weights\n(Probabilities)", highlight_row=0)

# Arrow to Final Multiply
draw_arrow(ax, (15.5, 2.7), (14, 2.7), "Multiply")

# 5. Value Matrix
ax.text(13.3, 2.7, "X", ha='center', va='center', fontsize=20, fontweight='bold')
draw_matrix(ax, 9.5, 1.5, V, "Value (V)\n[Words x $d_v$]")

# Arrow to Output
draw_arrow(ax, (9, 2.7), (7.5, 2.7), "=")

# 6. Final Output
draw_matrix(ax, 4, 1.5, Final_Output, "Contextualized Output\n[Words x $d_v$]", highlight_row=0)

# Explanation Box for the highlighted row
explanation = (
    "Tracing Word 1 (Highlighted Green):\n"
    "1. Word 1's Query dot-products with every word's Key.\n"
    "2. We scale the score and apply Softmax to get percentages.\n"
    "3. Word 1 decides to pay 90% attention to itself, and 10% to Word 2.\n"
    "4. The final output for Word 1 is a blend of the Values: (0.9 * V1) + (0.1 * V2)."
)
ax.text(1, 0, explanation, ha='left', va='bottom', fontsize=12, fontweight='bold', bbox=dict(facecolor='#fcf3cf', edgecolor='orange', pad=0.5))

plt.suptitle("Topic 2: Scaled Dot-Product Attention (Matrix Math)", fontsize=20, fontweight='bold', y=0.98)
plt.savefig('assets/scaled_dot_product_math.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated assets/scaled_dot_product_math.png successfully!")
