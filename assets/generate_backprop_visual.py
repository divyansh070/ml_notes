import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
import numpy as np

# --- CONFIGURATION & HELPERS ---
C_BG = '#f8f9fa'
C_MATH = '#f1c40f'
C_TEXT = 'black'

def draw_node(ax, x, y, text, radius=0.4, color=C_MATH, text_color='black', fontsize=16):
    circle = Circle((x, y), radius, facecolor=color, edgecolor='black', lw=2, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=text_color, zorder=4)

def draw_wire(ax, start, end, text=None, color='#333', text_color='black', text_y_offset=0, bold=True, rad=0.0, lw=2, ls='-'):
    arrow = FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", arrowstyle='-|>', mutation_scale=15, color=color, lw=lw, linestyle=ls, zorder=1)
    ax.add_patch(arrow)
    if text:
        mid_x = (start[0] + end[0]) / 2
        mid_y = ((start[1] + end[1]) / 2) + text_y_offset
        f_weight = 'bold' if bold else 'normal'
        ax.text(mid_x, mid_y, text, ha='center', va='center', fontsize=10, fontweight=f_weight, color=text_color,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9), zorder=5)

def draw_arrow(ax, start, end, text=None, rad=0.0, color='#333', lw=2):
    draw_wire(ax, start, end, text=text, rad=rad, color=color, lw=lw)

def draw_matrix(ax, x, y, matrix, title, color='gray'):
    rows, cols = matrix.shape
    height, width = rows * 0.8, cols * 1.5
    ax.text(x + width/2, y + height + 0.3, title, ha='center', va='bottom', fontsize=12, fontweight='bold')
    for r in range(rows):
        for c in range(cols):
            val = matrix[rows-1-r, c]
            rect = Rectangle((x + c*1.5, y + r*0.8), 1.5, 0.8, facecolor='white', edgecolor='black', lw=1, zorder=2)
            ax.add_patch(rect)
            ax.text(x + c*1.5 + 0.75, y + r*0.8 + 0.4, f"{val:.2f}", ha='center', va='center', fontsize=10, fontweight='bold')
    return x + width, y + height

def draw_math_box(ax, x, y, text, color='red', fontsize=10, bold=True):
    f_weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight=f_weight, color=color,
            bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.3', alpha=0.95), zorder=6)

def draw_gradient_wire(ax, start, end, derivative_text, gradient_val, color='red', fontsize=9, rad=0.0):
    """Draws a red backprop wire annotated with the partial derivative and gradient."""
    arrow = FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", arrowstyle='-|>', mutation_scale=15, color=color, lw=2.5, linestyle='--', zorder=1)
    ax.add_patch(arrow)
    mid_x, mid_y = (start[0]+end[0])/2, (start[1]+end[1])/2
    wire_text = r"$\times$ $\frac{\partial \text{Node}}{\partial \text{Path}}$" + "\n" + derivative_text
    ax.text(mid_x, mid_y + 0.3, wire_text, ha='center', va='center', fontsize=fontsize, fontweight='normal', color=color,
            bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.2', alpha=0.9), zorder=5)
    draw_math_box(ax, end[0], end[1]+0.8, gradient_val, 'red', fontsize=10, bold=True)

# ==========================================
# 3. BACKPASS TRACE: EXPLICIT CHAIN RULE
# ==========================================
fig, ax = plt.subplots(figsize=(14, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis('off')

# Data Setup
Q0 = np.array([[0.8, 0.8]])
K0 = np.array([[0.56, 1.04]])
Softmax_Weights = np.array([[0.8, 0.2], [0.1, 0.9]])

# Faded Forward Structure
draw_matrix(ax, 0.5, 9, Q0, "Query Q0", color='lightgray')
draw_matrix(ax, 6.5, 9.1, K0, "Key K0", color='lightgray')
rect_node = Rectangle((10.5, 9.1), 3, 1.6, facecolor='lightyellow', edgecolor='black', lw=1)
ax.add_patch(rect_node)
ax.text(12, 9.9, r"Node: $Q_0 \cdot K^T$ (Attn Score)", ha='center', fontsize=11, fontweight='bold')
ax.text(12, 9.5, r"Result $Score(0\cdot0)=1.28$", ha='center', fontsize=9, color='gray')
draw_matrix(ax, 15.5, 9.1, Softmax_Weights, "Attn Weights", color='lightgray')

# --- GRADIENT ERROR ARRIVES (dL = 2.0) ---
ax.text(17.5, 10.5, "Total Error penalty ($dL$) enters:", ha='center', fontsize=10, fontweight='bold', color='red', bbox=dict(facecolor='#ffcccc', edgecolor='red'))
draw_gradient_wire(ax, (17, 10.5), (16, 9.9), derivative_text="Allocating 2.0 down path", gradient_val="1.0")

# --- SPLIT 1: At Attention Scores node (Product Rule) ---
draw_gradient_wire(ax, (12, 9), (7, 9.1), derivative_text=r"$dL_{(1.0)} \cdot Q_0^T = [0.8, 0.8]$", gradient_val="dK0")
draw_gradient_wire(ax, (12, 9), (1, 9), derivative_text=r"$dL_{(1.0)} \cdot K_0^T = [0.56, 1.04]$", gradient_val="dQ0", rad=0.2)

# Explicit local gradients boxes
draw_math_box(ax, 7, 9.9, r"$dK_0 = [0.8, 0.8]$", 'red')
draw_math_box(ax, 1, 9.8, r"$dQ_0 = [0.56, 1.04]$", 'red')

# --- SPLIT 2: All the way back to input Weights (Accumulation) ---
draw_node(ax, 10, 1.5, r"$W_K$", radius=0.6, color=C_MATH)
ax.text(10, 0.6, "SHARED Weights", ha='center', fontsize=11, fontweight='bold', color='#333')

# Allocation path 1: contribution from Word 1 (K0)
draw_gradient_wire(ax, (6.5, 9.1), (9.4, 2.1), derivative_text=r"Chain rule (K=XW): $X_0^T \cdot dK_0$", gradient_val=r"$dW_{K(word0)}$", color='red')
local_wk_0 = np.array([[0.64, 0.64], [0.64, 0.64]])
draw_matrix(ax, 7.5, 4, local_wk_0, r"$dW_{K(word0)}$" + "\n" + r"Word 1's local insight", color='red')

# Allocation path 2: contribution from Word 2 (K1)
draw_wire(ax, (14, 6), (10.6, 2.1), color='red', text=r"$X_1^T \cdot dK_1$", lw=2.5, ls='--')
draw_math_box(ax, 11, 4.8, r"$dW_{K(word1)}$", 'red')

# THE SUMMATION NODE
draw_node(ax, 11, 2.1, r"$\oplus$", radius=0.2, color='red')
draw_arrow(ax, (10, 2.1), (10.8, 2.1), color='red', lw=2)
draw_arrow(ax, (11, 2.3), (11, 2.5), color='red', lw=2)
draw_math_box(ax, 11.5, 3.1, r"$\sum dW_K$ = Final Update", 'red', fontsize=12, bold=True)

plt.suptitle("Transformer synthesis 3: explicit Chain Rule backprop Trace (allocating K gradients d=2)", fontsize=20, fontweight='bold', y=0.98)
plt.savefig('assets/backprop_explicit_chain_rule.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated assets/backprop_explicit_chain_rule.png successfully!")
