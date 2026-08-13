import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

# --- CONFIGURATION ---
C_NODE = '#f0f0f0'
C_MATH = '#f1c40f'
C_LINE = '#333333'

def draw_node(ax, x, y, text, radius=0.4, color=C_NODE, text_color='black', fontsize=14):
    circle = Circle((x, y), radius, facecolor=color, edgecolor='black', lw=2, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=text_color, zorder=4)

def draw_arrow(ax, start, end, color=C_LINE, lw=2, rad=0.0):
    arrow = FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", 
                            color=color, arrowstyle='-|>', mutation_scale=20, lw=lw, zorder=2)
    ax.add_patch(arrow)

def draw_line(ax, start, end, color=C_LINE, lw=2):
    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=lw, zorder=1)

# ==========================================
# 1. FORWARD PASS
# ==========================================
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7)
ax.axis('off')

# Bounding box roughly
rect = Rectangle((1.5, 1.5), 9.5, 4.5, fill=True, facecolor='#f8f9fa', edgecolor='gray', lw=2, linestyle='--', zorder=0)
ax.add_patch(rect)

# Track: Cell State
draw_line(ax, (1, 5), (12, 5), lw=3, color='gray')
ax.text(1, 5.3, r"$C_0 = 0$", ha='center', fontsize=16, fontweight='bold')
ax.text(12, 5.3, r"$C_1 = 0.84$", ha='center', fontsize=16, fontweight='bold')
# Label formula
ax.text(6, 6, r"$C_1 = (0.88 \cdot 0) + (0.88 \cdot 0.96) = 0.84$", ha='center', fontsize=14, fontweight='bold', bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, pad=3))

# Input
ax.text(6, 0.5, r"$x_1 = 2$", ha='center', fontsize=16, fontweight='bold')
draw_line(ax, (6, 0.8), (6, 1.8), lw=2)
draw_line(ax, (3, 1.8), (9, 1.8), lw=2)

white_bbox = dict(facecolor='white', edgecolor='none', alpha=0.9, pad=2)

# Forget Gate
draw_arrow(ax, (3, 1.8), (3, 2.6))
draw_node(ax, 3, 3, r"$\sigma$", radius=0.3, color='#e74c3c', text_color='white')
ax.text(3, 2.3, r"$f_1=0.88$", ha='center', va='center', fontsize=12, fontweight='bold', color='red', bbox=white_bbox, zorder=5)
draw_arrow(ax, (3, 3.4), (3, 4.6))
draw_node(ax, 3, 5, r"$\otimes$", radius=0.3, color=C_MATH)

# Input Gate & Candidate
draw_arrow(ax, (5, 1.8), (5, 2.6))
draw_node(ax, 5, 3, r"$\sigma$", radius=0.3, color='#2ecc71', text_color='white')
ax.text(5, 2.3, r"$i_1=0.88$", ha='center', va='center', fontsize=12, fontweight='bold', color='green', bbox=white_bbox, zorder=5)

draw_arrow(ax, (7, 1.8), (7, 2.6))
draw_node(ax, 7, 3, r"$\tanh$", radius=0.3, color='#2ecc71', text_color='white', fontsize=10)
ax.text(7, 2.3, r"$\tilde{C}_1=0.96$", ha='center', va='center', fontsize=12, fontweight='bold', color='green', bbox=white_bbox, zorder=5)

draw_arrow(ax, (5, 3.4), (5.8, 3.8))
draw_arrow(ax, (7, 3.4), (6.2, 3.8))
draw_node(ax, 6, 4, r"$\otimes$", radius=0.3, color=C_MATH)
draw_arrow(ax, (6, 4.4), (6, 4.6))
draw_node(ax, 6, 5, r"$\oplus$", radius=0.3, color=C_MATH)

# Output Gate
draw_arrow(ax, (9, 1.8), (9, 2.6))
draw_node(ax, 9, 3, r"$\sigma$", radius=0.3, color='#3498db', text_color='white')
ax.text(9, 2.3, r"$o_1=0.88$", ha='center', va='center', fontsize=12, fontweight='bold', color='blue', bbox=white_bbox, zorder=5)

draw_arrow(ax, (10, 5), (10, 4.4))
draw_node(ax, 10, 4, r"$\tanh$", radius=0.3, color='#3498db', text_color='white', fontsize=10)

draw_arrow(ax, (9, 3.4), (9.7, 3.6))
draw_arrow(ax, (10, 3.6), (10, 3.4)) # from tanh to mult
draw_node(ax, 10, 3, r"$\otimes$", radius=0.3, color=C_MATH)

# Final hidden state
draw_arrow(ax, (10.4, 3), (11.5, 3))
ax.text(12, 3, r"$h_1 = 0.61$", ha='center', fontsize=16, fontweight='bold')
ax.text(10, 2.1, r"$0.88 \cdot \tanh(0.84)$", ha='center', fontsize=12, fontweight='bold', bbox=dict(facecolor='white', edgecolor='gray', pad=2), zorder=5)

plt.suptitle("LSTM Forward Pass: Calculating State Values", fontsize=20, fontweight='bold', y=0.95)
plt.savefig('assets/lstm_math_forward.png', dpi=300, bbox_inches='tight')
plt.close()


# ==========================================
# 2. BACKWARD PASS (BPTT)
# ==========================================
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7)
ax.axis('off')

# Same Bounding box
rect = Rectangle((1.5, 1.5), 9.5, 4.5, fill=True, facecolor='#f8f9fa', edgecolor='gray', lw=2, linestyle='--', zorder=0)
ax.add_patch(rect)

# Draw baseline structural elements
draw_line(ax, (1, 5), (12, 5), lw=3, color='gray') # Cell state
draw_line(ax, (6, 0.8), (6, 1.8), lw=2, color='gray') # Input stems
draw_line(ax, (3, 1.8), (9, 1.8), lw=2, color='gray')
draw_node(ax, 3, 3, r"$\sigma$", radius=0.3, color='#e74c3c', text_color='white')
draw_node(ax, 3, 5, r"$\otimes$", radius=0.3, color=C_MATH)
draw_node(ax, 5, 3, r"$\sigma$", radius=0.3, color='#2ecc71', text_color='white')
draw_node(ax, 7, 3, r"$\tanh$", radius=0.3, color='#2ecc71', text_color='white', fontsize=10)
draw_node(ax, 6, 4, r"$\otimes$", radius=0.3, color=C_MATH)
draw_node(ax, 6, 5, r"$\oplus$", radius=0.3, color=C_MATH)
draw_node(ax, 9, 3, r"$\sigma$", radius=0.3, color='#3498db', text_color='white')
draw_node(ax, 10, 4, r"$\tanh$", radius=0.3, color='#3498db', text_color='white', fontsize=10)
draw_node(ax, 10, 3, r"$\otimes$", radius=0.3, color=C_MATH)

# Base text
ax.text(6, 0.5, r"$x_1 = 2$", ha='center', fontsize=14, fontweight='bold', color='gray')
ax.text(1, 5.3, r"$C_0 = 0$", ha='center', fontsize=14, fontweight='bold', color='gray')

red_bbox = dict(facecolor='white', edgecolor='red', alpha=1.0, pad=3)

# 1. Error enters
ax.text(12, 3, r"$dh_1 = 2$", ha='center', va='center', fontsize=16, fontweight='bold', color='red', bbox=red_bbox)
draw_arrow(ax, (11.2, 3), (10.5, 3), color='red', lw=3)

# 2. Split at mult (10, 3)
draw_arrow(ax, (10, 3.4), (10, 3.8), color='red', lw=3)
ax.text(10, 4.6, r"$dC_1 = 0.92$", ha='center', va='center', fontsize=14, fontweight='bold', color='red', bbox=red_bbox, zorder=5)
draw_arrow(ax, (9.8, 5), (9.0, 5), color='red', lw=3) # flowing left

draw_arrow(ax, (9.6, 3), (9.3, 3), color='red', lw=3)
ax.text(8.3, 2.5, r"$do_1 = 1.38$", ha='center', va='center', fontsize=14, fontweight='bold', color='red', bbox=red_bbox, zorder=5)

# 3. Flowing across cell state
draw_arrow(ax, (8, 5.2), (6.5, 5.2), color='red', lw=3) # Above line
ax.text(7.2, 5.5, r"$dC_1 = 0.92$", ha='center', va='center', fontsize=12, fontweight='bold', color='red', bbox=red_bbox, zorder=5)

# 4. Split at addition (6, 5)
draw_arrow(ax, (5.8, 5.2), (3.5, 5.2), color='red', lw=3) # left to forget
ax.text(4.5, 5.5, r"$dC_1 = 0.92$", ha='center', va='center', fontsize=12, fontweight='bold', color='red', bbox=red_bbox, zorder=5)

# 5. Forget gate error
draw_arrow(ax, (2.6, 5), (2.6, 4.4), color='red', lw=3) # down from mult
ax.text(2.6, 4, r"$df_1 = 0$", ha='center', va='center', fontsize=12, fontweight='bold', color='red', bbox=red_bbox, zorder=5)

# 6. Input gate / candidate error
draw_arrow(ax, (6, 4.6), (6, 4.4), color='red', lw=3) # down from add
draw_arrow(ax, (5.7, 3.8), (5, 3.4), color='red', lw=3) # to input
ax.text(4.3, 3.6, r"$di_1 = 0.88$", ha='center', va='center', fontsize=12, fontweight='bold', color='red', bbox=red_bbox, zorder=5)

draw_arrow(ax, (6.3, 3.8), (7, 3.4), color='red', lw=3) # to cand
ax.text(7.7, 3.6, r"$d\tilde{C}_1 = 0.81$", ha='center', va='center', fontsize=12, fontweight='bold', color='red', bbox=red_bbox, zorder=5)


plt.suptitle("LSTM Backward Pass: Error Routing & Gradient Calculus", fontsize=20, fontweight='bold', y=0.95)
plt.savefig('assets/lstm_math_backward.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved assets/lstm_math_forward.png and assets/lstm_math_backward.png")
