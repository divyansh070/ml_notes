import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

def draw_node(ax, x, y, text, color, radius=0.35):
    circle = Circle((x, y), radius, facecolor=color, edgecolor='black', lw=2, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontsize=12, fontweight='bold', color='black' if color != '#333' else 'white', zorder=4)

def draw_wire(ax, start, end, text=None, color='#333333', text_color='black', text_y_offset=0):
    arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=15, color=color, lw=2, zorder=1)
    ax.add_patch(arrow)
    if text:
        mid_x = (start[0] + end[0]) / 2
        mid_y = ((start[1] + end[1]) / 2) + text_y_offset
        ax.text(mid_x, mid_y, text, ha='center', va='center', fontsize=10, fontweight='bold', color=text_color,
                bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.2', alpha=0.9), zorder=5)

# ==========================================
# 1. FORWARD PASS: The Computational Graph
# ==========================================
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Cell State Conveyor Belt
ax.plot([1, 13], [6, 6], color='gray', lw=4, zorder=0)
ax.text(1, 6.3, r"$C_0 = 0$", ha='center', fontsize=14, fontweight='bold')
ax.text(13, 6.3, r"$C_1 = 0.84$", ha='center', fontsize=14, fontweight='bold')

# Input
ax.text(6, 0.5, r"Input: $x_1 = 2$", ha='center', fontsize=14, fontweight='bold')
ax.plot([6, 6], [0.8, 1.5], color='black', lw=2)
ax.plot([2.5, 10.5], [1.5, 1.5], color='black', lw=2)

# Forget Gate (Red)
draw_wire(ax, (2.5, 1.5), (2.5, 2.65), text=r"$W_f \cdot x_1 = 2$")
draw_node(ax, 2.5, 3, r"$\sigma$", '#ff9999')
draw_wire(ax, (2.5, 3.35), (2.5, 5.65), text=r"$f_1 = 0.88$")
draw_node(ax, 2.5, 6, r"$\otimes$", '#f1c40f')
ax.text(2.5, 6.7, r"$0.88 \times 0 = \mathbf{0}$", ha='center', fontsize=10, bbox=dict(facecolor='#e8f6f3', edgecolor='gray'))

# Input Gate (Green)
draw_wire(ax, (5, 1.5), (5, 2.65), text=r"$W_i \cdot x_1 = 2$")
draw_node(ax, 5, 3, r"$\sigma$", '#99ff99')
draw_wire(ax, (5, 3.35), (6, 4.2), text=r"$i_1 = 0.88$")

# Candidate (Green)
draw_wire(ax, (7, 1.5), (7, 2.65), text=r"$W_c \cdot x_1 = 2$")
draw_node(ax, 7, 3, r"$\mathrm{tanh}$", '#99ff99')
draw_wire(ax, (7, 3.35), (6, 4.2), text=r"$\widetilde{C}_1 = 0.96$", text_y_offset=-0.2)

# Merge Input & Candidate
draw_node(ax, 6, 4.5, r"$\otimes$", '#f1c40f')
draw_wire(ax, (6, 4.85), (6, 5.65), text=r"$0.88 \times 0.96 = \mathbf{0.84}$")
draw_node(ax, 6, 6, r"$\oplus$", '#f1c40f')
ax.text(6, 6.7, r"$0 + 0.84 = \mathbf{0.84}$", ha='center', fontsize=10, bbox=dict(facecolor='#e8f6f3', edgecolor='gray'))

# Output Gate (Blue)
draw_wire(ax, (10.5, 1.5), (10.5, 2.65), text=r"$W_o \cdot x_1 = 2$")
draw_node(ax, 10.5, 3, r"$\sigma$", '#99ccff')
draw_wire(ax, (10.5, 3.35), (11.5, 4.2), text=r"$o_1 = 0.88$")

# Tanh of Cell State
draw_wire(ax, (11.5, 6), (11.5, 5.35))
draw_node(ax, 11.5, 5, r"$\mathrm{tanh}$", '#99ccff')
draw_wire(ax, (11.5, 4.65), (11.5, 4.2), text=r"$\mathrm{tanh}(0.84) = 0.69$", text_y_offset=0.2)

# Final Hidden State
draw_node(ax, 11.5, 4, r"$\otimes$", '#f1c40f')
draw_wire(ax, (11.5, 3.65), (11.5, 2.5), text=r"$h_1 = 0.88 \times 0.69 = \mathbf{0.61}$", text_y_offset=-0.4)

plt.suptitle("LSTM Forward Pass: Explicit Math Tracking", fontsize=18, fontweight='bold')
plt.savefig('assets/lstm_math_forward.png', dpi=300, bbox_inches='tight')
plt.close()


# ==========================================
# 2. BACKWARD PASS: Explicit Chain Rule
# ==========================================
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Structural elements (faded)
ax.plot([1, 13], [6, 6], color='lightgray', lw=4, zorder=0)
draw_node(ax, 2.5, 3, r"$\sigma$", '#ffe6e6')
draw_node(ax, 5, 3, r"$\sigma$", '#e6ffe6')
draw_node(ax, 7, 3, r"$\mathrm{tanh}$", '#e6ffe6')
draw_node(ax, 10.5, 3, r"$\sigma$", '#e6f2ff')
draw_node(ax, 11.5, 5, r"$\mathrm{tanh}$", '#e6f2ff')

# Error enters
ax.text(12.5, 2.5, r"Total Error: $dh_1 = 2$", ha='center', fontsize=12, fontweight='bold', color='red', bbox=dict(facecolor='#ffcccc', edgecolor='red'))
draw_wire(ax, (12.5, 2.8), (11.8, 3.8), color='red')

# Split 1: To Output Gate
draw_wire(ax, (11.3, 3.8), (10.7, 3.3), text=r"$dh_1 \times \mathrm{tanh}(C_1)$" + "\n" + r"$2 \times 0.69 = \mathbf{1.38}$", color='red')
ax.text(10.5, 2.2, r"$\times \sigma' \rightarrow 1.38 \times 0.11 = \mathbf{0.15}$", ha='center', color='red', fontsize=9, bbox=dict(facecolor='white', edgecolor='red'))

# Split 2: To Cell State (Through Tanh)
draw_wire(ax, (11.5, 4.2), (11.5, 4.6), color='red')
ax.text(12.8, 4.4, r"Chain Rule:" + "\n" + r"$dh_1 \times o_1$" + "\n" + r"$2 \times 0.88 = 1.76$", ha='center', color='red', fontsize=9, bbox=dict(facecolor='white', edgecolor='red'))
draw_wire(ax, (11.5, 5.4), (11.5, 5.8), text=r"$\times \mathrm{tanh}'$" + "\n" + r"$1.76 \times 0.52 = \mathbf{0.92}$", color='red')

# Flow across Cell State
draw_wire(ax, (11.2, 6.2), (6.2, 6.2), text=r"Gradient Superhighway: $dC_1 = 0.92$", color='red')

# Split 3: To Forget Gate (Flows down)
draw_wire(ax, (2.5, 5.8), (2.5, 3.4), text=r"$dC_1 \times C_0$" + "\n" + r"$0.92 \times 0 = \mathbf{0}$", color='red')
ax.text(2.5, 2.2, r"$\times \sigma' \rightarrow 0 \times 0.11 = \mathbf{0}$", ha='center', color='red', fontsize=9, bbox=dict(facecolor='white', edgecolor='red'))

# Split 4: Down to Input / Candidate
draw_wire(ax, (5.8, 5.8), (5.8, 4.7), color='red')
# To Input
draw_wire(ax, (5.5, 4.3), (5.2, 3.4), text=r"$dC_1 \times \widetilde{C}_1$" + "\n" + r"$0.92 \times 0.96 = \mathbf{0.88}$", color='red', text_y_offset=0.2)
ax.text(5, 2.2, r"$\times \sigma' \rightarrow 0.88 \times 0.11 = \mathbf{0.10}$", ha='center', color='red', fontsize=9, bbox=dict(facecolor='white', edgecolor='red'))
# To Candidate
draw_wire(ax, (6.1, 4.3), (6.8, 3.4), text=r"$dC_1 \times i_1$" + "\n" + r"$0.92 \times 0.88 = \mathbf{0.81}$", color='red', text_y_offset=0.2)
ax.text(7.5, 2.2, r"$\times \mathrm{tanh}' \rightarrow 0.81 \times 0.08 = \mathbf{0.06}$", ha='center', color='red', fontsize=9, bbox=dict(facecolor='white', edgecolor='red'))

plt.suptitle("LSTM Backward Pass: Exposing the Chain Rule", fontsize=18, fontweight='bold')
plt.savefig('assets/lstm_math_backward.png', dpi=300, bbox_inches='tight')
plt.close()

print("Computational Graphs generated successfully!")