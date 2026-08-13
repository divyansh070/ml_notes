import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

# --- CONFIGURATION ---
fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Colors
C_BOX = '#f8f9fa'
C_FORGET = '#e74c3c'  # Red
C_INPUT = '#2ecc71'   # Green
C_OUTPUT = '#3498db'  # Blue
C_MATH = '#f1c40f'    # Yellow
C_LINE = '#333333'

def draw_node(ax, x, y, text, radius=0.4, color=C_MATH, text_color='black'):
    circle = Circle((x, y), radius, facecolor=color, edgecolor='black', lw=2, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontsize=18, fontweight='bold', color=text_color, zorder=4)

def draw_arrow(ax, start, end, color=C_LINE, lw=2, rad=0.0):
    arrow = FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", 
                            color=color, arrowstyle='-|>', mutation_scale=20, lw=lw, zorder=2)
    ax.add_patch(arrow)

def draw_line(ax, start, end, color=C_LINE, lw=2):
    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=lw, zorder=1)

# --- BOUNDING BOX ---
rect = Rectangle((1.5, 1), 11, 6, fill=True, facecolor=C_BOX, edgecolor='black', lw=3, linestyle='--', zorder=0)
ax.add_patch(rect)
ax.text(2, 6.7, "LSTM Cell", fontsize=20, fontweight='bold', color='gray')

# --- THE TWO TRACKS ---
# Cell State (Top)
draw_line(ax, (0, 6), (14, 6), lw=4, color='gray')
draw_arrow(ax, (13, 6), (14, 6), lw=4, color='gray')
ax.text(7, 6.4, "Cell State ($C_t$) - The Gradient Superhighway", ha='center', fontsize=16, fontweight='bold', color='#555555')

# Hidden State (Bottom)
draw_line(ax, (0, 2), (14, 2), lw=3, color='gray')
draw_arrow(ax, (13, 2), (14, 2), lw=3, color='gray')
ax.text(7, 1.5, "Hidden State ($h_t$) - The Working Memory", ha='center', fontsize=16, fontweight='bold', color='#555555')

# Inputs & Outputs Text
ax.text(0.5, 6.3, "$C_{t-1}$", fontsize=18, fontweight='bold')
ax.text(13.5, 6.3, "$C_t$", fontsize=18, fontweight='bold')
ax.text(0.5, 2.3, "$h_{t-1}$", fontsize=18, fontweight='bold')
ax.text(13.5, 2.3, "$h_t$", fontsize=18, fontweight='bold')
ax.text(2.5, 0.2, "$x_t$", fontsize=18, fontweight='bold')

# --- INPUT MERGE ---
draw_arrow(ax, (2.5, 0.5), (2.5, 2), color='black') # xt entering
draw_node(ax, 2.5, 2, "Concat", radius=0.35, color='white')

# We draw a line from (2.5, 2) to (9.5, 2) that the gates pull from.
draw_line(ax, (2.5, 2), (9.5, 2), color='gray')

# --- FORGET GATE (Red) ---
x_f = 4.5
draw_arrow(ax, (x_f, 2), (x_f, 3.6), color=C_FORGET, lw=3)
draw_node(ax, x_f, 4, r"$\sigma$", color=C_FORGET, text_color='white')
draw_arrow(ax, (x_f, 4.4), (x_f, 5.6), color=C_FORGET, lw=3)
draw_node(ax, x_f, 6, r"$\otimes$", color=C_MATH)
ax.text(x_f, 3.2, "Forget\nGate", color=C_FORGET, fontsize=12, fontweight='bold', ha='center')

# --- INPUT GATE (Green) ---
x_i = 7
draw_line(ax, (x_i, 2), (x_i, 2.5), color=C_INPUT, lw=3)
# Split
draw_line(ax, (x_i, 2.5), (x_i-0.8, 2.5), color=C_INPUT, lw=3)
draw_line(ax, (x_i, 2.5), (x_i+0.8, 2.5), color=C_INPUT, lw=3)

# Left Branch (Sigmoid)
draw_arrow(ax, (x_i-0.8, 2.5), (x_i-0.8, 3.6), color=C_INPUT, lw=3)
draw_node(ax, x_i-0.8, 4, r"$\sigma$", color=C_INPUT, text_color='white')

# Right Branch (Tanh)
draw_arrow(ax, (x_i+0.8, 2.5), (x_i+0.8, 3.6), color=C_INPUT, lw=3)
draw_node(ax, x_i+0.8, 4, r"$\tanh$", color=C_INPUT, text_color='white')

# Converge to Multiply
draw_arrow(ax, (x_i-0.8, 4.4), (x_i-0.2, 4.8), color=C_INPUT, lw=3)
draw_arrow(ax, (x_i+0.8, 4.4), (x_i+0.2, 4.8), color=C_INPUT, lw=3)
draw_node(ax, x_i, 5, r"$\otimes$", color=C_MATH)

# Add to Cell State
draw_arrow(ax, (x_i, 5.4), (x_i, 5.6), color=C_INPUT, lw=3)
draw_node(ax, x_i, 6, r"$\oplus$", color=C_MATH)

ax.text(x_i-1.3, 3.2, "Input\nGate", color=C_INPUT, fontsize=12, fontweight='bold', ha='center')

# --- OUTPUT GATE (Blue) ---
x_o = 9.5
x_h = 11.5

# Sigmoid Gate
draw_arrow(ax, (x_o, 2), (x_o, 3.6), color=C_OUTPUT, lw=3)
draw_node(ax, x_o, 4, r"$\sigma$", color=C_OUTPUT, text_color='white')

# Cell State Tanh
draw_arrow(ax, (x_h, 6), (x_h, 4.4), color=C_OUTPUT, lw=3)
draw_node(ax, x_h, 4, r"$\tanh$", color=C_OUTPUT, text_color='white')

# Converge to new Hidden State
draw_arrow(ax, (x_o, 4.4), (x_h-0.3, 2.2), color=C_OUTPUT, lw=3, rad=-0.3)
draw_arrow(ax, (x_h, 3.6), (x_h, 2.4), color=C_OUTPUT, lw=3)
draw_node(ax, x_h, 2, r"$\otimes$", color=C_MATH)

ax.text(x_o-0.6, 3.2, "Output\nGate", color=C_OUTPUT, fontsize=12, fontweight='bold', ha='center')

plt.suptitle("Inside the LSTM: Controlling Memory via Mathematical Gates", fontsize=24, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('assets/lstm_cell_visual.png', dpi=300, bbox_inches='tight')
print("Saved assets/lstm_cell_visual.png")
