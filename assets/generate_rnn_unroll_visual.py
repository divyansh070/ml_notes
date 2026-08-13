import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')

# Colors
C_NODE = '#f0f0f0'
C_WHX = '#1f77b4' # Blue
C_WHH = '#d62728' # Red
C_WYH = '#2ca02c' # Green

def draw_node(x, y, text, radius=0.4):
    circle = Circle((x, y), radius, facecolor=C_NODE, edgecolor='black', lw=2, zorder=2)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontsize=18, fontweight='bold', zorder=3)

def draw_arrow(start, end, color, label='', offset=(0, 0), rad=0.0):
    arrow = FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", 
                            color=color, arrowstyle='-|>', mutation_scale=25, lw=3,
                            shrinkA=25, shrinkB=25, zorder=1)
    ax.add_patch(arrow)
    if label:
        lx = (start[0] + end[0]) / 2 + offset[0]
        ly = (start[1] + end[1]) / 2 + offset[1]
        ax.text(lx, ly, label, color=color, fontsize=16, fontweight='bold', ha='center', va='center')

# --- FOLDED RNN ---
draw_node(2, 1.5, "$x$")
draw_node(2, 3.5, "$h$")
draw_node(2, 5.5, "$y$")

# Folded arrows
draw_arrow((2, 1.5), (2, 3.5), C_WHX, "$W_{hx}$", offset=(-0.4, 0))
draw_arrow((2, 3.5), (2, 5.5), C_WYH, "$W_{yh}$", offset=(-0.4, 0))

# Folded self loop
loop = FancyArrowPatch((2.2, 3.8), (2.4, 3.2), connectionstyle="arc3,rad=-1.5", 
                       color=C_WHH, arrowstyle='-|>', mutation_scale=25, lw=3, zorder=1)
ax.add_patch(loop)
ax.text(2.9, 3.5, "$W_{hh}$", color=C_WHH, fontsize=16, fontweight='bold', ha='center', va='center')

# Equals sign
ax.text(4.5, 3.5, "=", fontsize=50, fontweight='bold', ha='center', va='center')

# --- UNROLLED RNN ---
times = [1, 2, 3]
x_pos = [7, 10, 13]

for i, t in enumerate(times):
    px = x_pos[i]
    draw_node(px, 1.5, f"$x_{t}$")
    draw_node(px, 3.5, f"$h_{t}$")
    draw_node(px, 5.5, f"$y_{t}$")
    
    # Input to hidden
    draw_arrow((px, 1.5), (px, 3.5), C_WHX, "$W_{hx}$", offset=(-0.4, 0))
    # Hidden to output
    draw_arrow((px, 3.5), (px, 5.5), C_WYH, "$W_{yh}$", offset=(-0.4, 0))
    
    # Hidden to hidden (horizontal)
    if i > 0:
        prev_x = x_pos[i-1]
        draw_arrow((prev_x, 3.5), (px, 3.5), C_WHH, "$W_{hh}$", offset=(0, 0.3))

# Titles and text
plt.suptitle("RNNs Unrolled Through Time: Temporal Weight Sharing", fontsize=24, fontweight='bold', y=0.95)

desc = "Notice that $W_{hx}$, $W_{hh}$, and $W_{yh}$ are the EXACT same weight matrices reused at every single time step.\n"
desc += "The network learns one set of rules applied universally across time."
fig.text(0.5, 0.05, desc, ha='center', va='center', fontsize=16, fontweight='bold',
         bbox=dict(facecolor='#f8f9fa', edgecolor='black', pad=15, lw=2))

plt.tight_layout(rect=[0, 0.1, 1, 0.9])
plt.savefig('assets/rnn_unroll_visual.png', dpi=300, bbox_inches='tight')
print("Saved assets/rnn_unroll_visual.png")
