import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

# --- CONFIGURATION ---
C_NODE = '#f0f0f0'
C_WHX = '#1f77b4'
C_WHH = '#d62728'
C_WYH = '#2ca02c'
C_ERR = '#e377c2'

def draw_node(ax, x, y, text, radius=0.6, color=C_NODE):
    circle = Circle((x, y), radius, facecolor=color, edgecolor='black', lw=2, zorder=2)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontsize=14, fontweight='bold', zorder=3)

def draw_arrow(ax, start, end, color, label='', offset=(0, 0), rad=0.0):
    arrow = FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", 
                            color=color, arrowstyle='-|>', mutation_scale=20, lw=3,
                            shrinkA=30, shrinkB=30, zorder=1)
    ax.add_patch(arrow)
    if label:
        lx = (start[0] + end[0]) / 2 + offset[0]
        ly = (start[1] + end[1]) / 2 + offset[1]
        ax.text(lx, ly, label, color=color, fontsize=12, fontweight='bold', ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

# --- FORWARD PASS ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.5)
ax.axis('off')

# Nodes
draw_node(ax, 1, 3, "$h_0=0$")
draw_node(ax, 4, 1, "$x_1=1$")
draw_node(ax, 4, 3, "$h_1=2$")
draw_node(ax, 7, 1, "$x_2=2$")
draw_node(ax, 7, 3, "$h_2=6$")
draw_node(ax, 7, 5.5, "$\hat{y}=6$")

# Arrows
draw_arrow(ax, (1, 3), (4, 3), C_WHH, "$W_{hh}=1$", offset=(0, 0.3))
draw_arrow(ax, (4, 1), (4, 3), C_WHX, "$W_{hx}=2$", offset=(-0.5, 0))
draw_arrow(ax, (4, 3), (7, 3), C_WHH, "$W_{hh}=1$", offset=(0, 0.3))
draw_arrow(ax, (7, 1), (7, 3), C_WHX, "$W_{hx}=2$", offset=(-0.5, 0))
draw_arrow(ax, (7, 3), (7, 5.5), C_WYH, "$W_{yh}=1$", offset=(-0.5, 0))

plt.suptitle("RNN Forward Pass: Unrolled Evaluation", fontsize=20, fontweight='bold', y=0.95)
plt.savefig('assets/rnn_forward_pass.png', dpi=300, bbox_inches='tight')
plt.close()

# --- BACKWARD PASS ---
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 10)
ax.set_ylim(-0.5, 6.5)
ax.axis('off')

# Nodes
draw_node(ax, 1, 3, "$h_0=0$")
draw_node(ax, 4, 1, "$x_1=1$")
draw_node(ax, 4, 3, "$h_1=2$")
draw_node(ax, 7, 1, "$x_2=2$")
draw_node(ax, 7, 3, "$h_2=6$")
draw_node(ax, 7, 5.5, "$d\hat{y}=2$\n(Error)", color='#ffb3b3') # Reddish error node

# Gradients (as text or small boxes)
ax.text(4, 4.5, "$dh_1 = 2$", ha='center', va='center', fontsize=12, fontweight='bold', color='red', bbox=dict(facecolor='white', edgecolor='red', pad=3))
ax.text(7, 4.5, "$dh_2 = 2$", ha='center', va='center', fontsize=12, fontweight='bold', color='red', bbox=dict(facecolor='white', edgecolor='red', pad=3))

ax.text(2.5, 1.5, "$dW_{hx(t=1)}=2$\n$dW_{hh(t=1)}=0$", ha='center', va='center', fontsize=11, fontweight='bold', color='red', bbox=dict(facecolor='white', edgecolor='red', pad=3))
ax.text(8.5, 1.5, "$dW_{hx(t=2)}=4$\n$dW_{hh(t=2)}=4$", ha='center', va='center', fontsize=11, fontweight='bold', color='red', bbox=dict(facecolor='white', edgecolor='red', pad=3))


# Backward Arrows
draw_arrow(ax, (7, 5.5), (7, 4.8), 'red', "$dW_{yh}=12$", offset=(0.8, 0))
draw_arrow(ax, (7, 4.2), (7, 3), 'red', "", offset=(0, 0))
draw_arrow(ax, (7, 3), (8.5, 2.0), 'red', "", offset=(0, 0)) # To gradients t=2

draw_arrow(ax, (7, 3), (4, 3), 'red', "BPTT\n$dh_2 \cdot W_{hh}$", offset=(0, 0.4))
draw_arrow(ax, (4, 4.2), (4, 3), 'red', "", offset=(0, 0))
draw_arrow(ax, (4, 3), (2.5, 2.0), 'red', "", offset=(0, 0)) # To gradients t=1

# Summary Box
summary = "$dW_{hx} = 4\ (from\ t_2) + 2\ (from\ t_1) = \mathbf{6}$\n"
summary += "$dW_{hh} = 4\ (from\ t_2) + 0\ (from\ t_1) = \mathbf{4}$"
ax.text(5, -0.5, summary, ha='center', va='center', fontsize=14, fontweight='bold',
        bbox=dict(facecolor='#f8f9fa', edgecolor='black', pad=10, lw=2))

plt.suptitle("RNN Backward Pass (BPTT): Gradient Accumulation", fontsize=20, fontweight='bold', y=0.95)
plt.savefig('assets/rnn_backward_pass.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved assets/rnn_forward_pass.png and assets/rnn_backward_pass.png")
