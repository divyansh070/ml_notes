import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

C_NODE = '#f0f0f0'
C_WHX = '#1f77b4'
C_WHH = '#d62728'
C_WYH = '#2ca02c'

def draw_node(ax, x, y, text, radius=0.5, color=C_NODE):
    circle = Circle((x, y), radius, facecolor=color, edgecolor='black', lw=2, zorder=2)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontsize=16, fontweight='bold', zorder=3)

def draw_arrow(ax, start, end, color, label='', text_pos=None, text_bbox=None):
    arrow = FancyArrowPatch(start, end, connectionstyle="arc3,rad=0", 
                            color=color, arrowstyle='-|>', mutation_scale=25, lw=3,
                            shrinkA=25, shrinkB=25, zorder=1)
    ax.add_patch(arrow)
    if label and text_pos:
        bbox_props = text_bbox if text_bbox else dict(facecolor='white', edgecolor='none', alpha=0.8, pad=3)
        ax.text(text_pos[0], text_pos[1], label, color=color, fontsize=14, fontweight='bold', 
                ha='center', va='center', bbox=bbox_props, zorder=4)

# ==========================================
# 1. FORWARD PASS
# ==========================================
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-1, 6)
ax.set_ylim(-1, 6)
ax.axis('off')

# Nodes
draw_node(ax, 0, 2, "$h_0=0$")
draw_node(ax, 2, 0, "$x_1=1$")
draw_node(ax, 2, 2, "$h_1=2$")
draw_node(ax, 4, 0, "$x_2=2$")
draw_node(ax, 4, 2, "$h_2=6$")
draw_node(ax, 4, 4, "$\hat{y}=6$")

# Arrows & Labels
draw_arrow(ax, (0, 2), (2, 2), C_WHH, "$W_{hh}=1$", text_pos=(1, 2.4))
draw_arrow(ax, (2, 0), (2, 2), C_WHX, "$W_{hx}=2$", text_pos=(1.2, 1))
draw_arrow(ax, (2, 2), (4, 2), C_WHH, "$W_{hh}=1$", text_pos=(3, 2.4))
draw_arrow(ax, (4, 0), (4, 2), C_WHX, "$W_{hx}=2$", text_pos=(3.2, 1))
draw_arrow(ax, (4, 2), (4, 4), C_WYH, "$W_{yh}=1$", text_pos=(3.2, 3))

plt.suptitle("RNN Forward Pass: Unrolled Evaluation", fontsize=20, fontweight='bold', y=0.95)
plt.savefig('assets/rnn_forward_pass.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 2. BACKWARD PASS (BPTT)
# ==========================================
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 6)
ax.axis('off')

# Nodes
draw_node(ax, 0, 2, "$h_0=0$")
draw_node(ax, 2, 0, "$x_1=1$")
draw_node(ax, 2, 2, "$h_1=2$")
draw_node(ax, 4, 0, "$x_2=2$")
draw_node(ax, 4, 2, "$h_2=6$")
draw_node(ax, 4, 4, "$d\hat{y}=2$\n(Error)", color='#ffcccc') # Error node

red_bbox = dict(facecolor='white', alpha=1.0, edgecolor='red')

# Arrows (Flowing backwards)
# Error -> h2
draw_arrow(ax, (4, 4), (4, 2), 'red', "$dW_{yh}=12$", text_pos=(4.8, 3.0), text_bbox=red_bbox)
# h2 -> h1
draw_arrow(ax, (4, 2), (2, 2), 'red', "BPTT\n$dh_1=2$", text_pos=(3.0, 2.6), text_bbox=red_bbox)

# Text Boxes explicitly placed beside nodes without drawing downward arrows
# dh2 text
ax.text(4.8, 2.2, "$dh_2=2$", color='red', fontsize=14, fontweight='bold', ha='center', va='center', bbox=red_bbox, zorder=4)

# t=2 Gradients beside x2
ax.text(5.5, 0, "$dW_{hx(t=2)}=4$\n$dW_{hh(t=2)}=4$", color='red', fontsize=14, fontweight='bold', ha='center', va='center', bbox=red_bbox, zorder=4)

# t=1 Gradients beside x1
ax.text(0.5, 0, "$dW_{hx(t=1)}=2$\n$dW_{hh(t=1)}=0$", color='red', fontsize=14, fontweight='bold', ha='center', va='center', bbox=red_bbox, zorder=4)

# Summary Box at the bottom
summary = "$dW_{hx} = 4\ (from\ t_2) + 2\ (from\ t_1) = \mathbf{6}$\n"
summary += "$dW_{hh} = 4\ (from\ t_2) + 0\ (from\ t_1) = \mathbf{4}$"
ax.text(3, -0.8, summary, ha='center', va='center', fontsize=16, fontweight='bold',
        bbox=dict(facecolor='#f8f9fa', edgecolor='black', pad=10, lw=2))

plt.suptitle("RNN Backward Pass (BPTT): Gradient Accumulation", fontsize=20, fontweight='bold', y=0.95)
plt.savefig('assets/rnn_backward_pass.png', dpi=300, bbox_inches='tight')
plt.close()

print("Saved assets/rnn_forward_pass.png and assets/rnn_backward_pass.png")
