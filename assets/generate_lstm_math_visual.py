import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

# --- CONFIGURATION & HELPERS ---
C_BOX = '#f8f9fa'
C_FORGET = '#e74c3c'  
C_INPUT = '#2ecc71'   
C_OUTPUT = '#3498db'  
C_MATH = '#f1c40f'    
C_LINE = '#333333'

def draw_node(ax, x, y, text, radius=0.4, color=C_MATH, text_color='black', fontsize=18):
    circle = Circle((x, y), radius, facecolor=color, edgecolor='black', lw=2, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=text_color, zorder=4)

def draw_arrow(ax, start, end, color=C_LINE, lw=2, rad=0.0):
    arrow = FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", 
                            color=color, arrowstyle='-|>', mutation_scale=20, lw=lw, zorder=2)
    ax.add_patch(arrow)

def draw_line(ax, start, end, color=C_LINE, lw=2):
    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=lw, zorder=1)

def draw_math_box(ax, x, y, text, color='gray'):
    ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold', color='black',
            bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.3', alpha=0.95), zorder=5)

# ==========================================
# 1. FORWARD PASS TEMPLATE
# ==========================================
fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Bounding Box
rect = Rectangle((1.5, 1), 11, 6, fill=True, facecolor=C_BOX, edgecolor='black', lw=3, linestyle='--', zorder=0)
ax.add_patch(rect)
ax.text(2, 6.7, "LSTM Cell", fontsize=20, fontweight='bold', color='gray')

# Tracks
draw_line(ax, (0, 6), (14, 6), lw=4, color='gray')
draw_arrow(ax, (13, 6), (14, 6), lw=4, color='gray')
draw_line(ax, (0, 2), (14, 2), lw=3, color='gray')
draw_arrow(ax, (13, 2), (14, 2), lw=3, color='gray')

# Base Inputs/Outputs
ax.text(0.5, 6.3, r"$C_0 = 0$", fontsize=16, fontweight='bold')
ax.text(13.5, 6.3, r"$C_1 = 0.84$", fontsize=16, fontweight='bold')
ax.text(0.5, 2.3, r"$h_0 = 0$", fontsize=16, fontweight='bold')
ax.text(13.5, 2.3, r"$h_1 = 0.61$", fontsize=16, fontweight='bold')
ax.text(2.5, 0.2, r"$x_1 = 2$", fontsize=16, fontweight='bold')

# Input Merge
draw_arrow(ax, (2.5, 0.5), (2.5, 2), color='black')
draw_node(ax, 2.5, 2, "Concat", radius=0.4, color='white', fontsize=12)
draw_line(ax, (2.5, 2), (9.5, 2), color='gray')

# --- FORGET GATE ---
x_f = 4.5
draw_arrow(ax, (x_f, 2), (x_f, 3.6), color=C_FORGET, lw=3)
draw_node(ax, x_f, 4, r"$\sigma$", color=C_FORGET, text_color='white')
draw_arrow(ax, (x_f, 4.4), (x_f, 5.6), color=C_FORGET, lw=3)
draw_node(ax, x_f, 6, r"$\otimes$", color=C_MATH)
draw_math_box(ax, x_f - 1.2, 4, r"$f_1 = \sigma(2) = 0.88$", C_FORGET)
draw_math_box(ax, x_f, 6.7, r"$0 \cdot 0.88 = \mathbf{0}$", C_FORGET)

# --- INPUT GATE ---
x_i = 7
draw_line(ax, (x_i, 2), (x_i, 2.5), color=C_INPUT, lw=3)
draw_line(ax, (x_i, 2.5), (x_i-0.8, 2.5), color=C_INPUT, lw=3)
draw_line(ax, (x_i, 2.5), (x_i+0.8, 2.5), color=C_INPUT, lw=3)

draw_arrow(ax, (x_i-0.8, 2.5), (x_i-0.8, 3.6), color=C_INPUT, lw=3)
draw_node(ax, x_i-0.8, 4, r"$\sigma$", color=C_INPUT, text_color='white')
draw_math_box(ax, x_i-1.8, 3.7, r"$i_1 = 0.88$", C_INPUT)

draw_arrow(ax, (x_i+0.8, 2.5), (x_i+0.8, 3.6), color=C_INPUT, lw=3)
draw_node(ax, x_i+0.8, 4, r"$\mathrm{tanh}$", color=C_INPUT, text_color='white', fontsize=12)
draw_math_box(ax, x_i+1.9, 3.7, r"$\widetilde{C}_1 = 0.96$", C_INPUT)

draw_arrow(ax, (x_i-0.8, 4.4), (x_i-0.2, 4.8), color=C_INPUT, lw=3)
draw_arrow(ax, (x_i+0.8, 4.4), (x_i+0.2, 4.8), color=C_INPUT, lw=3)
draw_node(ax, x_i, 5, r"$\otimes$", color=C_MATH)
draw_math_box(ax, x_i+1.4, 4.8, r"$0.88 \cdot 0.96 = 0.84$", C_INPUT)

draw_arrow(ax, (x_i, 5.4), (x_i, 5.6), color=C_INPUT, lw=3)
draw_node(ax, x_i, 6, r"$\oplus$", color=C_MATH)
draw_math_box(ax, x_i, 6.7, r"$0 + 0.84 = \mathbf{0.84}$", C_INPUT)

# --- OUTPUT GATE ---
x_o = 9.5
x_h = 11.5
draw_arrow(ax, (x_o, 2), (x_o, 3.6), color=C_OUTPUT, lw=3)
draw_node(ax, x_o, 4, r"$\sigma$", color=C_OUTPUT, text_color='white')
draw_math_box(ax, x_o - 1.0, 4, r"$o_1 = 0.88$", C_OUTPUT)

draw_arrow(ax, (x_h, 6), (x_h, 4.4), color=C_OUTPUT, lw=3)
draw_node(ax, x_h, 4, r"$\mathrm{tanh}$", color=C_OUTPUT, text_color='white', fontsize=12)
draw_math_box(ax, x_h + 1.2, 4, r"$\mathrm{tanh}(0.84) = 0.69$", C_OUTPUT)

draw_arrow(ax, (x_o, 4.4), (x_h-0.3, 2.2), color=C_OUTPUT, lw=3, rad=-0.3)
draw_arrow(ax, (x_h, 3.6), (x_h, 2.4), color=C_OUTPUT, lw=3)
draw_node(ax, x_h, 2, r"$\otimes$", color=C_MATH)
draw_math_box(ax, x_h, 1.3, r"$0.88 \cdot 0.69 = \mathbf{0.61}$", C_OUTPUT)

plt.suptitle("LSTM Forward Pass: Explicit Math on the Standard Architecture", fontsize=22, fontweight='bold', y=0.98)
plt.savefig('assets/lstm_template_forward.png', dpi=300, bbox_inches='tight')
plt.close()


# ==========================================
# 2. BACKWARD PASS TEMPLATE
# ==========================================
fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Bounding Box & Structural Lines (Faded)
rect = Rectangle((1.5, 1), 11, 6, fill=True, facecolor=C_BOX, edgecolor='gray', lw=2, linestyle='--', zorder=0)
ax.add_patch(rect)
draw_line(ax, (0, 6), (14, 6), lw=2, color='lightgray')
draw_line(ax, (0, 2), (14, 2), lw=2, color='lightgray')

# Nodes
draw_node(ax, x_f, 4, r"$\sigma$", color='#fadbd8', text_color='black')
draw_node(ax, x_f, 6, r"$\otimes$", color=C_MATH)
draw_node(ax, x_i-0.8, 4, r"$\sigma$", color='#d5f5e3', text_color='black')
draw_node(ax, x_i+0.8, 4, r"$\mathrm{tanh}$", color='#d5f5e3', text_color='black', fontsize=12)
draw_node(ax, x_i, 5, r"$\otimes$", color=C_MATH)
draw_node(ax, x_i, 6, r"$\oplus$", color=C_MATH)
draw_node(ax, x_o, 4, r"$\sigma$", color='#d6eaf8', text_color='black')
draw_node(ax, x_h, 4, r"$\mathrm{tanh}$", color='#d6eaf8', text_color='black', fontsize=12)
draw_node(ax, x_h, 2, r"$\otimes$", color=C_MATH)

# Base Text
ax.text(2.5, 0.2, r"$x_1 = 2$", fontsize=16, fontweight='bold', color='gray')

# --- BPTT GRAdients (RED) ---
# Error Enters
draw_arrow(ax, (13.5, 2), (12, 2), color='red', lw=3)
draw_math_box(ax, 12.8, 2.5, r"Error: $dh_1 = 2$", 'red')

# Split at Output Multiply (Product Rule)
draw_arrow(ax, (x_h, 2.5), (x_h, 3.5), color='red', lw=3)
draw_math_box(ax, x_h + 1.2, 2.8, r"$2 \cdot \mathrm{tanh}(C_1) = \mathbf{1.38}$" + "\n" + r"$\cdot \sigma' \rightarrow 1.38 \cdot 0.11 = 0.15$", 'red')
draw_arrow(ax, (x_h-0.3, 2.2), (x_o, 4.4), color='red', lw=3, rad=0.3)
draw_arrow(ax, (x_h, 4.5), (x_h, 5.8), color='red', lw=3)
draw_math_box(ax, x_h + 1.2, 4.8, r"Chain: $2 \cdot 0.88 = 1.76$" + "\n" + r"$\cdot \mathrm{tanh}' \rightarrow 1.76 \cdot 0.52 = \mathbf{0.92}$", 'red')

# Cell State Gradient Superhighway
draw_arrow(ax, (11, 6), (7.5, 6), color='red', lw=4)
draw_math_box(ax, 9.2, 6.5, r"$dC_1 = 0.92$", 'red')
draw_arrow(ax, (6.5, 6), (5, 6), color='red', lw=4)

# Flow down to Input Gate / Candidate
draw_arrow(ax, (x_i, 5.5), (x_i, 5.1), color='red', lw=3)
draw_math_box(ax, x_i + 1.1, 5.5, r"Flows across $\oplus$", 'red')

# Split at Input Multiply
draw_arrow(ax, (x_i-0.2, 4.8), (x_i-0.8, 4.4), color='red', lw=3)
draw_math_box(ax, x_i-1.8, 4.8, r"$0.92 \cdot 0.96 = \mathbf{0.88}$" + "\n" + r"$\cdot \sigma' \rightarrow 0.88 \cdot 0.11 = 0.10$", 'red')

draw_arrow(ax, (x_i+0.2, 4.8), (x_i+0.8, 4.4), color='red', lw=3)
draw_math_box(ax, x_i+1.8, 4.3, r"$0.92 \cdot 0.88 = \mathbf{0.81}$" + "\n" + r"$\cdot \mathrm{tanh}' \rightarrow 0.81 \cdot 0.08 = 0.06$", 'red')

# Flow to Forget Gate
draw_arrow(ax, (x_f, 5.5), (x_f, 4.4), color='red', lw=3)
draw_math_box(ax, x_f - 1.4, 5, r"$0.92 \cdot C_0 = \mathbf{0}$" + "\n" + r"$\cdot \sigma' \rightarrow 0$", 'red')

plt.suptitle("LSTM Backward Pass: Exposing the Product & Chain Rule", fontsize=22, fontweight='bold', y=0.98)
plt.savefig('assets/lstm_template_backward.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated lstm_template_forward.png and lstm_template_backward.png")