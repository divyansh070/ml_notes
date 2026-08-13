import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

# --- CONFIGURATION & HELPERS ---
C_BOX = '#f8f9fa'
C_FORGET = '#e74c3c'  
C_INPUT = '#2ecc71'   
C_OUTPUT = '#3498db'  
C_MATH = '#f1c40f'    
C_LINE = '#333333'

def draw_node(ax, x, y, text, radius=0.4, color=C_MATH, text_color='black', fontsize=16):
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
    ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold', color='black',
            bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.4', alpha=0.95), zorder=5)

# ==========================================
# 1. FORWARD PASS TEMPLATE
# ==========================================
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 15)
ax.set_ylim(0, 9)
ax.axis('off')

# Bounding Box
rect = Rectangle((1.5, 1.5), 12, 6, fill=True, facecolor=C_BOX, edgecolor='black', lw=3, linestyle='--', zorder=0)
ax.add_patch(rect)
ax.text(2, 7.2, "LSTM Cell", fontsize=20, fontweight='bold', color='gray')

# Tracks
draw_line(ax, (0, 6.5), (15, 6.5), lw=4, color='gray')
draw_arrow(ax, (14, 6.5), (15, 6.5), lw=4, color='gray')
draw_line(ax, (0, 2.5), (15, 2.5), lw=3, color='gray')
draw_arrow(ax, (14, 2.5), (15, 2.5), lw=3, color='gray')

# Base Inputs/Outputs
ax.text(0.5, 6.8, r"$C_0 = 0$", fontsize=16, fontweight='bold')
ax.text(14.5, 6.8, r"$C_1 = 0.84$", fontsize=16, fontweight='bold')
ax.text(0.5, 2.8, r"$h_0 = 0$", fontsize=16, fontweight='bold')
ax.text(14.5, 2.8, r"$h_1 = 0.61$", fontsize=16, fontweight='bold')
ax.text(2.5, 0.5, r"$x_1 = 2$", fontsize=16, fontweight='bold')

# Input Merge
draw_arrow(ax, (2.5, 0.8), (2.5, 2.5), color='black')
draw_node(ax, 2.5, 2.5, "Concat", radius=0.4, color='white', fontsize=10)
draw_line(ax, (2.5, 2.5), (10, 2.5), color='gray')

# --- FORGET GATE ---
x_f = 4.5
draw_arrow(ax, (x_f, 2.5), (x_f, 4.1), color=C_FORGET, lw=3)
draw_node(ax, x_f, 4.5, r"$\sigma$", color=C_FORGET, text_color='white')
draw_arrow(ax, (x_f, 4.9), (x_f, 6.1), color=C_FORGET, lw=3)
draw_node(ax, x_f, 6.5, r"$\otimes$", color=C_MATH)

draw_math_box(ax, x_f - 1.4, 4.0, r"$f_1 = \sigma(W_f \cdot x_1)$" + "\n" + r"$f_1 = 0.88$", C_FORGET)
draw_math_box(ax, x_f, 7.3, r"$(f_1 \cdot C_0)$" + "\n" + r"$0.88 \cdot 0 = \mathbf{0}$", C_FORGET)

# --- INPUT GATE ---
x_i = 7.5
draw_line(ax, (x_i, 2.5), (x_i, 3.0), color=C_INPUT, lw=3)
draw_line(ax, (x_i, 3.0), (x_i-1, 3.0), color=C_INPUT, lw=3)
draw_line(ax, (x_i, 3.0), (x_i+1, 3.0), color=C_INPUT, lw=3)

draw_arrow(ax, (x_i-1, 3.0), (x_i-1, 4.1), color=C_INPUT, lw=3)
draw_node(ax, x_i-1, 4.5, r"$\sigma$", color=C_INPUT, text_color='white')
draw_math_box(ax, x_i-2.2, 4.0, r"$i_1 = \sigma(W_i \cdot x_1)$" + "\n" + r"$i_1 = 0.88$", C_INPUT)

draw_arrow(ax, (x_i+1, 3.0), (x_i+1, 4.1), color=C_INPUT, lw=3)
draw_node(ax, x_i+1, 4.5, r"$\mathrm{tanh}$", color=C_INPUT, text_color='white', fontsize=12)
draw_math_box(ax, x_i+2.4, 4.0, r"$\widetilde{C}_1 = \mathrm{tanh}(W_c \cdot x_1)$" + "\n" + r"$\widetilde{C}_1 = 0.96$", C_INPUT)

draw_arrow(ax, (x_i-1, 4.9), (x_i-0.3, 5.3), color=C_INPUT, lw=3)
draw_arrow(ax, (x_i+1, 4.9), (x_i+0.3, 5.3), color=C_INPUT, lw=3)
draw_node(ax, x_i, 5.5, r"$\otimes$", color=C_MATH)
draw_math_box(ax, x_i+1.5, 5.5, r"$(i_1 \cdot \widetilde{C}_1)$" + "\n" + r"$0.88 \cdot 0.96 = 0.84$", C_INPUT)

draw_arrow(ax, (x_i, 5.9), (x_i, 6.1), color=C_INPUT, lw=3)
draw_node(ax, x_i, 6.5, r"$\oplus$", color=C_MATH)
draw_math_box(ax, x_i, 7.3, r"$C_1 = 0 + 0.84 = \mathbf{0.84}$", C_INPUT)

# --- OUTPUT GATE ---
x_o = 11
x_h = 13
draw_arrow(ax, (x_o, 2.5), (x_o, 4.1), color=C_OUTPUT, lw=3)
draw_node(ax, x_o, 4.5, r"$\sigma$", color=C_OUTPUT, text_color='white')
draw_math_box(ax, x_o - 1.2, 4.0, r"$o_1 = \sigma(W_o \cdot x_1)$" + "\n" + r"$o_1 = 0.88$", C_OUTPUT)

draw_arrow(ax, (x_h, 6.5), (x_h, 4.9), color=C_OUTPUT, lw=3)
draw_node(ax, x_h, 4.5, r"$\mathrm{tanh}$", color=C_OUTPUT, text_color='white', fontsize=12)
draw_math_box(ax, x_h + 1.2, 4.5, r"$\mathrm{tanh}(C_1)$" + "\n" + r"$\mathrm{tanh}(0.84) = 0.69$", C_OUTPUT)

draw_arrow(ax, (x_o, 4.9), (x_h-0.3, 2.7), color=C_OUTPUT, lw=3, rad=-0.3)
draw_arrow(ax, (x_h, 4.1), (x_h, 2.9), color=C_OUTPUT, lw=3)
draw_node(ax, x_h, 2.5, r"$\otimes$", color=C_MATH)
draw_math_box(ax, x_h, 1.7, r"$h_1 = o_1 \cdot \mathrm{tanh}(C_1)$" + "\n" + r"$0.88 \cdot 0.69 = \mathbf{0.61}$", C_OUTPUT)

plt.suptitle("LSTM Forward Pass: Explicit Math on the Standard Architecture", fontsize=22, fontweight='bold', y=0.98)
plt.savefig('assets/lstm_template_forward.png', dpi=300, bbox_inches='tight')
plt.close()


# ==========================================
# 2. BACKWARD PASS TEMPLATE
# ==========================================
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 15)
ax.set_ylim(0, 9)
ax.axis('off')

# Bounding Box & Structural Lines
rect = Rectangle((1.5, 1.5), 12, 6, fill=True, facecolor=C_BOX, edgecolor='gray', lw=2, linestyle='--', zorder=0)
ax.add_patch(rect)
draw_line(ax, (0, 6.5), (15, 6.5), lw=2, color='lightgray')
draw_line(ax, (0, 2.5), (15, 2.5), lw=2, color='lightgray')

# Nodes
draw_node(ax, x_f, 4.5, r"$\sigma$", color='#fadbd8', text_color='black')
draw_node(ax, x_f, 6.5, r"$\otimes$", color=C_MATH)
draw_node(ax, x_i-1, 4.5, r"$\sigma$", color='#d5f5e3', text_color='black')
draw_node(ax, x_i+1, 4.5, r"$\mathrm{tanh}$", color='#d5f5e3', text_color='black', fontsize=12)
draw_node(ax, x_i, 5.5, r"$\otimes$", color=C_MATH)
draw_node(ax, x_i, 6.5, r"$\oplus$", color=C_MATH)
draw_node(ax, x_o, 4.5, r"$\sigma$", color='#d6eaf8', text_color='black')
draw_node(ax, x_h, 4.5, r"$\mathrm{tanh}$", color='#d6eaf8', text_color='black', fontsize=12)
draw_node(ax, x_h, 2.5, r"$\otimes$", color=C_MATH)
ax.text(2.5, 0.5, r"$x_1 = 2$", fontsize=16, fontweight='bold', color='gray')

# --- BPTT Gradients (RED) ---
# Error Enters
draw_arrow(ax, (14.5, 2.5), (13.5, 2.5), color='red', lw=3)
draw_math_box(ax, 14, 3.1, r"Total Error" + "\n" + r"$dh_1 = 2$", 'red')

# Split at Output Multiply (Product Rule)
draw_arrow(ax, (x_h, 2.9), (x_h, 4.1), color='red', lw=3)
draw_math_box(ax, x_h + 1.2, 3.4, r"$do_1 = dh_1 \cdot \mathrm{tanh}(C_1) = \mathbf{1.38}$" + "\n" + r"$dW_o = do_1 \cdot \sigma' \cdot x_1 \approx \mathbf{0.30}$", 'red')
draw_arrow(ax, (x_h-0.3, 2.7), (x_o, 4.9), color='red', lw=3, rad=0.3)
draw_arrow(ax, (x_h, 4.9), (x_h, 6.3), color='red', lw=3)
draw_math_box(ax, x_h + 1.2, 5.5, r"Chain Rule:" + "\n" + r"$dh_1 \cdot o_1 = 1.76$" + "\n" + r"$dC_1 = 1.76 \cdot \mathrm{tanh}' = \mathbf{0.92}$", 'red')

# Cell State Gradient Superhighway
draw_arrow(ax, (12.5, 6.5), (8, 6.5), color='red', lw=4)
draw_math_box(ax, 10.2, 7.1, r"$dC_1 = 0.92$", 'red')
draw_arrow(ax, (7, 6.5), (5, 6.5), color='red', lw=4)

# Flow down to Input Gate / Candidate
draw_arrow(ax, (x_i, 6.1), (x_i, 5.9), color='red', lw=3)
draw_math_box(ax, x_i + 1.1, 6.1, r"Flows across $\oplus$", 'red')

# Split at Input Multiply
draw_arrow(ax, (x_i-0.2, 5.3), (x_i-1, 4.9), color='red', lw=3)
draw_math_box(ax, x_i-2.2, 5.3, r"$di_1 = dC_1 \cdot \widetilde{C}_1 = \mathbf{0.88}$" + "\n" + r"$dW_i = di_1 \cdot \sigma' \cdot x_1 \approx \mathbf{0.19}$", 'red')

draw_arrow(ax, (x_i+0.2, 5.3), (x_i+1, 4.9), color='red', lw=3)
draw_math_box(ax, x_i+2.2, 5.3, r"$d\widetilde{C}_1 = dC_1 \cdot i_1 = \mathbf{0.81}$" + "\n" + r"$dW_c = d\widetilde{C}_1 \cdot \mathrm{tanh}' \cdot x_1 \approx \mathbf{0.13}$", 'red')

# Flow to Forget Gate
draw_arrow(ax, (x_f, 6.1), (x_f, 4.9), color='red', lw=3)
draw_math_box(ax, x_f - 1.5, 5.4, r"$df_1 = dC_1 \cdot C_0 = \mathbf{0}$" + "\n" + r"$dW_f = df_1 \cdot \sigma' \cdot x_1 = \mathbf{0}$", 'red')

plt.suptitle("LSTM Backward Pass: Exposing the Product & Chain Rule", fontsize=22, fontweight='bold', y=0.98)
plt.savefig('assets/lstm_template_backward.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated Updated lstm_template_forward.png and lstm_template_backward.png")