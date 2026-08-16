import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

# --- STATQUEST COLORS ---
C_LONG_TERM = '#2ca02c'  # Green line (Cell State)
C_SHORT_TERM = '#d62728' # Pink/Red line (Hidden State)
C_SIGMOID = '#3498db'    # Blue block
C_TANH = '#f39c12'       # Orange block

def draw_box(ax, x, y, width, height, text, edge_color, face_color='white', text_color='black', fontsize=10):
    rect = Rectangle((x, y), width, height, facecolor=face_color, edgecolor=edge_color, lw=2, zorder=2)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=text_color, zorder=3)

def draw_arrow(ax, start, end, color='black', lw=2):
    arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=15, color=color, lw=lw, zorder=1)
    ax.add_patch(arrow)

# ==========================================
# STATQUEST LSTM FORWARD PASS
# ==========================================
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis('off')

# --- 1. THE MEMORY LINES ---
# Long-Term Memory (Top Green Line)
ax.plot([1, 17], [8, 8], color=C_LONG_TERM, lw=4, zorder=0)
draw_box(ax, 0.5, 7.5, 1.5, 1, "Long-Term\nMemory\n(2.0)", C_LONG_TERM, text_color=C_LONG_TERM)
draw_box(ax, 16.5, 7.5, 1.5, 1, "New\nLong-Term\n(2.96)", C_LONG_TERM, text_color=C_LONG_TERM)

# Short-Term Memory (Bottom Pink Line)
ax.plot([1, 17], [2, 2], color=C_SHORT_TERM, lw=3, zorder=0)
draw_box(ax, 0.5, 1.5, 1.5, 1, "Short-Term\nMemory\n(1.0)", C_SHORT_TERM, text_color=C_SHORT_TERM)
draw_box(ax, 16.5, 1.5, 1.5, 1, "New\nShort-Term\n(0.98)", C_SHORT_TERM, text_color=C_SHORT_TERM)

# Input
draw_box(ax, 2, 0.2, 1.5, 0.8, "Input\n(1.0)", 'black')
ax.plot([2.75, 15], [1.0, 1.0], color='gray', lw=2, ls='--', zorder=0) # Input bus
draw_arrow(ax, (2.75, 1.0), (2.75, 1.5), color='gray')

# --- 2. STAGE 1: FORGET GATE (What % to remember) ---
draw_box(ax, 3.5, 4.5, 2.5, 2, "Sigmoid\nFunction\n(0.997)", C_SIGMOID)
# Math box
ax.text(4.75, 3.8, "(1 x 2.70) + (1 x 1.63)\n+ 1.62 = 5.95", ha='center', fontsize=9, fontweight='bold', bbox=dict(facecolor='#ebdef0', edgecolor='gray'))
draw_arrow(ax, (4.75, 4.0), (4.75, 4.5))
draw_arrow(ax, (4.75, 2), (4.75, 3.5), color=C_SHORT_TERM) # From short-term
draw_arrow(ax, (4.25, 1), (4.25, 3.5), color='gray') # From input

# Multiply Long-Term
draw_box(ax, 4.5, 7.75, 0.5, 0.5, "X", 'black', face_color='#fcf3cf')
draw_arrow(ax, (4.75, 6.5), (4.75, 7.75))
ax.text(4.75, 7.2, "0.997", ha='center', fontweight='bold')
ax.text(4.75, 8.5, "2.0 * 0.997 = 1.99", ha='center', fontweight='bold', color=C_LONG_TERM)


# --- 3. STAGE 2: INPUT GATE (Create Potential Memory & Add it) ---
# Potential Memory (Tanh)
draw_box(ax, 9.5, 4.5, 2.5, 2, "Tanh\nFunction\n(0.97)", C_TANH)
ax.text(10.75, 3.8, "(1 x 1.41) + (1 x 0.94)\n- 0.32 = 2.03", ha='center', fontsize=9, fontweight='bold', bbox=dict(facecolor='#ebdef0', edgecolor='gray'))
draw_arrow(ax, (10.75, 4.0), (10.75, 4.5))
draw_arrow(ax, (10.75, 2), (10.75, 3.5), color=C_SHORT_TERM)
draw_arrow(ax, (10.25, 1), (10.25, 3.5), color='gray')

# Percentage to add (Sigmoid)
draw_box(ax, 6.5, 4.5, 2.5, 2, "Sigmoid\nFunction\n(1.0)", C_SIGMOID)
ax.text(7.75, 3.8, "(1 x 2.00) + (1 x 1.65)\n+ 0.62 = 4.27", ha='center', fontsize=9, fontweight='bold', bbox=dict(facecolor='#ebdef0', edgecolor='gray'))
draw_arrow(ax, (7.75, 4.0), (7.75, 4.5))
draw_arrow(ax, (7.75, 2), (7.75, 3.5), color=C_SHORT_TERM)
draw_arrow(ax, (7.25, 1), (7.25, 3.5), color='gray')

# Multiply them together
draw_box(ax, 9.25, 6.75, 0.5, 0.5, "X", 'black', face_color='#fcf3cf')
draw_arrow(ax, (10.75, 6.5), (9.75, 6.75)) # Tanh output up
draw_arrow(ax, (7.75, 6.5), (9.25, 6.75)) # Sigmoid output up
ax.text(9.5, 7.4, "1.0 * 0.97 = 0.97", ha='center', fontweight='bold')

# Add to Long-Term
draw_box(ax, 9.25, 7.75, 0.5, 0.5, "+", 'black', face_color='#d5f5e3')
draw_arrow(ax, (9.5, 7.25), (9.5, 7.75))
ax.text(9.5, 8.5, "1.99 + 0.97 = 2.96", ha='center', fontweight='bold', color=C_LONG_TERM)


# --- 4. STAGE 3: OUTPUT GATE (Update Short-Term Memory) ---
# Tanh of new Long-Term
draw_box(ax, 12, 6, 2, 1.5, "Tanh\n(0.99)", C_TANH)
draw_arrow(ax, (13, 8), (13, 7.5))

# Percentage to pass on (Sigmoid)
draw_box(ax, 13.5, 3.5, 2.5, 2, "Sigmoid\nFunction\n(0.99)", C_SIGMOID)
ax.text(14.75, 2.8, "(1 x 4.38) + (1 x -0.19)\n+ 0.59 = 4.78", ha='center', fontsize=9, fontweight='bold', bbox=dict(facecolor='#ebdef0', edgecolor='gray'))
draw_arrow(ax, (14.75, 3.0), (14.75, 3.5))
draw_arrow(ax, (14.75, 2), (14.75, 2.5), color=C_SHORT_TERM)
draw_arrow(ax, (14.25, 1), (14.25, 2.5), color='gray')

# Multiply to create new Short Term
draw_box(ax, 14.5, 6, 0.5, 0.5, "X", 'black', face_color='#fcf3cf')
draw_arrow(ax, (14, 6.75), (14.5, 6.25)) # Tanh over
draw_arrow(ax, (14.75, 5.5), (14.75, 6)) # Sigmoid up
ax.text(15.5, 6.8, "0.99 * 0.99 = 0.98", ha='center', fontweight='bold', color=C_SHORT_TERM)
draw_arrow(ax, (14.75, 6.5), (14.75, 2)) # Drops down to short term line

# STATQUEST BAM
ax.text(16, 9, "BAM!!!", fontsize=28, fontweight='bold', color='orange')

plt.suptitle("StatQuest Style: End-to-End LSTM Forward Pass", fontsize=20, fontweight='bold', y=0.95)
plt.savefig('assets/statquest_lstm_forward.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated assets/statquest_lstm_forward.png successfully!")
