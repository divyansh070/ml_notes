import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# --- CONFIGURATION ---
C_EMBED = '#d6eaf8'
C_HEAD1 = '#fadbd8'
C_HEAD2 = '#d5f5e3'
C_HEAD3 = '#fcf3cf'
C_OUT = '#e8daef'

def draw_box(ax, x, y, width, height, text, bg_color):
    rect = Rectangle((x, y), width, height, facecolor=bg_color, edgecolor='black', lw=2, zorder=2)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=11, fontweight='bold', zorder=3)

def draw_arrow(ax, start, end, text=None):
    arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=15, color='gray', lw=2, zorder=1)
    ax.add_patch(arrow)
    if text:
        ax.text((start[0]+end[0])/2, (start[1]+end[1])/2 + 0.3, text, ha='center', va='center', 
                fontsize=10, fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))

# ==========================================
# MULTI-HEAD ATTENTION GRAPH
# ==========================================
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# 1. Input Q, K, V
draw_box(ax, 1, 4, 2, 2, "Input\nQ, K, V\n(Dim = 512)", C_EMBED)

# 2. Linear Projections (Splitting)
draw_arrow(ax, (3, 5), (4.5, 7.5), "Linear $W_1$")
draw_arrow(ax, (3, 5), (4.5, 5), "Linear $W_2$")
draw_arrow(ax, (3, 5), (4.5, 2.5), "Linear $W_h$")

# 3. The Heads
draw_box(ax, 4.5, 6.5, 3, 2, "Head 1\nScaled Dot-Product\n(Dim = 64)", C_HEAD1)
draw_box(ax, 4.5, 4, 3, 2, "Head 2\nScaled Dot-Product\n(Dim = 64)", C_HEAD2)
draw_box(ax, 4.5, 1.5, 3, 2, "Head h (8)\nScaled Dot-Product\n(Dim = 64)", C_HEAD3)

# 4. Concatenation
draw_arrow(ax, (7.5, 7.5), (9.5, 5))
draw_arrow(ax, (7.5, 5), (9.5, 5))
draw_arrow(ax, (7.5, 2.5), (9.5, 5))

draw_box(ax, 9.5, 3.5, 2, 3, "Concat\nAll Heads\n(Dim = 512)", C_OUT)

# 5. Final Linear Projection
draw_arrow(ax, (11.5, 5), (13, 5), "Linear $W^O$")
draw_box(ax, 13, 4, 2, 2, "Final\nOutput\n(Dim = 512)", C_EMBED)

# Explanation Box
explanation = (
    "Why Multi-Head?\n"
    "Instead of doing one massive attention calculation, we split the 512 dimensions\n"
    "into 8 separate heads of 64 dimensions each. \n"
    "Head 1 might look for Subject-Verb relationships.\n"
    "Head 2 might look for Adjective-Noun relationships.\n"
    "They process in parallel, concat back to 512, and blend via a final weight matrix."
)
ax.text(8, 0.5, explanation, ha='center', va='bottom', fontsize=12, fontweight='bold', 
        bbox=dict(facecolor='#fcf3cf', edgecolor='orange', pad=0.8))

plt.suptitle("Topic 3: Multi-Head Attention Architecture", fontsize=20, fontweight='bold', y=0.95)
plt.savefig('assets/multi_head_attention.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated assets/multi_head_attention.png successfully!")
