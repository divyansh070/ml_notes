import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# --- CONFIGURATION ---
C_BG = '#f8f9fa'
C_ATTN = '#fcf3cf'   # Yellow
C_FFN = '#d6eaf8'    # Blue
C_NORM = '#d5f5e3'   # Green
C_TEXT = 'black'

def draw_box(ax, x, y, width, height, text, bg_color):
    rect = Rectangle((x, y), width, height, facecolor=bg_color, edgecolor='black', lw=2, zorder=2)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=12, fontweight='bold', zorder=3)

def draw_arrow(ax, start, end, text=None, rad=0.0):
    arrow = FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", 
                            arrowstyle='-|>', mutation_scale=15, color='gray', lw=2, zorder=1)
    ax.add_patch(arrow)
    if text:
        ax.text((start[0]+end[0])/2, (start[1]+end[1])/2, text, ha='center', va='center', 
                fontsize=10, fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))

# ==========================================
# TRANSFORMER ENCODER BLOCK
# ==========================================
fig, ax = plt.subplots(figsize=(10, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Outer Block Boundary
rect = Rectangle((1, 1.5), 8, 11.5, fill=True, facecolor=C_BG, edgecolor='black', lw=3, linestyle='--', zorder=0)
ax.add_patch(rect)
ax.text(1.5, 12.5, "One Encoder Block\n(e.g., 1 of 12 in BERT)", fontsize=14, fontweight='bold', color='gray')

# 1. Input
ax.text(5, 0.5, "Input Matrix ($X$)\n[Words x 512]", ha='center', fontsize=14, fontweight='bold')
draw_arrow(ax, (5, 1.0), (5, 2.0))

# 2. Multi-Head Attention
draw_box(ax, 3, 3.5, 4, 1.5, "Multi-Head\nSelf-Attention", C_ATTN)
# Q, K, V split arrows
draw_arrow(ax, (5, 2.0), (5, 3.5))
draw_arrow(ax, (5, 2.5), (3.5, 3.5), rad=0.2)
draw_arrow(ax, (5, 2.5), (6.5, 3.5), rad=-0.2)
ax.text(3.4, 3.0, "Q", fontweight='bold')
ax.text(5.2, 3.0, "K", fontweight='bold')
ax.text(6.6, 3.0, "V", fontweight='bold')

# 3. Add & Norm 1
draw_box(ax, 3, 6, 4, 1.5, "Add & Norm\nLayerNorm(X + Sublayer(X))", C_NORM)
draw_arrow(ax, (5, 5), (5, 6)) # Main path

# Residual Connection 1
draw_arrow(ax, (5, 2.0), (2, 2.0))
draw_arrow(ax, (2, 2.0), (2, 6.75))
draw_arrow(ax, (2, 6.75), (3, 6.75))
ax.text(1.8, 4.5, "Residual\nConnection ($+ X$)", ha='right', va='center', fontsize=10, fontweight='bold', color='red')

# 4. Feed Forward Network
draw_box(ax, 3, 8.5, 4, 1.5, "Feed-Forward Network\n(Expands to 2048, then 512)", C_FFN)
draw_arrow(ax, (5, 7.5), (5, 8.5))

# 5. Add & Norm 2
draw_box(ax, 3, 11, 4, 1.5, "Add & Norm\nLayerNorm(X + Sublayer(X))", C_NORM)
draw_arrow(ax, (5, 10), (5, 11))

# Residual Connection 2
draw_arrow(ax, (5, 7.8), (2, 7.8))
draw_arrow(ax, (2, 7.8), (2, 11.75))
draw_arrow(ax, (2, 11.75), (3, 11.75))
ax.text(1.8, 9.8, "Residual\nConnection ($+ X$)", ha='right', va='center', fontsize=10, fontweight='bold', color='red')

# Output
draw_arrow(ax, (5, 12.5), (5, 13.5))
ax.text(5, 13.8, "Encoder Output Matrix\n[Words x 512]", ha='center', fontsize=14, fontweight='bold')

plt.suptitle("Topic 5: The Transformer Encoder Block", fontsize=20, fontweight='bold', y=0.95)
plt.savefig('assets/encoder_block_architecture.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated assets/encoder_block_architecture.png successfully!")
