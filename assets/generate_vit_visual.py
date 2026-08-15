import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import numpy as np

# --- CONFIGURATION & HELPERS ---
C_BG = '#ffffff'
C_IMAGE = '#fcf3cf'
C_PATCH = '#d5f5e3'
C_TOKEN = '#d6eaf8'
C_CLS = '#e8daef'
C_ENC = '#fef9e7'
C_CNN = '#fadbd8'

def draw_arrow(ax, start, end, color='#555', lw=2, text=None, text_offset=(0, 0.2), ls='-'):
    arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=15, color=color, lw=lw, linestyle=ls, zorder=1)
    ax.add_patch(arrow)
    if text:
        mx, my = (start[0]+end[0])/2 + text_offset[0], (start[1]+end[1])/2 + text_offset[1]
        ax.text(mx, my, text, ha='center', va='center', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1), zorder=4)

def draw_box(ax, x, y, w, h, text, bg, fontsize=10, text_color='black', lw=1.5):
    rect = Rectangle((x, y), w, h, facecolor=bg, edgecolor='black', lw=lw, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=text_color, zorder=3)

fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis('off')
fig.patch.set_facecolor(C_BG)

# ==========================================
# TOP SECTION: Vision Transformer (Global)
# ==========================================
ax.text(10, 11.5, "Vision Transformer (ViT) — Global Receptive Field", ha='center', fontsize=16, fontweight='bold')

# 1. Image
draw_box(ax, 0.5, 8.5, 2, 2, "Input\nImage\n(224x224)", C_IMAGE)

# Grid on image
for i in range(1, 3):
    ax.plot([0.5, 2.5], [8.5 + i*(2/3), 8.5 + i*(2/3)], color='gray', lw=1)
    ax.plot([0.5 + i*(2/3), 0.5 + i*(2/3)], [8.5, 10.5], color='gray', lw=1)

draw_arrow(ax, (2.6, 9.5), (3.4, 9.5), text="Extract\nPatches")

# 2. Patches & Linear Projection
for i in range(4):
    draw_box(ax, 3.5 + i*1.2, 9.2, 0.8, 0.8, f"P{i+1}", C_PATCH)
    draw_arrow(ax, (3.9 + i*1.2, 9.1), (3.9 + i*1.2, 8.5))

draw_box(ax, 3.5, 7.8, 4.4, 0.6, "Linear Projection Matrix (E)", '#ecf0f1')

for i in range(4):
    draw_arrow(ax, (3.9 + i*1.2, 7.7), (3.9 + i*1.2, 7.1))
    draw_box(ax, 3.6 + i*1.2, 6.2, 0.6, 0.8, f"T{i+1}", C_TOKEN)

# CLS Token
draw_box(ax, 2.4, 6.2, 0.6, 0.8, "[CLS]", C_CLS)
ax.text(3.1, 6.6, "+", fontweight='bold', fontsize=14)

# Positional Encodings
draw_box(ax, 2.4, 5.3, 4.4, 0.4, "+ Positional Encodings (1D/2D)", '#ecf0f1', fontsize=9)

# 3. Transformer Encoder
draw_arrow(ax, (4.6, 6.2), (7.8, 6.6), text="Sequence\nInput")

draw_box(ax, 8, 5.5, 4, 3, "Transformer\nEncoder Layers\n(Self-Attention)", C_ENC, fontsize=12)

# Global Attention Visuals inside Encoder
ax.plot([8.5, 11.5], [7.5, 7.5], 'o-', color='#3498db', alpha=0.5)
ax.plot([8.5, 11.5], [6.5, 6.5], 'o-', color='#3498db', alpha=0.5)
ax.plot([8.5, 11.5], [7.5, 6.5], 'o-', color='#3498db', alpha=0.5)
ax.plot([11.5, 8.5], [7.5, 6.5], 'o-', color='#3498db', alpha=0.5)
ax.text(10, 7, "All tokens attend\nto ALL tokens", ha='center', color='#2980b9', fontsize=9)

# 4. Output MLP
draw_arrow(ax, (12.2, 7.5), (13.5, 7.5), text="[CLS] out")
draw_box(ax, 13.8, 7, 2, 1, "MLP Head", '#ecf0f1')
draw_arrow(ax, (16, 7.5), (17.5, 7.5))
draw_box(ax, 17.8, 7, 1.8, 1, "Cat: 98%", '#d5f5e3')

# 5. ViT Backprop (Red Path)
draw_arrow(ax, (18.7, 8.1), (18.7, 10.5), color='red', lw=2.5, ls='--')
draw_arrow(ax, (18.7, 10.5), (10, 10.5), color='red', lw=2.5, ls='--', text="dL propagates instantly")
draw_arrow(ax, (10, 10.5), (10, 8.7), color='red', lw=2.5, ls='--')
# Fan out to all tokens
draw_arrow(ax, (10, 8.7), (8.2, 8.5), color='red', lw=2, ls='--')
draw_arrow(ax, (10, 8.7), (11.8, 8.5), color='red', lw=2, ls='--')
ax.text(10, 9.8, "Global Error Allocation to ALL patches simultaneously", ha='center', color='red', fontsize=10, fontweight='bold')


# ==========================================
# BOTTOM SECTION: CNN (Local)
# ==========================================
ax.plot([0, 20], [4.5, 4.5], color='black', lw=2, ls=':') # Separator
ax.text(10, 4, "Convolutional Neural Network (CNN) — Local Receptive Field", ha='center', fontsize=16, fontweight='bold')

# 1. Image
draw_box(ax, 0.5, 1, 2, 2, "Input\nImage", C_IMAGE)

# 2. Convolutions
draw_arrow(ax, (2.7, 2), (3.8, 2), text="3x3 Conv")
draw_box(ax, 4, 1.2, 1.6, 1.6, "Feature\nMap 1", C_CNN)
# Draw local connection
ax.plot([2.5, 4], [2.8, 2.5], color='gray', lw=1)
ax.plot([2.5, 4], [2.4, 2.1], color='gray', lw=1)

draw_arrow(ax, (5.8, 2), (6.8, 2), text="3x3 Conv")
draw_box(ax, 7, 1.4, 1.2, 1.2, "Feature\nMap 2", C_CNN)

draw_arrow(ax, (8.4, 2), (9.4, 2), text="... Layers ...")
draw_box(ax, 9.6, 1.6, 0.8, 0.8, "Deep\nMap", C_CNN)

# 3. Output MLP
draw_arrow(ax, (10.6, 2), (13.5, 2), text="Flatten")
draw_box(ax, 13.8, 1.5, 2, 1, "Dense Layers", '#ecf0f1')
draw_arrow(ax, (16, 2), (17.5, 2))
draw_box(ax, 17.8, 1.5, 1.8, 1, "Cat: 98%", '#d5f5e3')

# 4. CNN Backprop (Red Path)
draw_arrow(ax, (18.7, 1.3), (18.7, 0.2), color='red', lw=2.5, ls='--')
draw_arrow(ax, (18.7, 0.2), (10, 0.2), color='red', lw=2.5, ls='--', text="dL restricted to local paths")
draw_arrow(ax, (10, 0.2), (10, 1.4), color='red', lw=2.5, ls='--')
draw_arrow(ax, (9.6, 0.2), (7.6, 1.2), color='red', lw=1.5, ls='--')
draw_arrow(ax, (6.8, 0.2), (4.8, 1.2), color='red', lw=1.5, ls='--')

ax.text(10, -0.3, "Error must slowly trickle backward through restricted local kernels", ha='center', color='red', fontsize=10, fontweight='bold')


plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('assets/vit_vs_cnn_architecture.png', dpi=200, bbox_inches='tight', facecolor=C_BG)
plt.close()
print("Generated assets/vit_vs_cnn_architecture.png successfully!")
