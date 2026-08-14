import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import numpy as np

# --- CONFIGURATION ---
C_BG = '#f8f9fa'
C_MASK_ATTN = '#fadbd8'   # Faded Red for Masked Self-Attention
C_CROSS_ATTN = '#fcf3cf'  # Yellow for Cross-Attention
C_FFN = '#d6eaf8'         # Blue
C_NORM = '#d5f5e3'        # Green
C_ENC_OUT = '#e8daef'     # Purple for Encoder Context Input
C_TEXT = 'black'

def draw_box(ax, x, y, width, height, text, bg_color):
    rect = Rectangle((x, y), width, height, facecolor=bg_color, edgecolor='black', lw=2, zorder=2)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=11, fontweight='bold', zorder=3)

def draw_arrow(ax, start, end, text=None, rad=0.0):
    arrow = FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={rad}", 
                            arrowstyle='-|>', mutation_scale=15, color='gray', lw=2, zorder=1)
    ax.add_patch(arrow)
    if text:
        ax.text((start[0]+end[0])/2, (start[1]+end[1])/2 + 0.1, text, ha='center', va='center', 
                fontsize=9, fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))

# ==========================================
# TRANSFORMER DECODER BLOCK (Vertical)
# ==========================================
fig, ax = plt.subplots(figsize=(12, 16))
ax.set_xlim(0, 12)
ax.set_ylim(0, 18)
ax.axis('off')

# Outer Block Boundary
rect = Rectangle((1.5, 2), 9, 14.5, fill=True, facecolor=C_BG, edgecolor='black', lw=3, linestyle='--', zorder=0)
ax.add_patch(rect)
ax.text(2, 16, "One Decoder Block\n(e.g., GPT-3, standard translation model)", fontsize=14, fontweight='bold', color='gray')

# --- 1. Input Track (Decoder Inputs, e.g., <Start> The quick) ---
ax.text(6, 0.5, "Decoder Input Matrix ($X$)\n(Output generated so far)", ha='center', fontsize=14, fontweight='bold')
draw_arrow(ax, (6, 1.0), (6, 2.5))

# --- 2. Masked Self-Attention (Crucial Difference) ---
draw_box(ax, 4, 3.5, 4, 1.5, "Masked\nSelf-Attention\n(Hides the future)", C_MASK_ATTN)
draw_arrow(ax, (6, 2.5), (6, 3.5)) # Q, K, V feed in
ax.text(5.5, 2.8, "Q, K, V", color='gray', fontweight='bold', fontsize=10)

# Causal Mask Visualizer
mask_verts = np.array([[0,1,1],[0,0,1],[0,0,0]])
ax.imshow(mask_verts, cmap='gray_r', extent=[2, 3.5, 3.5, 5], origin='lower', zorder=5)
ax.text(2.75, 5.2, "Causal Mask Matrix\n(Used in $Q \cdot K^T$)", ha='center', fontsize=9, color='red')

# --- 3. Add & Norm 1 (Masked Attention) ---
draw_box(ax, 4, 6, 4, 1.5, "Add & Norm\nLayerNorm(X + Attn_M(X))", C_NORM)
draw_arrow(ax, (6, 5), (6, 6))
# Residual 1 (Bypasses masked attention)
draw_arrow(ax, (6, 2.8), (2.5, 2.8))
draw_arrow(ax, (2.5, 2.8), (2.5, 6.75))
draw_arrow(ax, (2.5, 6.75), (4, 6.75))

# --- 4. Cross-Attention (Bridges to Encoder) ---
draw_box(ax, 4, 8.5, 4, 1.5, "Cross-Attention\n(Looks at Encoder output)", C_CROSS_ATTN)
# Main path up from previous Norm (Decoder Query)
draw_arrow(ax, (6, 7.5), (6, 8.5))
ax.text(5.5, 8.0, "Q", fontweight='bold', color='gray')

# --- EXTERNAL ENCODER INPUT ---
rect_enc = Rectangle((9, 8.25), 2.5, 3, facecolor=C_ENC_OUT, edgecolor='black', lw=2)
ax.add_patch(rect_enc)
ax.text(10.25, 9.75, "Faded Output of\nFINAL Encoder Block\n(Full Context)", ha='center', va='center', fontsize=11, fontweight='bold', color='purple')
# Encoder output feeding K, V into Cross-Attention
draw_arrow(ax, (10.25, 8.25), (10.25, 7.5))
draw_arrow(ax, (10.25, 7.5), (7, 8.5), rad=0.2) # to Cross-Attention K, V
draw_arrow(ax, (10.25, 7.5), (6.5, 8.5), rad=0.1)
ax.text(9.0, 7.7, "K, V", fontweight='bold', color='purple')

# --- 5. Add & Norm 2 (Cross-Attention) ---
draw_box(ax, 4, 11, 4, 1.5, "Add & Norm\nLayerNorm(X + Attn_C(X))", C_NORM)
draw_arrow(ax, (6, 10), (6, 11))
# Residual 2 (Bypasses cross-attention)
draw_arrow(ax, (6, 8.0), (2.5, 8.0))
draw_arrow(ax, (2.5, 8.0), (2.5, 11.75))
draw_arrow(ax, (2.5, 11.75), (4, 11.75))

# --- 6. Position-wise Feed-Forward Network ---
draw_box(ax, 4, 13.5, 4, 1.5, "Feed-Forward Network\n(Expanded Memory)", C_FFN)
draw_arrow(ax, (6, 12.5), (6, 13.5))

# --- 7. Add & Norm 3 (FFN) ---
draw_box(ax, 4, 16, 4, 1.5, "Add & Norm\nLayerNorm(X + FFN(X))", C_NORM)
draw_arrow(ax, (6, 15), (6, 16))
# Residual 3 (Bypasses FFN)
draw_arrow(ax, (6, 13.0), (2.5, 13.0))
draw_arrow(ax, (2.5, 13.0), (2.5, 16.75))
draw_arrow(ax, (2.5, 16.75), (4, 16.75))

# --- Output -> Prediction layer ---
draw_arrow(ax, (6, 17.5), (6, 18.5))
ax.text(6, 18.8, "Decoder Output\n(Passed to Softmax to predict next word)", ha='center', fontsize=14, fontweight='bold')

plt.suptitle("Topic 6: The Transformer Decoder Block (The Generative Engine)", fontsize=22, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('assets/decoder_block_architecture.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated assets/decoder_block_architecture.png successfully!")
