import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

# --- CONFIGURATION ---
C_RAW = '#f9ebea'      # Light Red for Raw Embeddings
C_WEIGHTS = '#fdf2e9'  # Light Orange for Attention Weights
C_FINAL = '#e8f8f5'    # Light Teal for Contextualized Vectors

def draw_vector(ax, x, y, values, title, color):
    """Draws a vertical vector of squares."""
    ax.text(x + 0.5, y + len(values) + 0.3, title, ha='center', va='bottom', fontsize=11, fontweight='bold')
    for i, val in enumerate(reversed(values)):
        rect = Rectangle((x, y + i), 1, 1, facecolor=color, edgecolor='black', lw=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.5, y + i + 0.5, f"{val:.1f}", ha='center', va='center', fontsize=11, fontweight='bold')

def draw_arrow(ax, start, end, text=None):
    arrow = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=15, color='gray', lw=2)
    ax.add_patch(arrow)
    if text:
        ax.text((start[0]+end[0])/2, (start[1]+end[1])/2 + 0.2, text, ha='center', va='bottom', 
                fontsize=10, fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))

# ==========================================
# BEFORE VS. AFTER ATTENTION VISUALIZER
# ==========================================
fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 18)
ax.set_ylim(0, 7)
ax.axis('off')

# --- 1. RAW EMBEDDINGS (BEFORE) ---
ax.text(3, 6, "1. Raw Embeddings (Isolated Meaning)", ha='center', fontsize=14, fontweight='bold')
# "Bank" might have financial features (top) and nature features (bottom)
bank_raw = [0.9, 0.8, 0.2, 0.1] 
# "River" has strong nature features
river_raw = [0.0, 0.1, 0.9, 0.8]

draw_vector(ax, 1.5, 1, bank_raw, '"Bank"', C_RAW)
draw_vector(ax, 4.5, 1, river_raw, '"River"', C_RAW)

ax.text(2, 0.5, "Financial\nFeatures", ha='right', fontsize=9, color='gray')
ax.text(2, 3.5, "Nature\nFeatures", ha='right', fontsize=9, color='gray')

# --- 2. ATTENTION WEIGHTS (THE ROUTING) ---
draw_arrow(ax, (6, 3), (7.5, 3), "Dot Product\n+ Softmax")

ax.text(10, 6, "2. Attention Weights (The Matrix)", ha='center', fontsize=14, fontweight='bold')

# The Attention Matrix Box
rect = Rectangle((8, 1.5), 4, 3, facecolor=C_WEIGHTS, edgecolor='black', lw=2)
ax.add_patch(rect)

# Explicit percentages
ax.text(10, 3.5, "How much should\n'Bank' look at 'River'?", ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(10, 2.3, "'Bank' = 10% self\n'Bank' = 90% 'River'", ha='center', va='center', fontsize=12, fontweight='bold', color='blue')

# --- 3. CONTEXTUALIZED VECTORS (AFTER) ---
draw_arrow(ax, (12.5, 3), (14, 3), "Matrix Multiply\n(Blending)")

ax.text(15.5, 6, "3. Contextualized Output (Blended)", ha='center', fontsize=14, fontweight='bold')

# The Math: New Bank = (0.10 * Bank_Raw) + (0.90 * River_Raw)
# Financial: (0.1 * 0.9) + (0.9 * 0.0) = 0.09
# Nature:    (0.1 * 0.1) + (0.9 * 0.8) = 0.73
bank_contextualized = [0.09, 0.17, 0.83, 0.73]

draw_vector(ax, 15, 1, bank_contextualized, 'New "Bank"\n(Context: Nature)', C_FINAL)

# Explanation Box
explanation = (
    "The Transformation of 'Bank':\n"
    "Before Attention, 'Bank' had strong financial vector values (0.9, 0.8).\n"
    "Because the Attention Matrix routed 90% of 'River' into 'Bank',\n"
    "the new vector for 'Bank' is mathematically overridden by nature features.\n"
    "The network now definitively knows we are talking about a muddy riverbank!"
)
ax.text(9, -0.5, explanation, ha='center', va='top', fontsize=12, fontweight='bold', 
        bbox=dict(facecolor='#fcf3cf', edgecolor='orange', pad=0.8))

plt.suptitle("Visualizing the Change: Vectors Before vs. After Self-Attention", fontsize=20, fontweight='bold', y=1.0)
plt.savefig('assets/vectors_before_after_attention.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated assets/vectors_before_after_attention.png successfully!")
