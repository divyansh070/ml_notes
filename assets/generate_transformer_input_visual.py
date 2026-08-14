import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- CONFIGURATION ---
C_EMBED = '#d5f5e3'  # Soft Green
C_POS = '#d6eaf8'    # Soft Blue
C_FINAL = '#fcf3cf'  # Soft Yellow
C_TEXT = 'black'

def draw_vector(ax, start_x, start_y, values, title, bg_color):
    """Draws a 1D vector as a row of connected squares."""
    # Title above the vector
    ax.text(start_x + 2, start_y + 1.2, title, ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw the squares and values
    for i, val in enumerate(values):
        rect = Rectangle((start_x + i, start_y), 1, 1, facecolor=bg_color, edgecolor='black', lw=2)
        ax.add_patch(rect)
        ax.text(start_x + i + 0.5, start_y + 0.5, str(val), ha='center', va='center', fontsize=12, fontweight='bold', color=C_TEXT)

# ==========================================
# TRANSFORMER INPUT PIPELINE
# ==========================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis('off')

# --- WORD 0 (pos = 0) ---
ax.text(0.5, 4.5, 'Word 0\n("The")', ha='center', va='center', fontsize=14, fontweight='bold', color='gray')

# Embedding (Invented semantic values)
draw_vector(ax, 1.5, 4, [0.5, -0.2, 0.1, 0.8], "Semantic Embedding", C_EMBED)

# Plus Sign
ax.text(6, 4.5, "+", ha='center', va='center', fontsize=24, fontweight='bold')

# Positional Encoding (Exact math from notes)
draw_vector(ax, 6.5, 4, [0.0, 1.0, 0.0, 1.0], "Positional Encoding (pos=0)", C_POS)

# Equals Sign
ax.text(11, 4.5, "=", ha='center', va='center', fontsize=24, fontweight='bold')

# Final Input
draw_vector(ax, 11.5, 4, [0.5, 0.8, 0.1, 1.8], "Final Transformer Input", C_FINAL)


# --- WORD 1 (pos = 1) ---
ax.text(0.5, 1.5, 'Word 1\n("Dog")', ha='center', va='center', fontsize=14, fontweight='bold', color='gray')

# Embedding (Invented semantic values)
draw_vector(ax, 1.5, 1, [-0.1, 0.9, 0.4, -0.5], "Semantic Embedding", C_EMBED)

# Plus Sign
ax.text(6, 1.5, "+", ha='center', va='center', fontsize=24, fontweight='bold')

# Positional Encoding (Exact math from notes)
draw_vector(ax, 6.5, 1, [0.84, 0.54, 0.01, 1.0], "Positional Encoding (pos=1)", C_POS)

# Equals Sign
ax.text(11, 1.5, "=", ha='center', va='center', fontsize=24, fontweight='bold')

# Final Input
draw_vector(ax, 11.5, 1, [0.74, 1.44, 0.41, 0.5], "Final Transformer Input", C_FINAL)

# --- ANNOTATIONS ---
# Draw an arrow showing how the final inputs go into the Transformer
ax.annotate('', xy=(13.5, 0.2), xytext=(13.5, 0.8),
            arrowprops=dict(facecolor='gray', shrink=0.05, width=3, headwidth=10))
ax.text(13.5, -0.2, "Passed into Attention Layers", ha='center', va='center', fontsize=12, fontweight='bold', color='gray')

plt.suptitle("Transformer Input: Element-wise Addition of Vectors", fontsize=20, fontweight='bold', y=0.98)
plt.savefig('assets/transformer_input_addition.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated assets/transformer_input_addition.png successfully!")
