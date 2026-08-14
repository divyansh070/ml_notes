import numpy as np
import matplotlib.pyplot as plt

def get_positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            denominator = np.power(10000, i / d_model)
            pe[pos, i] = np.sin(pos / denominator)
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / denominator)
    return pe

seq_len = 50
d_model = 128

pe_matrix = get_positional_encoding(seq_len, d_model)

plt.figure(figsize=(14, 8))
plt.pcolormesh(pe_matrix, cmap='RdBu_r', shading='auto', vmin=-1, vmax=1)
plt.colorbar(label='Encoding Value')

plt.xlabel('Embedding Dimension ($d_{model}$)', fontsize=14, fontweight='bold', labelpad=10)
plt.ylabel('Position (Word Index in Sequence)', fontsize=14, fontweight='bold', labelpad=10)
plt.title('Transformer Positional Encoding: Injecting Time via Sine & Cosine Waves', fontsize=18, fontweight='bold', pad=20)

plt.gca().invert_yaxis() # Put position 0 at the top

# Add explanation text box
textstr = "Notice how the left side (lower dimensions) oscillates rapidly, acting like the second-hand on a clock.\nThe right side (higher dimensions) oscillates slowly, acting like the hour-hand.\nThis gives every position a unique mathematical time-stamp!"
props = dict(boxstyle='round,pad=0.7', facecolor='#f8f9fa', alpha=1.0, edgecolor='gray', lw=2)
plt.gca().text(0.5, -0.15, textstr, transform=plt.gca().transAxes, fontsize=13,
        verticalalignment='top', horizontalalignment='center', bbox=props, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(bottom=0.25) # Make room for the text box below the plot

plt.savefig('assets/transformer_pe.png', dpi=300, bbox_inches='tight')
plt.close()

print("Successfully generated assets/transformer_pe.png")
