import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

os.makedirs('assets', exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#f8f9fa')

# --- LEFT PANEL: The Binary Analogy (Nitish's Insight) ---
ax1.set_title("The Inspiration: Binary Encoding", fontsize=16, fontweight='bold')
ax1.axis('off')

# Draw a grid showing binary counting 0 to 7
binary_vals = [f"{i:03b}" for i in range(8)]
ax1.text(0.5, 0.9, "Highest Bit\n(Slow)", ha='center', fontweight='bold', color='#e74c3c')
ax1.text(1.5, 0.9, "Middle Bit\n(Medium)", ha='center', fontweight='bold', color='#f39c12')
ax1.text(2.5, 0.9, "Lowest Bit\n(Fast)", ha='center', fontweight='bold', color='#3498db')

for row, b in enumerate(binary_vals):
    ax1.text(-0.5, 0.75 - row*0.1, f"Pos {row}:", ha='right', va='center', fontsize=12, fontweight='bold')
    colors = ['#e74c3c', '#f39c12', '#3498db']
    for col, bit in enumerate(b):
        rect = Rectangle((col, 0.7 - row*0.1), 1, 0.1, facecolor='white', edgecolor=colors[col], lw=2)
        ax1.add_patch(rect)
        ax1.text(col + 0.5, 0.75 - row*0.1, bit, ha='center', va='center', fontsize=14, fontweight='bold', color=colors[col])

# --- RIGHT PANEL: The Continuous Transformation (Sine/Cosine) ---
ax2.set_title("The Solution: Continuous Wave Encoding", fontsize=16, fontweight='bold')
x = np.linspace(0, 10, 500)

# High frequency (like the lowest bit)
ax2.plot(x, np.sin(x), color='#3498db', lw=2, label='Dim 0 & 1 (High Frequency)')
# Medium frequency
ax2.plot(x, np.sin(x/2.5), color='#f39c12', lw=2, label='Dim 2 & 3 (Medium Frequency)')
# Low frequency (like the highest bit)
ax2.plot(x, np.sin(x/6), color='#e74c3c', lw=2, label='Dim 511 & 512 (Low Frequency)')

ax2.set_xlabel("Word Position in Sentence", fontsize=12, fontweight='bold')
ax2.set_ylabel("Positional Vector Value (-1 to 1)", fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower left')

# Anotate
ax2.annotate('Rapid Fluctuation\n(Acts like the lowest bit)', xy=(2, np.sin(2)), xytext=(2, 1.5),
             arrowprops=dict(facecolor='#3498db', shrink=0.05), ha='center', color='#3498db', fontweight='bold')
ax2.annotate('Slow Fluctuation\n(Acts like the highest bit)', xy=(6, np.sin(6/6)), xytext=(6, 1.5),
             arrowprops=dict(facecolor='#e74c3c', shrink=0.05), ha='center', color='#e74c3c', fontweight='bold')

plt.suptitle("CampusX Intuition: Positional Encoding as Continuous Binary", fontsize=22, fontweight='bold', y=1.05)
plt.savefig('assets/campusx_positional_encoding.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated assets/campusx_positional_encoding.png successfully!")
