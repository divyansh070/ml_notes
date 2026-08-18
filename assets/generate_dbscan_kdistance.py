import matplotlib.pyplot as plt
import numpy as np

# Configure styles
plt.style.use('bmh')
C_LINE = '#3498db'
C_KNEE = '#e74c3c'
C_TEXT = '#2c3e50'

# Generate synthetic k-distance data
np.random.seed(42)
# Create a curve that starts flat and suddenly spikes up
x = np.arange(0, 100)
# Exponential growth curve to simulate k-distance
y = np.exp(x / 15.0) / 10.0 + np.random.normal(0, 0.5, len(x))
y = np.sort(y)

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#ffffff')

ax.plot(x, y, color=C_LINE, lw=4, zorder=2)

# Knee point
knee_x = 75
knee_y = y[knee_x]

ax.scatter(knee_x, knee_y, color=C_KNEE, s=150, zorder=3, edgecolor='black', lw=1.5)

# Annotations
ax.axhline(knee_y, color=C_KNEE, ls='--', lw=2, zorder=1)
ax.axvline(knee_x, color=C_KNEE, ls='--', lw=2, zorder=1)

ax.annotate('Optimal $\epsilon$ (The "Knee")', 
            xy=(knee_x, knee_y), xytext=(40, knee_y + 15),
            arrowprops=dict(facecolor=C_TEXT, shrink=0.05, width=2, headwidth=10),
            fontsize=16, fontweight='bold', color=C_TEXT)

ax.set_title("K-Distance Graph for Choosing Epsilon ($\epsilon$)", fontsize=18, fontweight='bold', color=C_TEXT)
ax.set_xlabel("Points sorted by distance to $k$-th Nearest Neighbor", fontsize=14)
ax.set_ylabel("$k$-Distance", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)

# Hide ticks for a cleaner conceptual look
ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.savefig('assets/dbscan_kdistance_knee.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated assets/dbscan_kdistance_knee.png successfully!")
