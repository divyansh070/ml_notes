import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Configure styles
plt.style.use('bmh')
C_DATA = '#34495e'
C_REG = '#e74c3c'
C_MEAN = '#3498db'
C_RESID = '#e74c3c'
C_TOT = '#3498db'

np.random.seed(42)
x = np.linspace(1, 10, 15)
y = 2.5 * x + np.random.normal(0, 4, len(x)) + 10

mean_y = np.mean(y)
slope, intercept = np.polyfit(x, y, 1)
y_pred = slope * x + intercept

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Total Sum of Squares (SST) - Baseline Mean Model
ax1.scatter(x, y, color=C_DATA, zorder=5, s=60, label='Data Points')
ax1.axhline(mean_y, color=C_MEAN, lw=3, label='Baseline Model (Mean)')
for i in range(len(x)):
    ax1.plot([x[i], x[i]], [y[i], mean_y], color=C_TOT, ls='--', lw=1.5, zorder=4)
    # small square to represent "squared" error
    # width = height = abs(y[i] - mean_y)/3 for visual representation
ax1.set_title("Total Sum of Squares (SST)\nBaseline Model (Predicts Mean)", fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.set_xlabel("Feature (X)")
ax1.set_ylabel("Target (y)")

# Plot 2: Sum of Squared Residuals (SSR) - Regression Model
ax2.scatter(x, y, color=C_DATA, zorder=5, s=60, label='Data Points')
ax2.plot(x, y_pred, color=C_REG, lw=3, label='Regression Model (Best Fit)')
for i in range(len(x)):
    ax2.plot([x[i], x[i]], [y[i], y_pred[i]], color=C_RESID, ls='--', lw=1.5, zorder=4)

ax2.set_title("Sum of Squared Residuals (SSR)\nRegression Model Error", fontsize=14, fontweight='bold')
ax2.legend(loc='upper left')
ax2.set_xlabel("Feature (X)")
ax2.set_ylabel("Target (y)")

plt.suptitle(r"$R^2 = 1 - \frac{SSR}{SST}$ (How much better is the Regression Line than the Mean Line?)", fontsize=18, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('assets/r2_score_visual.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generated assets/r2_score_visual.png")
