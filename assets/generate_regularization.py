import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Lasso, Ridge

# 1. Dataset Generation
# We create 50 features, but strictly only 5 of them are useful (the rest are pure noise)
X, y = make_regression(n_samples=100, n_features=50, n_informative=5, noise=10, random_state=42)

# 2. Train Models (simulating a single dense layer)
# Unregularized
model_none = LinearRegression()
model_none.fit(X, y)

# L1 (Lasso)
model_l1 = Lasso(alpha=2.0, random_state=42) # alpha=2.0 to make the sparsity very obvious
model_l1.fit(X, y)

# L2 (Ridge)
model_l2 = Ridge(alpha=10.0, random_state=42) # alpha=10.0 to show heavy shrinkage
model_l2.fit(X, y)

# 3. Visualization Setup
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Plotting function
def plot_weights(ax, weights, title, description, color):
    ax.bar(np.arange(len(weights)), np.abs(weights), color=color, edgecolor='black')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Index (0-49)', fontsize=12)
    ax.text(0.5, 0.95, description, transform=ax.transAxes, 
            fontsize=12, va='top', ha='center', 
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))
    if ax == axes[0]:
        ax.set_ylabel('Absolute Weight Magnitude', fontsize=12)

# We find the global max weight across all 3 models so the Y-axis scale is identical for comparison
max_weight = max(np.max(np.abs(model_none.coef_)), 
                 np.max(np.abs(model_l1.coef_)), 
                 np.max(np.abs(model_l2.coef_)))

plot_weights(axes[0], model_none.coef_, "Unregularized Regression", 
             "High Variance:\nAssigns large weights to\npure noise features.", 'salmon')
             
plot_weights(axes[1], model_l1.coef_, "L1 Regularization (Lasso)", 
             "Sparsity:\nForces noisy weights to exactly zero.\nOnly the 5 true features survive.", 'mediumseagreen')
             
plot_weights(axes[2], model_l2.coef_, "L2 Regularization (Ridge)", 
             "Weight Decay:\nShrinks all weights smoothly.\nNoisy weights are tiny, but none are zero.", 'cornflowerblue')

for ax in axes:
    ax.set_ylim(0, max_weight * 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.suptitle("How Regularization Constrains Neural Network Weights", fontsize=20, fontweight='bold', y=1.08)
plt.savefig("assets/regularization_weights.png", dpi=300, bbox_inches='tight')
print("Successfully generated assets/regularization_weights.png")
