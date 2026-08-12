import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress ConvergenceWarnings for cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 1. Dataset
X, y = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)

# 2. Setup Figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# 3. Model setup (warm_start=True is critical to allow pausing and resuming)
mlp = MLPClassifier(hidden_layer_sizes=(16, 16), activation='relu', solver='adam', 
                    warm_start=True, max_iter=1, random_state=42)

# Meshgrid for contourf decision boundary mapping
x_min, x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

epochs_to_plot = [10, 50, 150, 500]
plot_idx = 0
current_epoch = 0

print("Training MLP and generating plots...")

for target_epoch in epochs_to_plot:
    # Train sequentially up to the target epoch
    while current_epoch < target_epoch:
        mlp.fit(X, y)
        current_epoch += 1
        
    # Plotting
    ax = axes[plot_idx]
    
    # Predict over meshgrid to get regions
    Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Contour plot for regions
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # Scatter plot for data points
    ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=20, edgecolor='k', label='Outer Circle (Class 0)')
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', s=20, edgecolor='k', label='Inner Circle (Class 1)')
    
    if target_epoch == 10:
        title = f"Epoch {target_epoch}: Random Initial Boundary"
    elif target_epoch == 50:
        title = f"Epoch {target_epoch}: Starting to Bend"
    elif target_epoch == 150:
        title = f"Epoch {target_epoch}: Wrapping the Structure"
    elif target_epoch == 500:
        title = f"Epoch {target_epoch}: Perfect Non-Linear Separation"

    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    
    if plot_idx == 0:
        ax.legend(loc='upper right')
        
    plot_idx += 1

plt.tight_layout()
plt.suptitle("How a Multi-Layer Perceptron Learns Over Time", fontsize=16, fontweight='bold', y=1.02)
plt.savefig("assets/mlp_decision_boundary.png", dpi=300, bbox_inches='tight')
print("Successfully generated assets/mlp_decision_boundary.png")
