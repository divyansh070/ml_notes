import numpy as np
import matplotlib.pyplot as plt

# 1. Define the Loss Function and its Gradient
# f(x, y) = x^2 + 100y^2 (A classic stretched paraboloid/ravine)
def loss_function(x, y):
    return x**2 + 100 * y**2

def gradient(x, y):
    return np.array([2 * x, 200 * y])

# 2. Optimizer Implementations (From Scratch)
def sgd(start_point, lr, iterations):
    path = [start_point]
    current_point = np.array(start_point)
    for _ in range(iterations):
        grad = gradient(*current_point)
        current_point = current_point - lr * grad
        path.append(current_point)
    return np.array(path)

def momentum(start_point, lr, iterations, beta=0.9):
    path = [start_point]
    current_point = np.array(start_point)
    v = np.zeros(2)
    for _ in range(iterations):
        grad = gradient(*current_point)
        # Standard Momentum Formula
        v = beta * v + (1 - beta) * grad
        current_point = current_point - lr * v
        path.append(current_point)
    return np.array(path)

def adam(start_point, lr, iterations, beta1=0.9, beta2=0.999, epsilon=1e-8):
    path = [start_point]
    current_point = np.array(start_point)
    m = np.zeros(2)
    v = np.zeros(2)
    for t in range(1, iterations + 1):
        grad = gradient(*current_point)
        # Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * grad
        # Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * (grad**2)
        
        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # Update weights
        current_point = current_point - lr * m_hat / (np.sqrt(v_hat) + epsilon)
        path.append(current_point)
    return np.array(path)

# 3. Setup the 2D Contour Plot
x_range = np.linspace(-3.5, 3.5, 200)
y_range = np.linspace(-1.0, 1.0, 200)
X, Y = np.meshgrid(x_range, y_range)
Z = loss_function(X, Y)

fig, ax = plt.subplots(figsize=(14, 9))
# Use log scale for contour levels to see the steep ravine clearly
levels = np.logspace(-1, 3, 20)
contour = ax.contourf(X, Y, Z, levels=levels, cmap='coolwarm', alpha=0.8)
fig.colorbar(contour, label='Loss Value (Log Scale)')

# 4. Run Optimizers
start_point = [3.0, 0.5]
iterations = 50

# Note: Learning rates are carefully tuned for a fair comparison.
# Adam inherently requires a larger learning rate because it normalizes updates.
path_sgd = sgd(start_point, lr=0.009, iterations=iterations)
path_momentum = momentum(start_point, lr=0.009, iterations=iterations)
path_adam = adam(start_point, lr=0.3, iterations=iterations)

# 5. Plot Paths
ax.plot(path_sgd[:, 0], path_sgd[:, 1], color='red', marker='o', markersize=4, linewidth=1.5, 
        label='SGD: Bounces violently across ridges, extremely slow horizontal progress')
ax.plot(path_momentum[:, 0], path_momentum[:, 1], color='blue', marker='s', markersize=4, linewidth=1.5, 
        label='Momentum: Dampens vertical bounces and accelerates horizontal progress')
ax.plot(path_adam[:, 0], path_adam[:, 1], color='green', marker='^', markersize=4, linewidth=1.5, 
        label='Adam: Perfectly dampens oscillations, tracks straight to the minimum')

# Mark start and end points
ax.scatter(*start_point, color='black', s=120, zorder=5, label='Start Point (3.0, 0.5)')
ax.scatter(0, 0, color='gold', s=250, marker='*', edgecolor='black', zorder=5, label='Global Minimum (0, 0)')

# Formatting
ax.set_title("Navigating Pathological Curvature (Ravine)\nSGD vs Momentum vs Adam", fontsize=16, fontweight='bold')
ax.set_xlabel("Weight 1", fontsize=12)
ax.set_ylabel("Weight 2", fontsize=12)
ax.legend(loc='lower left', fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.savefig("assets/optimizers_pathological_curvature.png", dpi=300, bbox_inches='tight')
print("Successfully generated assets/optimizers_pathological_curvature.png")
