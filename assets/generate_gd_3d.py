import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Define the Cost Function (A perfect bowl: Z = X^2 + Y^2)
def loss_function(w1, w2):
    return w1**2 + w2**2

# Define the gradients (derivative of w^2 is 2w)
def gradients(w1, w2):
    return 2*w1, 2*w2

# 2. Generate the 3D Surface Data
w1_range = np.linspace(-10, 10, 50)
w2_range = np.linspace(-10, 10, 50)
W1, W2 = np.meshgrid(w1_range, w2_range)
Loss = loss_function(W1, W2)

# 3. Simulate Gradient Descent
w1_current = 8.0
w2_current = 8.0
learning_rate = 0.1
iterations = 15

# Lists to store the path for visualization
path_w1, path_w2, path_loss = [], [], []

for _ in range(iterations + 1):
    # Record current position
    current_loss = loss_function(w1_current, w2_current)
    path_w1.append(w1_current)
    path_w2.append(w2_current)
    path_loss.append(current_loss)
    
    # Calculate gradients
    grad_w1, grad_w2 = gradients(w1_current, w2_current)
    
    # Update weights (take a step down the hill)
    w1_current = w1_current - (learning_rate * grad_w1)
    w2_current = w2_current - (learning_rate * grad_w2)

# 4. Visualization Setup
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot the translucent loss surface
surf = ax.plot_surface(W1, W2, Loss, cmap='viridis', alpha=0.6, edgecolor='none')

# Plot the Gradient Descent Path
ax.plot(path_w1, path_w2, path_loss, color='red', marker='o', markersize=6, 
        linewidth=3, label='Gradient Descent Path')

# 5. Formatting and Annotations
ax.set_title("Gradient Descent on a 3D Loss Surface\n(Finding the Global Minimum)", fontsize=16, fontweight='bold')
ax.set_xlabel('Weight 1', fontsize=12, labelpad=10)
ax.set_ylabel('Weight 2', fontsize=12, labelpad=10)
ax.set_zlabel('Loss (Error)', fontsize=12, labelpad=10)

# Annotate Start Point
ax.text(path_w1[0], path_w2[0], path_loss[0] + 15, "Start (8, 8)", 
        color='black', fontsize=12, fontweight='bold', ha='center')

# Annotate Global Minimum
ax.text(0, 0, -25, "Global Minimum (0, 0, 0)", 
        color='black', fontsize=12, fontweight='bold', ha='center')

# Adjust view angle for best perspective
ax.view_init(elev=30, azim=-45)

ax.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig("assets/gradient_descent_3d.png", dpi=300, bbox_inches='tight')
print("Successfully generated assets/gradient_descent_3d.png")
