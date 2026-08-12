import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

plt.style.use('seaborn-v0_8-whitegrid')

# 1. Generate noisy quadratic dataset
np.random.seed(42)
m = 200
X = np.random.rand(m, 1) * 2 - 1 # range [-1, 1]
# y = 0.5 X^2 + X + noise
y = 0.5 * X**2 + X + 0.1 * np.random.randn(m, 1)

# 2. Train two DecisionTreeRegressors
tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg2 = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_reg1.fit(X, y)
tree_reg2.fit(X, y)

# 3. Plotting function
def plot_regression_predictions(tree_reg, X, y, axes=[-1, 1, -1.2, 1.8], ylabel="$y$"):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$X$", fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18, rotation=0)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred, "r-", linewidth=3, label=r"$\hat{y}$")

# 4. Generate side-by-side plots
plt.figure(figsize=(12, 5))

plt.subplot(121)
plot_regression_predictions(tree_reg1, X, y)
plt.title("max_depth=2 (4 Steps)", fontsize=16)
plt.legend(loc="upper left", fontsize=14)

plt.subplot(122)
plot_regression_predictions(tree_reg2, X, y, ylabel=None)
plt.title("max_depth=3 (8 Steps)", fontsize=16)

plt.tight_layout()
plt.savefig("/Users/divyanshverma/Desktop/ml_interview_questions/assets/decision_tree_regression_predictions.png", dpi=300)
plt.close()
print("Saved decision_tree_regression_predictions.png")
