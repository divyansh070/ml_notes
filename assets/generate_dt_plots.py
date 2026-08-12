import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Generate Toy 2D Dataset
X, y = make_moons(n_samples=200, noise=0.25, random_state=42)

# 2. Train Two Models
# Unconstrained (severe overfitting)
tree_clf_unconstrained = DecisionTreeClassifier(random_state=42)
tree_clf_unconstrained.fit(X, y)

# Pruned (generalized)
tree_clf_pruned = DecisionTreeClassifier(min_samples_leaf=5, max_depth=3, random_state=42)
tree_clf_pruned.fit(X, y)

# 3. Generate Decision Boundary Plots
def plot_decision_boundary(clf, X, y, axes):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=plt.cm.brg)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$", fontsize=14, rotation=0)

plt.figure(figsize=(12, 5))
plt.subplot(121)
plot_decision_boundary(tree_clf_unconstrained, X, y, [-1.5, 2.5, -1, 1.5])
plt.title("Unconstrained Tree (Overfitting)", fontsize=16)

plt.subplot(122)
plot_decision_boundary(tree_clf_pruned, X, y, [-1.5, 2.5, -1, 1.5])
plt.title("Pruned Tree (min_samples_leaf=5, max_depth=3)", fontsize=16)

plt.tight_layout()
plt.savefig("/Users/divyanshverma/Desktop/ml_interview_questions/assets/Tree_Boundaries_Comparison.png", dpi=300)
plt.close()

# 4. Generate Tree Structure Plot
plt.figure(figsize=(14, 10))
plot_tree(tree_clf_pruned, filled=True, rounded=True, class_names=["Blue Square", "Green Triangle"], feature_names=["x1", "x2"])
plt.title("Pruned Tree Structure", fontsize=18)
plt.savefig("/Users/divyanshverma/Desktop/ml_interview_questions/assets/Pruned_Tree_Structure.png", dpi=300)
plt.close()

print("Decision Tree plots generated.")
