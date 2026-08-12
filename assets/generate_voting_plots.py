import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

plt.style.use('seaborn-v0_8-whitegrid')

# 1. Generate Toy 2D Dataset (Slightly noisy moons)
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)

# 2. Train Models
# Model 1: Single Unconstrained Decision Tree
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X, y)

# We use BaggingClassifier to train 500 different trees on random subsets of data.
# By default, scikit-learn's BaggingClassifier uses Soft Voting if predict_proba is available.
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    bootstrap=True, n_jobs=-1, random_state=42
)
bag_clf.fit(X, y)

def hard_voting_predict(X_new):
    # Extract hard votes from all 500 trees and take the majority vote
    preds = np.array([tree.predict(X_new) for tree in bag_clf.estimators_])
    return np.round(np.mean(preds, axis=0))

# 3. Plotting Boundaries
def plot_boundaries(X, y, pred_func, ax, title):
    x0s = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
    x1s = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X_new = np.c_[x0.ravel(), x1.ravel()]
    
    y_pred = pred_func(X_new).reshape(x0.shape)
    
    ax.contourf(x0, x1, y_pred, alpha=0.3, cmap=plt.cm.brg)
    ax.plot(X[:, 0][y==0], X[:, 1][y==0], "bs", alpha=0.6)
    ax.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", alpha=0.6)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(r"$x_1$", fontsize=14)
    ax.set_ylabel(r"$x_2$", fontsize=14, rotation=0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# Plot 1: Single Decision Tree
plot_boundaries(X, y, tree_clf.predict, axes[0], "Single Decision Tree\n(Severe Overfitting)")

# Plot 2: Hard Voting (500 Trees)
plot_boundaries(X, y, hard_voting_predict, axes[1], "Hard Voting (500 Trees)\n(Smoother, but some rigid edges)")

# Plot 3: Soft Voting (500 Trees)
plot_boundaries(X, y, bag_clf.predict, axes[2], "Soft Voting (500 Trees)\n(Highly Generalized, smooth curves)")

plt.tight_layout()
plt.savefig("/Users/divyanshverma/Desktop/ml_interview_questions/assets/Voting_Boundaries_Comparison.png", dpi=300)
plt.close()
print("Saved Voting_Boundaries_Comparison.png")
