import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

plt.style.use('seaborn-v0_8-whitegrid')

# 1. Generate Toy 2D Dataset (Slightly noisy moons)
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)

# Split into train/test to verify the OOB score accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 2. Train Models
# Model 1: Single Unconstrained Decision Tree
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

# Model 2: Bagging Classifier (500 trees)
# Setting oob_score=True automatically calculates the out-of-bag evaluation score!
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    bootstrap=True, oob_score=True, n_jobs=-1, random_state=42
)
bag_clf.fit(X_train, y_train)

# 3. Extract and Prove OOB Score Accuracy
print(f"Bagging OOB Score (Free Validation): {bag_clf.oob_score_:.4f}")
y_pred = bag_clf.predict(X_test)
print(f"Actual Test Set Accuracy:            {accuracy_score(y_test, y_pred):.4f}")

# 4. Plotting Boundaries
def plot_boundaries(X, y, clf, ax, title):
    x0s = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
    x1s = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X_new = np.c_[x0.ravel(), x1.ravel()]
    
    y_pred = clf.predict(X_new).reshape(x0.shape)
    
    ax.contourf(x0, x1, y_pred, alpha=0.3, cmap=plt.cm.brg)
    ax.plot(X[:, 0][y==0], X[:, 1][y==0], "bs", alpha=0.6)
    ax.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", alpha=0.6)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(r"$x_1$", fontsize=14)
    ax.set_ylabel(r"$x_2$", fontsize=14, rotation=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

plot_boundaries(X_train, y_train, tree_clf, axes[0], "Decision Tree\n(Highly Jagged / Overfitted)")
plot_boundaries(X_train, y_train, bag_clf, axes[1], "Bagging Ensemble of 500 Trees\n(Smooth / Generalized)")

plt.tight_layout()
plt.savefig("/Users/divyanshverma/Desktop/ml_interview_questions/assets/Bagging_Boundaries_Comparison.png", dpi=300)
plt.close()
