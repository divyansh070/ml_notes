import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')

def plot_soft_svm_training():
    # Generate linearly separable data with a little noise
    np.random.seed(42)
    X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
    y = [0] * 20 + [1] * 20
    # Add an extreme outlier that forces a soft margin decision
    X = np.vstack([X, [0, 2]])
    y = np.append(y, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Common bounds
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx = np.linspace(x_min, x_max, 50)
    yy = np.linspace(y_min, y_max, 50)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    # Step 1: Early Training (SGDClassifier with high alpha for soft margin behavior)
    clf1 = SGDClassifier(loss='hinge', alpha=0.5, max_iter=2, learning_rate='constant', eta0=0.01, random_state=43)
    clf1.fit(X, y)
    
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, edgecolors='k')
    Z1 = clf1.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z1, colors='k', levels=[-1, 0, 1], alpha=0.8, linestyles=['--', '-', '--'])
    ax.set_title("Epoch 2: Severely Contorted by Outlier", fontsize=13)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    
    # Step 2: Mid Training
    clf2 = SGDClassifier(loss='hinge', alpha=0.5, max_iter=8, learning_rate='constant', eta0=0.01, random_state=43)
    clf2.fit(X, y)
    
    ax = axes[1]
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, edgecolors='k')
    Z2 = clf2.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z2, colors='k', levels=[-1, 0, 1], alpha=0.8, linestyles=['--', '-', '--'])
    ax.set_title("Epoch 8: Accepting Violation to Widen Margin", fontsize=13)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    
    # Step 3: Convergence (True SVC with C parameter)
    clf3 = SVC(kernel='linear', C=0.2) # Soft margin solution
    clf3.fit(X, y)
    
    ax = axes[2]
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, edgecolors='k')
    Z3 = clf3.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z3, colors='k', levels=[-1, 0, 1], alpha=0.8, linestyles=['--', '-', '--'])
    
    # Highlight support vectors on the final plot
    ax.scatter(clf3.support_vectors_[:, 0], clf3.support_vectors_[:, 1], s=200,
               linewidth=2, facecolors='none', edgecolors='gold', label='Support Vectors (incl. Outlier)')
    
    ax.set_title("Convergence: Optimal Soft Margin", fontsize=13)
    ax.legend(loc='lower right')
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig('/Users/divyanshverma/Desktop/ml_interview_questions/assets/svm_soft_training_process.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_soft_svm_training()
    print("Soft SVM training plot generated.")
