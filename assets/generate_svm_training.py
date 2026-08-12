import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_blobs
import warnings

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')

def plot_svm_training():
    # Generate perfectly separable data
    X, y = make_blobs(n_samples=60, centers=2, random_state=6, cluster_std=0.8)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Common bounds
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx = np.linspace(x_min, x_max, 50)
    yy = np.linspace(y_min, y_max, 50)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    # Step 1: Early Training
    clf1 = SGDClassifier(loss='hinge', max_iter=2, learning_rate='constant', eta0=0.01, random_state=42)
    clf1.fit(X, y)
    
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, edgecolors='k')
    Z1 = clf1.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z1, colors='k', levels=[-1, 0, 1], alpha=0.8, linestyles=['--', '-', '--'])
    ax.set_title("Epoch 2: Just Starting (High Loss)", fontsize=13)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    
    # Step 2: Mid Training
    clf2 = SGDClassifier(loss='hinge', max_iter=8, learning_rate='constant', eta0=0.01, random_state=42)
    clf2.fit(X, y)
    
    ax = axes[1]
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, edgecolors='k')
    Z2 = clf2.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z2, colors='k', levels=[-1, 0, 1], alpha=0.8, linestyles=['--', '-', '--'])
    ax.set_title("Epoch 8: Mid-Training (Rotating to find margins)", fontsize=13)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    
    # Step 3: Convergence
    from sklearn.svm import SVC
    clf3 = SVC(kernel='linear', C=1000) # True SVM solution
    clf3.fit(X, y)
    
    ax = axes[2]
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, edgecolors='k')
    Z3 = clf3.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z3, colors='k', levels=[-1, 0, 1], alpha=0.8, linestyles=['--', '-', '--'])
    
    # Highlight support vectors on the final plot
    ax.scatter(clf3.support_vectors_[:, 0], clf3.support_vectors_[:, 1], s=200,
               linewidth=2, facecolors='none', edgecolors='gold', label='Support Vectors')
    
    ax.set_title("Convergence: Optimal Maximum Margin", fontsize=13)
    ax.legend(loc='lower right')
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig('/Users/divyanshverma/Desktop/ml_interview_questions/assets/svm_training_process.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_svm_training()
    print("SVM training plot generated.")
