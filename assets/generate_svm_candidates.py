import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

plt.style.use('seaborn-v0_8-whitegrid')

def plot_hyperplane_candidates():
    # Generate perfectly separable data
    X, y = make_blobs(n_samples=60, centers=2, random_state=6, cluster_std=0.8)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Common bounds
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    x_fit = np.linspace(x_min, x_max, 100)
    
    # Candidate 1: Bad slope
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, edgecolors='k')
    # Custom bad line 1
    y_fit1 = -0.3 * x_fit + 3.8
    ax.plot(x_fit, y_fit1, '-k', linewidth=2)
    # Margins for candidate 1
    ax.plot(x_fit, y_fit1 + 0.3, '--k', alpha=0.5)
    ax.plot(x_fit, y_fit1 - 0.3, '--k', alpha=0.5)
    ax.set_title("Candidate 1: Valid but tiny margin (Fragile)", fontsize=13)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    
    # Candidate 2: Bad intercept
    ax = axes[1]
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, edgecolors='k')
    # Custom bad line 2
    y_fit2 = -1.2 * x_fit + 11.2
    ax.plot(x_fit, y_fit2, '-k', linewidth=2)
    ax.plot(x_fit, y_fit2 + 0.2, '--k', alpha=0.5)
    ax.plot(x_fit, y_fit2 - 0.2, '--k', alpha=0.5)
    ax.set_title("Candidate 2: Valid but too close to Red (Skewed)", fontsize=13)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    
    # Candidate 3: Optimal SVM
    ax = axes[2]
    clf = SVC(kernel='linear', C=1000) # Hard margin to show clear support vectors
    clf.fit(X, y)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, edgecolors='k')
    
    # Plot decision boundary and margins
    xx = np.linspace(x_min, x_max, 30)
    yy = np.linspace(y_min, y_max, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.8,
               linestyles=['--', '-', '--'])
    
    # Highlight support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=200,
               linewidth=2, facecolors='none', edgecolors='gold', label='Support Vectors')
    
    ax.set_title("Optimal SVM: Maximum Margin (Robust)", fontsize=13)
    ax.legend(loc='lower right')
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig('/Users/divyanshverma/Desktop/ml_interview_questions/assets/svm_candidates.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_hyperplane_candidates()
    print("SVM candidates plot generated.")
