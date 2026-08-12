import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_circles

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# 1. Soft Margin & Support Vectors
def plot_soft_margin():
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Generate linearly separable data with a little noise
    np.random.seed(42)
    X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
    y = [0] * 20 + [1] * 20
    # Add an outlier
    X = np.vstack([X, [0, 2]])
    y = np.append(y, 0)
    
    clf = SVC(kernel='linear', C=0.5) # Soft margin
    clf.fit(X, y)
    
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, edgecolors='k')
    
    # Plot decision boundary and margins
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # Highlight support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=150,
               linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')
    
    ax.set_title('Soft Margin SVM (Allowing Misclassifications for Robustness)', fontsize=14)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    plt.tight_layout()
    plt.savefig('/Users/divyanshverma/Desktop/ml_interview_questions/assets/svm_soft_margin.png', dpi=300)
    plt.close()

# 2. Polynomial Kernel (1D to 2D Projection)
def plot_polynomial_kernel():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1D Data
    dosages = np.array([-3, -2.5, -2, -1, 0, 1, 2, 2.5, 3])
    cured = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0])
    
    ax1.scatter(dosages[cured==0], np.zeros_like(dosages[cured==0]), color='red', s=100, label='Not Cured', edgecolors='k')
    ax1.scatter(dosages[cured==1], np.zeros_like(dosages[cured==1]), color='green', s=100, label='Cured', edgecolors='k')
    ax1.set_title('1D: Not Linearly Separable', fontsize=14)
    ax1.set_xlabel('Dosage')
    ax1.set_yticks([])
    ax1.legend()
    
    # 2D Projection (Dosage vs Dosage^2)
    dosages_sq = dosages ** 2
    ax2.scatter(dosages[cured==0], dosages_sq[cured==0], color='red', s=100, edgecolors='k', label='Not Cured')
    ax2.scatter(dosages[cured==1], dosages_sq[cured==1], color='green', s=100, edgecolors='k', label='Cured')
    ax2.axhline(y=2.5, color='blue', linestyle='--', linewidth=2, label='Support Vector Classifier')
    
    ax2.set_title('2D Projection ($X^2$): Linearly Separable!', fontsize=14)
    ax2.set_xlabel('Dosage (X)')
    ax2.set_ylabel('Dosage Squared ($X^2$)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('/Users/divyanshverma/Desktop/ml_interview_questions/assets/svm_polynomial_kernel.png', dpi=300)
    plt.close()

# 3. RBF Gamma Parameter
def plot_rbf_gamma():
    # Generate circular data
    X, y = make_circles(n_samples=100, factor=0.3, noise=0.1, random_state=42)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, gamma, title in zip([ax1, ax2], [100, 0.1], ['High Gamma (Strict/Overfit)', 'Low Gamma (Relaxed/Smooth)']):
        clf = SVC(kernel='rbf', gamma=gamma)
        clf.fit(X, y)
        
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, edgecolors='k')
        
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx = np.linspace(xlim[0], xlim[1], 100)
        yy = np.linspace(ylim[0], ylim[1], 100)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf.decision_function(xy).reshape(XX.shape)
        
        ax.contourf(XX, YY, Z, alpha=0.3, cmap='coolwarm')
        ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=1, linestyles=['-'])
        
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout()
    plt.savefig('/Users/divyanshverma/Desktop/ml_interview_questions/assets/svm_rbf_gamma.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_soft_margin()
    plot_polynomial_kernel()
    plot_rbf_gamma()
    print("SVM plots successfully generated.")
