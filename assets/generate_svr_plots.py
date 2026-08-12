import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')

def generate_linear_svr():
    m = 50
    X = 2 * np.random.rand(m, 1)
    y = (4 + 3 * X + np.random.randn(m, 1)).ravel()

    svm_reg1 = SVR(kernel="linear", epsilon=1.5)
    svm_reg2 = SVR(kernel="linear", epsilon=0.5)
    svm_reg1.fit(X, y)
    svm_reg2.fit(X, y)

    def plot_svm_regression(svm_reg, X, y, axes):
        x1s = np.linspace(axes[0], axes[1], 100).reshape(-1, 1)
        y_pred = svm_reg.predict(x1s)
        plt.plot(x1s, y_pred, "k-", linewidth=2, label=r"$\hat{y}$")
        plt.plot(x1s, y_pred + svm_reg.epsilon, "k--")
        plt.plot(x1s, y_pred - svm_reg.epsilon, "k--")
        
        # Calculate support vectors for plotting
        y_pred_X = svm_reg.predict(X)
        off_margin = (np.abs(y - y_pred_X) >= svm_reg.epsilon)
        support_indices = np.where(off_margin)[0]
        
        plt.scatter(X[support_indices], y[support_indices], s=180, facecolors='#FFAAAA')
        plt.plot(X, y, "bo")
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.legend(loc="upper left", fontsize=18)
        plt.axis(axes)

    fig, axes = plt.subplots(ncols=2, figsize=(12, 5), sharey=True)
    plt.sca(axes[0])
    plot_svm_regression(svm_reg1, X, y, [0, 2, 3, 11])
    plt.title(r"$\epsilon = {}$ (Wide Street)".format(svm_reg1.epsilon), fontsize=18)
    plt.ylabel(r"$y$", fontsize=18, rotation=0)
    plt.annotate(
        '', xy=(1.0, 4 + 3 * 1.0 + svm_reg1.epsilon), xycoords='data',
        xytext=(1.0, 4 + 3 * 1.0 - svm_reg1.epsilon),
        textcoords='data', arrowprops={'arrowstyle': '<->', 'linewidth': 2}
    )
    plt.text(0.91, 5.6, r"$\epsilon$", fontsize=20)
    
    plt.sca(axes[1])
    plot_svm_regression(svm_reg2, X, y, [0, 2, 3, 11])
    plt.title(r"$\epsilon = {}$ (Narrow Street)".format(svm_reg2.epsilon), fontsize=18)
    plt.tight_layout()
    plt.savefig("/Users/divyanshverma/Desktop/ml_interview_questions/assets/svr_linear.png", dpi=300)
    plt.close()

def generate_poly_svr():
    m = 100
    X = 2 * np.random.rand(m, 1) - 1
    y = (0.2 + 0.1 * X + 0.5 * X**2 + np.random.randn(m, 1)/10).ravel()

    svm_poly_reg1 = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
    svm_poly_reg2 = SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1)
    svm_poly_reg1.fit(X, y)
    svm_poly_reg2.fit(X, y)

    def plot_svm_regression(svm_reg, X, y, axes):
        x1s = np.linspace(axes[0], axes[1], 100).reshape(-1, 1)
        y_pred = svm_reg.predict(x1s)
        plt.plot(x1s, y_pred, "k-", linewidth=2, label=r"$\hat{y}$")
        plt.plot(x1s, y_pred + svm_reg.epsilon, "k--")
        plt.plot(x1s, y_pred - svm_reg.epsilon, "k--")
        
        y_pred_X = svm_reg.predict(X)
        off_margin = (np.abs(y - y_pred_X) >= svm_reg.epsilon)
        support_indices = np.where(off_margin)[0]
        
        plt.scatter(X[support_indices], y[support_indices], s=180, facecolors='#FFAAAA')
        plt.plot(X, y, "bo")
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.legend(loc="upper left", fontsize=18)
        plt.axis(axes)

    fig, axes = plt.subplots(ncols=2, figsize=(12, 5), sharey=True)
    plt.sca(axes[0])
    plot_svm_regression(svm_poly_reg1, X, y, [-1, 1, 0, 1])
    plt.title(r"Polynomial, $C={}, \epsilon = {}$".format(svm_poly_reg1.C, svm_poly_reg1.epsilon), fontsize=18)
    plt.ylabel(r"$y$", fontsize=18, rotation=0)
    
    plt.sca(axes[1])
    plot_svm_regression(svm_poly_reg2, X, y, [-1, 1, 0, 1])
    plt.title(r"Polynomial, $C={}, \epsilon = {}$".format(svm_poly_reg2.C, svm_poly_reg2.epsilon), fontsize=18)
    plt.tight_layout()
    plt.savefig("/Users/divyanshverma/Desktop/ml_interview_questions/assets/svr_poly.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    generate_linear_svr()
    generate_poly_svr()
    print("SVR plots generated.")
