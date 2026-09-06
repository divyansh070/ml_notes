> 📖 **Navigation:** [← Previous: Part 19: Entropy & Information Gain Mathematics](./19_entropy_and_information_gain.md) | [🏠 Index](./README.md) | [Next: Part 21: Paper-and-Pencil Exam Checklist →](./21_paper_and_pencil_checklist.md)

---

# PART 19 — ML MATHEMATICS ROADMAP TABLE

| Machine Learning Algorithm | Primary Mathematical Foundations | Key Equations / Operations |
| :--- | :--- | :--- |
| **Linear Regression** | Matrix Inverses, Least Squares, QR Decomposition, Pseudoinverse | $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y} = X^+ \mathbf{y}$, $R\mathbf{w} = Q^T \mathbf{y}$ |
| **Ridge Regression ($L_2$)** | Positive Definite Inversion, Quadratic Forms, Regularization | $(X^T X + \alpha I)^{-1} X^T \mathbf{y}$, $(1 - \alpha \lambda) \mathbf{w}$ |
| **Lasso Regression ($L_1$)** | L1 Geometry, Non-differentiable Optimization, Subgradients | $\text{MSE} + \alpha \sum_{j=1}^{p} \lvert w_j \rvert \implies \text{Sparsity}$ |
| **Logistic Regression** | Sigmoid Activation, Log-Loss (BCE), Gradients | $\sigma(z) = \frac{1}{1 + e^{-z}}$, $\nabla_{\mathbf{w}} = \frac{1}{N} X^T (\hat{\mathbf{p}} - \mathbf{y})$ |
| **K-Nearest Neighbors (KNN)** | Vector Norms, Metric Spaces, Curse of Dimensionality | $d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$ |
| **K-Means Clustering** | Euclidean Distance, Centroids, Optimization (Lloyd's) | $\arg\min_{\mu} \sum_{i=1}^{n} \lVert \mathbf{x}_i - \mu_k \rVert_2^2$ |
| **Support Vector Machines (SVM)** | Hyperplane Geometry, Projections, Quadratic Programming | $\text{Margin} = \frac{2}{\lVert \mathbf{w} \rVert_2}$, Kernel Trick $K(\mathbf{x}, \mathbf{z})$ |
| **PCA** | Covariance Matrix (PSD), Eigendecomposition, Orthogonal Projections | $\Sigma \mathbf{v} = \lambda \mathbf{v}$, $\mathbf{x}^T \Sigma \mathbf{x} \ge 0$, $Z = X_c V_k$ |
| **Singular Value Decomposition (SVD)** | Matrix Factorization, Geometric Transformations, Pseudoinverse | $A = U \Sigma V^T$, $A^+ = V \Sigma^+ U^T$ |
| **Decision Trees** | Probability, Shannon Entropy, Information Gain, Gini Impurity | $H(S) = -\sum_{i=1}^{C} p_i \log_2 p_i$, $\text{Gini} = 1 - \sum_{i=1}^{C} p_i^2$ |
| **Neural Networks / Deep Learning** | Linear Transformations, Hessians (Positive Definite), Chain Rule | $\mathbf{z} = W\mathbf{x} + \mathbf{b}$, $H \succ 0 \implies \text{Local Minimum}$ |

---

> 📖 **Navigation:** [← Previous: Part 19: Entropy & Information Gain Mathematics](./19_entropy_and_information_gain.md) | [🏠 Index](./README.md) | [Next: Part 21: Paper-and-Pencil Exam Checklist →](./21_paper_and_pencil_checklist.md)
