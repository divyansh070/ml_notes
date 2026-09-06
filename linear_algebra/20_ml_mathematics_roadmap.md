> 📖 **Navigation:** [← Previous: Part 19: Entropy & Information Gain Mathematics](./19_entropy_and_information_gain.md) | [🏠 Index](./README.md) | [Next: Part 21: Paper-and-Pencil Exam Checklist →](./21_paper_and_pencil_checklist.md)

---

# PART 20 — ML MATHEMATICS ROADMAP & ALGORITHM MAPPING

How does every machine learning algorithm connect to linear algebra, calculus, and probability? Use this master roadmap to understand exactly which mathematical concepts power each model and why they are necessary.

---

## 20.1 Master Algorithm-to-Mathematics Matrix

| Machine Learning Algorithm | Primary Mathematical Foundations | Core Matrix Operations & Key Equations | Why This Math is Needed / Geometric Picture |
| :--- | :--- | :--- | :--- |
| **Linear Regression (OLS)** | Systems of equations, Orthogonal projection, Matrix inverses, QR decomposition | $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y} = X^+ \mathbf{y}$<br>$R\mathbf{w} = Q^T \mathbf{y}$ | Projects target vector $\mathbf{y}$ orthogonally onto the column space $C(X)$ spanned by feature vectors. |
| **Ridge Regression ($L_2$)** | Quadratic forms, Positive Definite matrices, SVD spectral filtering | $\mathbf{w}_{\text{ridge}} = (X^T X + \alpha I)^{-1} X^T \mathbf{y}$ | Adds $\alpha I$ to guarantee $X^T X + \alpha I \succ 0$ is invertible; shrinks weights along small singular values. |
| **Lasso Regression ($L_1$)** | Non-differentiable optimization, Subgradients, $L_1$ ball geometry | $\min \frac{1}{2n}\|X\mathbf{w}-\mathbf{y}\|_2^2 + \alpha \|\mathbf{w}\|_1$<br>Soft-thresholding operator | Diamond-shaped $L_1$ constraint ball has sharp axis corners, forcing less predictive feature weights to exact zero. |
| **ElasticNet** | Convex combinations of $L_1$ and $L_2$ norms | $\min \text{MSE} + \alpha_1 \|\mathbf{w}\|_1 + \frac{\alpha_2}{2} \|\mathbf{w}\|_2^2$ | Combines feature selection of Lasso with group selection stability of Ridge for highly correlated features. |
| **Logistic Regression** | Sigmoid activation, Log-Loss (Cross-Entropy), Convex optimization | $\hat{p} = \sigma(\mathbf{w}^T \mathbf{x} + b)$<br>$\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{n} X^T (\hat{\mathbf{p}} - \mathbf{y})$ | Separates binary classes with a linear hyperplane in log-odds space; minimizes convex negative log-likelihood. |
| **Support Vector Machines (SVM)** | Hyperplane geometry, Orthogonal distances, Quadratic programming, Mercer kernels | Margin $= \frac{2}{\|\mathbf{w}\|_2}$<br>Dual: $\max \sum \alpha_i - \frac{1}{2} \sum \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$ | Maximizes geometric margin between classes; uses Positive Semidefinite Gram matrix $K \succeq 0$ for non-linear boundaries. |
| **Principal Component Analysis (PCA)** | Sample covariance (PSD), Eigendecomposition, Orthogonal projection | $\Sigma = \frac{1}{n-1} X_c^T X_c$<br>$\Sigma \mathbf{v}_i = \lambda_i \mathbf{v}_i, \quad Z = X_c V_k$ | Rotates coordinate system to align with directions of maximal data variance; discards orthogonal low-variance axes. |
| **Singular Value Decomposition (SVD)** | Matrix factorization, Low-rank approximations, Moore-Penrose pseudoinverse | $X_c = U \Sigma V^T$<br>$A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$ (Eckart-Young) | Factors any rectangular data matrix into Rotate $\to$ Scale $\to$ Rotate; optimal rank-$k$ compression (LoRA, Latent Semantic Analysis). |
| **K-Means Clustering** | Euclidean distance ($L_2$ norm), Centroid optimization, Voronoi partitioning | $\arg\min_{S} \sum_{k=1}^K \sum_{\mathbf{x} \in S_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|_2^2$ | Partitions feature space into convex Voronoi cells by iteratively updating orthogonal cluster centroids. |
| **K-Nearest Neighbors (KNN)** | Vector norms ($L_1, L_2, L_\infty$), Metric spaces, Curse of Dimensionality | $d_2(\mathbf{x}, \mathbf{z}) = \sqrt{\sum (x_i - z_i)^2}$<br>Cosine: $1 - \frac{\mathbf{x} \cdot \mathbf{z}}{\|\mathbf{x}\|_2 \|\mathbf{z}\|_2}$ | Classifies queries based on local neighborhood geometry in high-dimensional feature space. |
| **Decision Trees & Random Forests** | Probability, Shannon Entropy, Information Gain, Gini Impurity | $H(S) = -\sum p_i \log_2 p_i$<br>$\text{Gini}(S) = 1 - \sum p_i^2$ | Recursively splits feature space with axis-aligned orthogonal hyperplanes to maximize information gain (purity). |
| **Gaussian Mixture Models (GMM)** | Multivariate Normal distribution, Covariance matrices, Mahalanobis distance | $\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \Sigma) = \frac{1}{\sqrt{(2\pi)^d |\Sigma|}} e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})}$ | Fits ellipsoidal probabilistic cluster envelopes using full, diagonal, or spherical covariance matrices $\Sigma$. |
| **Neural Networks (MLP / CNN / Transformer)** | Affine transformations, Matrix calculus, Chain rule (Backprop), Hessians | $\mathbf{z} = W\mathbf{x} + \mathbf{b}, \quad \mathbf{a} = \phi(\mathbf{z})$<br>$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$ | Chains affine transformations and non-linearities to learn hierarchical feature representations; optimizes via backpropagation. |

---

## 20.2 The Three Mathematical Pillars of Machine Learning

```
                           THE THREE PILLARS OF ML
                                      ▲
                                     ╱ ╲
                                    ╱   ╲
                                   ╱     ╲
                                  ╱  ML   ╲
                                 ╱         ╲
       ┌────────────────────────▼───────────▼────────────────────────┐
       │                                                             │
LINEAR ALGEBRA                  MULTIVARIATE CALCULUS            PROBABILITY & STATS
• Vectors & Spaces              • Gradients & Directional Deriv. • Variance & Covariance
• Matrix Transformations        • Chain Rule & Backprop          • Gaussian Distributions
• Eigendecomposition & SVD      • Hessian & Loss Curvature       • Maximum Likelihood (MLE)
• Orthogonal Projections        • Taylor Approximations          • Shannon Entropy & KL Div
       │                                                             │
       └────────────────────────┬───────────┬────────────────────────┘
                                ╲     │     ╱
                                 ╲    │    ╱
                                  ▼   ▼   ▼
                           SCALABLE ML SYSTEMS
```

---

> 📖 **Navigation:** [← Previous: Part 19: Entropy & Information Gain Mathematics](./19_entropy_and_information_gain.md) | [🏠 Index](./README.md) | [Next: Part 21: Paper-and-Pencil Exam Checklist →](./21_paper_and_pencil_checklist.md)
