> 📖 **Navigation:** [← Previous: Part 08: Eigenvalues in Principal Component Analysis (PCA)](./08_eigenvalues_in_pca.md) | [🏠 Index](./README.md) | [Next: Part 10: Complete PCA Walkthrough from Scratch →](./10_pca_scratch_walkthrough.md)

---

# PART 9 — THE SAMPLE COVARIANCE MATRIX

The **Covariance Matrix** $\Sigma$ captures the variance of individual features along its diagonal and the pairwise linear associations between features in its off-diagonals.

---

## 9.1 Mathematical Definition & Derivation

For $n$ observations across $d$ features, let $X_c \in \mathbb{R}^{n \times d}$ be the mean-centered data matrix ($x_{ij} - \bar{x}_j$).
The sample covariance between feature $j$ and feature $k$ is:
$$
\Sigma_{jk} = \text{Cov}(X_j, X_k) = \frac{1}{n-1} \sum_{i=1}^{n} (x_{ij} - \bar{x}_j)(x_{ik} - \bar{x}_k)
$$

### Matrix Formulation:
$$
\Sigma = \frac{1}{n-1} X_c^T X_c =
\begin{bmatrix}
\text{Var}(X_1) & \text{Cov}(X_1, X_2) & \dots & \text{Cov}(X_1, X_d) \\
\text{Cov}(X_2, X_1) & \text{Var}(X_2) & \dots & \text{Cov}(X_2, X_d) \\
\vdots & \vdots & \ddots & \vdots \\
\text{Cov}(X_d, X_1) & \text{Cov}(X_d, X_2) & \dots & \text{Var}(X_d)
\end{bmatrix}
$$

* **Why $X_c^T X_c$ computes covariance:** Row $j$ of $X_c^T$ is the centered column for feature $j$. Multiplying by Column $k$ of $X_c$ computes the exact dot product $\sum_{i=1}^n x_{ij} x_{ik}$.
* **Symmetry:** Because $\text{Cov}(X_j, X_k) = \text{Cov}(X_k, X_j)$, $\Sigma$ is **always symmetric** ($\Sigma = \Sigma^T$).

---

## 9.2 Step-by-Step Hand Calculation Example ($n=3, d=2$)

Dataset: $X = [1, 2, 3]^T$ (Study Hours), $Y = [2, 3, 7]^T$ (Exam Score).
1. Means: $\bar{x} = 2.0, \bar{y} = 4.0$.
2. Centered deviations:
   * $X_c = [-1, 0, 1]^T$
   * $Y_c = [-2, -1, 3]^T$
3. Compute sum of squares & cross-products:
   * $\sum (x_i - \bar{x})^2 = (-1)^2 + 0^2 + 1^2 = 2$
   * $\sum (y_i - \bar{y})^2 = (-2)^2 + (-1)^2 + 3^2 = 14$
   * $\sum (x_i - \bar{x})(y_i - \bar{y}) = (-1)(-2) + (0)(-1) + (1)(3) = 2 + 0 + 3 = 5$
4. Divide by $n-1 = 2$:
   $$
   \Sigma = \begin{bmatrix} 2/2 & 5/2 \\ 5/2 & 14/2 \end{bmatrix} = \begin{bmatrix} 1.0 & 2.5 \\ 2.5 & 7.0 \end{bmatrix}
   $$

---

## 9.3 Covariance vs. Correlation

* **Covariance ($\text{Cov}(X, Y)$):** Scale-dependent (e.g. converting height from meters to centimeters multiplies covariance by 100).
* **Pearson Correlation ($\rho_{XY}$):** Scale-invariant normalization:
  $$
  \rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1.0, +1.0]
  $$
* **When StandardScaler matters in ML:** If features have vastly different scales (e.g. Salary in \$100,000s vs Age in 10s), raw covariance is dominated by the large-scale feature. Standardizing ($z = \frac{x - \mu}{\sigma}$) makes the covariance matrix identical to the **Correlation Matrix**, giving all features equal initial weight in PCA.

---

## 9.4 Proof that $\Sigma$ is Always Positive Semidefinite (PSD)

For any test vector $\mathbf{u} \in \mathbb{R}^d$:
$$
\mathbf{u}^T \Sigma \mathbf{u} = \mathbf{u}^T \left( \frac{1}{n-1} X_c^T X_c \right) \mathbf{u} = \frac{1}{n-1} (X_c \mathbf{u})^T (X_c \mathbf{u}) = \frac{1}{n-1} \|X_c \mathbf{u}\|_2^2 \ge 0
$$

* Because the squared Euclidean norm of any real vector is strictly $\ge 0$, **projected variance can NEVER be negative**.
* Thus, all eigenvalues of a covariance matrix satisfy $\lambda_i \ge 0$.

---

## Advanced / Optional — Do not study until Core track is complete

### A.1 Mahalanobis Distance (Scale & Correlation Invariant Distance)
Measures distance from point $\mathbf{x}$ to mean $\boldsymbol{\mu}$ scaled by inverse covariance:
$$
D_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})}
$$
*(Used in multivariate outlier detection and Gaussian Discriminant Analysis).*

---

> 📖 **Navigation:** [← Previous: Part 08: Eigenvalues in Principal Component Analysis (PCA)](./08_eigenvalues_in_pca.md) | [🏠 Index](./README.md) | [Next: Part 10: Complete PCA Walkthrough from Scratch →](./10_pca_scratch_walkthrough.md)
