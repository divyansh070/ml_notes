> 📖 **Navigation:** [← Previous: Part 21: Linear Algebra for ML Synthesis](./21_linear_algebra_for_ml_synthesis.md) | [🏠 Index](./README.md) | [Next: Part 23: Gram-Schmidt & QR Decomposition →](./23_gram_schmidt_and_qr_decomposition.md)

---

# PART 22 — COMPLETE PCA & COVARIANCE WALKTHROUGH

*Advanced Companion Module — Deep Dive on Principal Component Analysis from Scratch*

Principal Component Analysis (PCA) is the premier linear dimensionality reduction algorithm in machine learning. This companion module brings together sample covariance, Lagrangian optimization, eigendecomposition, projection, and reconstruction through a complete, step-by-step 9-step numerical walkthrough.

---

## 22.1 The Sample Covariance Matrix

For $n$ observations across $d$ features, let $X_c \in \mathbb{R}^{n \times d}$ be the mean-centered data matrix where column means are subtracted ($\bar{x}_j = 0$):

$$
\Sigma = \frac{1}{n-1} X_c^T X_c \in \mathbb{R}^{d \times d}
$$

* **Diagonal entries $\Sigma_{jj}$:** The sample variance $\text{Var}(X_j)$ of feature $j$.
* **Off-diagonal entries $\Sigma_{jk}$:** The sample covariance $\text{Cov}(X_j, X_k)$ between feature $j$ and feature $k$.
* **Symmetry:** Because $\text{Cov}(X_j, X_k) = \text{Cov}(X_k, X_j)$, $\Sigma$ is **always symmetric** ($\Sigma = \Sigma^T$).
* **Positive Semidefinite:** Projected variance $\mathbf{u}^T \Sigma \mathbf{u} = \frac{1}{n-1}\|X_c \mathbf{u}\|_2^2 \ge 0$, guaranteeing all eigenvalues $\lambda_i \ge 0$.

---

## 22.2 Why Principal Components Are Eigenvectors: The Lagrangian Proof

We seek a unit projection vector $\mathbf{w} \in \mathbb{R}^d$ ($\|\mathbf{w}\|_2 = 1$) such that projecting the centered data onto $\mathbf{w}$ **maximizes the variance** of the projected points:

$$
\max_{\mathbf{w}} \text{Var}(X_c \mathbf{w}) = \max_{\mathbf{w}} \mathbf{w}^T \Sigma \mathbf{w} \quad \text{subject to} \quad \mathbf{w}^T \mathbf{w} = 1
$$

### Step 1: Form the Lagrangian Function
Introduce Lagrange multiplier $\lambda$:

$$
\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^T \Sigma \mathbf{w} - \lambda (\mathbf{w}^T \mathbf{w} - 1)
$$

### Step 2: Differentiate with Respect to $\mathbf{w}$ and Set to Zero
Using matrix calculus ($\nabla_{\mathbf{w}} (\mathbf{w}^T \Sigma \mathbf{w}) = 2\Sigma\mathbf{w}$ because $\Sigma = \Sigma^T$):

$$
\nabla_{\mathbf{w}} \mathcal{L} = 2 \Sigma \mathbf{w} - 2 \lambda \mathbf{w} = \mathbf{0}
$$

### Step 3: Rearrange to the Fundamental Equation

$$
\Sigma \mathbf{w} = \lambda \mathbf{w}
$$

> [!IMPORTANT]
> **The Core PCA Discovery:**
> The variance-maximizing projection vector $\mathbf{w}$ **must be an eigenvector of the covariance matrix $\Sigma$**!
> Multiplying on the left by $\mathbf{w}^T$:
>
> $$
> \mathbf{w}^T \Sigma \mathbf{w} = \lambda (\mathbf{w}^T \mathbf{w}) = \lambda (1) = \lambda
> $$
>
> The projected variance along eigenvector $\mathbf{w}_i$ is **exactly equal to its eigenvalue $\lambda_i$**!
> To maximize variance, select the eigenvector with the largest eigenvalue $\lambda_{\max}$.

---

## 22.3 Complete 9-Step Numerical PCA Walkthrough from Scratch

We have a 2D dataset with $n = 4$ observations and $d = 2$ features:

$$
X = \begin{bmatrix}
2 & 4 \\
3 & 6 \\
5 & 8 \\
6 & 10
\end{bmatrix}
$$

---

### Step 1: Compute Feature Means

$$
\bar{x}_1 = \frac{2 + 3 + 5 + 6}{4} = \frac{16}{4} = 4.0, \quad \bar{x}_2 = \frac{4 + 6 + 8 + 10}{4} = \frac{28}{4} = 7.0
$$

---

### Step 2: Mean-Center the Data ($X_c = X - \bar{X}$)

$$
X_c = \begin{bmatrix}
2 - 4 & 4 - 7 \\
3 - 4 & 6 - 7 \\
5 - 4 & 8 - 7 \\
6 - 4 & 10 - 7
\end{bmatrix} =
\begin{bmatrix}
-2 & -3 \\
-1 & -1 \\
1 & 1 \\
2 & 3
\end{bmatrix}
$$

---

### Step 3: Compute Sample Covariance Matrix $\Sigma = \frac{1}{n-1} X_c^T X_c$

$$
X_c^T X_c = \begin{bmatrix} -2 & -1 & 1 & 2 \\ -3 & -1 & 1 & 3 \end{bmatrix} \begin{bmatrix} -2 & -3 \\ -1 & -1 \\ 1 & 1 \\ 2 & 3 \end{bmatrix} =
\begin{bmatrix} 4 + 1 + 1 + 4 & 6 + 1 + 1 + 6 \\ 6 + 1 + 1 + 6 & 9 + 1 + 1 + 9 \end{bmatrix} =
\begin{bmatrix} 10 & 14 \\ 14 & 20 \end{bmatrix}
$$

Divide by $n - 1 = 3$:

$$
\Sigma = \frac{1}{3} \begin{bmatrix} 10 & 14 \\ 14 & 20 \end{bmatrix} \approx \begin{bmatrix} 3.33 & 4.67 \\ 4.67 & 6.67 \end{bmatrix}
$$

---

### Step 4: Solve for Eigenvalues of $X_c^T X_c$

$$
\det(X_c^T X_c - \lambda I) = (10 - \lambda)(20 - \lambda) - 14^2 = \lambda^2 - 30\lambda + (200 - 196) = \lambda^2 - 30\lambda + 4 = 0
$$

Using the quadratic formula:

$$
\lambda = \frac{30 \pm \sqrt{900 - 16}}{2} = \frac{30 \pm \sqrt{884}}{2} = \frac{30 \pm 29.732}{2}
$$

* $\lambda_1 \approx 29.866$
* $\lambda_2 \approx 0.134$

Dividing by $n - 1 = 3$ gives covariance eigenvalues:
* $\lambda_1(\Sigma) = 9.955$
* $\lambda_2(\Sigma) = 0.045$

---

### Step 5: Find the Dominant Eigenvector $\mathbf{v}_1$
Substitute $\lambda_1 = 29.866$ into $(X_c^T X_c - \lambda I)\mathbf{v} = \mathbf{0}$:

$$
\begin{bmatrix} 10 - 29.866 & 14 \\ 14 & 20 - 29.866 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} =
\begin{bmatrix} -19.866 & 14 \\ 14 & -9.866 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

$$
-19.866 v_1 + 14 v_2 = 0 \implies v_2 = \frac{19.866}{14} v_1 \approx 1.419 v_1
$$

Normalize to unit length:

$$
\|\mathbf{v}_1\|_2 = \sqrt{1^2 + 1.419^2} = \sqrt{1 + 2.014} = \sqrt{3.014} \approx 1.736
$$

$$
\mathbf{v}_1 = \begin{bmatrix} 1 / 1.736 \\ 1.419 / 1.736 \end{bmatrix} \approx \begin{bmatrix} 0.576 \\ 0.817 \end{bmatrix}
$$

---

### Step 6: Compute Explained Variance Ratio

$$
\text{Explained Variance Ratio}_1 = \frac{\lambda_1}{\lambda_1 + \lambda_2} = \frac{29.866}{29.866 + 0.134} = \frac{29.866}{30.000} = \mathbf{99.55\%}
$$

Keeping just this 1 principal component captures $99.55\%$ of all information in the original 2D data!

---

### Step 7: Project Centered Data onto 1D Subspace ($z = X_c \mathbf{v}_1$)

$$
\mathbf{z} = X_c \mathbf{v}_1 =
\begin{bmatrix}
-2 & -3 \\
-1 & -1 \\
1 & 1 \\
2 & 3
\end{bmatrix}
\begin{bmatrix} 0.576 \\ 0.817 \end{bmatrix}
$$

* $z_1 = -2(0.576) - 3(0.817) = -1.152 - 2.451 = \mathbf{-3.603}$
* $z_2 = -1(0.576) - 1(0.817) = -0.576 - 0.817 = \mathbf{-1.393}$
* $z_3 = 1(0.576) + 1(0.817) = 0.576 + 0.817 = \mathbf{+1.393}$
* $z_4 = 2(0.576) + 3(0.817) = 1.152 + 2.451 = \mathbf{+3.603}$

1D Compressed Coordinates: $\mathbf{z} = [-3.603, -1.393, 1.393, 3.603]^T$.

---

### Step 8: Reconstruct Data Back to 2D ($\hat{X} = \mathbf{z} \mathbf{v}_1^T + \bar{X}$)
For Sample 1 ($z_1 = -3.603$):

$$
\hat{\mathbf{x}}_1 = -3.603 \begin{bmatrix} 0.576 \\ 0.817 \end{bmatrix} + \begin{bmatrix} 4.0 \\ 7.0 \end{bmatrix} =
\begin{bmatrix} -2.075 \\ -2.944 \end{bmatrix} + \begin{bmatrix} 4.0 \\ 7.0 \end{bmatrix} =
\begin{bmatrix} 1.925 \\ 4.056 \end{bmatrix}
$$

*(Original point was $[2.0, 4.0]^T$; reconstruction is $[1.93, 4.06]^T$ — nearly identical!).*

---

### Step 9: Residual Error Analysis
* The residual error vector $\mathbf{e}_i = \mathbf{x}_i - \hat{\mathbf{x}}_i$ is **strictly orthogonal to principal direction $\mathbf{v}_1$**.
* The total sum of squared reconstruction errors across all points equals the discarded eigenvalue:

$$
\sum_{i=1}^{n} \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|_2^2 = \lambda_2 = 0.134 \quad (0.45\% \text{ of total variance})
$$

---

> 📖 **Navigation:** [← Previous: Part 21: Linear Algebra for ML Synthesis](./21_linear_algebra_for_ml_synthesis.md) | [🏠 Index](./README.md) | [Next: Part 23: Gram-Schmidt & QR Decomposition →](./23_gram_schmidt_and_qr_decomposition.md)
