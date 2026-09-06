> 📖 **Navigation:** [← Previous: Part 07: Eigenvalues & Eigenvectors](./07_eigenvalues_and_eigenvectors.md) | [🏠 Index](./README.md) | [Next: Part 09: Covariance Matrix Complete Calculation →](./09_covariance_matrix.md)

---

# PART 8 — EIGENVALUES IN PCA: THE COMPLETE MATHEMATICAL DERIVATION

Principal Component Analysis (PCA) is the premier dimensionality reduction algorithm in data science. This module provides the complete, step-by-step mathematical proof explaining **WHY the principal component directions are the eigenvectors of the sample covariance matrix**.

---

## 8.1 The Setup: Finding the Direction of Maximum Variance

Let $X_c \in \mathbb{R}^{n \times d}$ be a mean-centered data matrix ($n$ observations, $d$ features, column means $= 0$).
The sample covariance matrix is:
$$
\Sigma = \frac{1}{n-1} X_c^T X_c \in \mathbb{R}^{d \times d}
$$

We want to find a unit projection vector $\mathbf{w} \in \mathbb{R}^d$ ($\|\mathbf{w}\|_2 = 1$) such that projecting our data onto $\mathbf{w}$ maximizes the variance of the projected points:

```
                            PCA PROJECTION OBJECTIVE
                 x2                                          x2
                 │      ●  ●  ●                              │        ● ╱
                 │    ●  ●  ●                                │       ● ╱  w (1st Principal Component)
                 │  ●  ●  ●                                  │      ● ╱   (Maximum Spread / Variance!)
                 └──┴────────────► x1                        └──┴────┴───────► x1
                Raw Centered Data                           Projected onto Eigenvector w
```

### 1. Projected Data Variance Formula:
The projected coordinate of observation $\mathbf{x}_i$ is the scalar $z_i = \mathbf{w}^T \mathbf{x}_i$.
Because data is centered ($\mathbb{E}[z] = 0$), the sample variance of the projected data is:
$$
\text{Var}(z) = \frac{1}{n-1} \sum_{i=1}^{n} (\mathbf{w}^T \mathbf{x}_i)^2 = \frac{1}{n-1} \mathbf{w}^T (X_c^T X_c) \mathbf{w} = \mathbf{w}^T \Sigma \mathbf{w}
$$

---

## 8.2 The Lagrangian Optimization Derivation

We formulate the constrained optimization problem:
$$
\max_{\mathbf{w}} \mathbf{w}^T \Sigma \mathbf{w} \quad \text{subject to} \quad \mathbf{w}^T \mathbf{w} = 1
$$

### Step 1: Form the Lagrangian Function
Introduce Lagrange multiplier $\lambda$:
$$
\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^T \Sigma \mathbf{w} - \lambda (\mathbf{w}^T \mathbf{w} - 1)
$$

### Step 2: Differentiate with respect to $\mathbf{w}$
Using matrix calculus identities ($\nabla_{\mathbf{w}} (\mathbf{w}^T \Sigma \mathbf{w}) = 2\Sigma\mathbf{w}$ because $\Sigma = \Sigma^T$):
$$
\nabla_{\mathbf{w}} \mathcal{L} = 2 \Sigma \mathbf{w} - 2 \lambda \mathbf{w} = \mathbf{0}
$$

### Step 3: Solve the Stationarity Condition
Divide by 2:
$$
\Sigma \mathbf{w} = \lambda \mathbf{w}
$$

> [!IMPORTANT]
> **The Core PCA Discovery:**
> The stationarity condition $\Sigma \mathbf{w} = \lambda \mathbf{w}$ is **PRECISELY the fundamental eigenvalue equation**!
> * Therefore, the projection vector $\mathbf{w}$ that maximizes variance **MUST be an eigenvector of the covariance matrix $\Sigma$**.

---

## 8.3 Why the Eigenvalue Equals the Variance

Multiply both sides of $\Sigma \mathbf{w} = \lambda \mathbf{w}$ by $\mathbf{w}^T$ on the left:
$$
\mathbf{w}^T \Sigma \mathbf{w} = \mathbf{w}^T (\lambda \mathbf{w}) = \lambda (\mathbf{w}^T \mathbf{w}) = \lambda (1) = \lambda
$$

* **Crucial Result:** The variance of data projected along eigenvector $\mathbf{w}_i$ is **exactly equal to its corresponding eigenvalue $\lambda_i$**:
  $$
  \text{Var}(\text{Projected Data along } \mathbf{w}_i) = \lambda_i
  $$
* To maximize variance, choose the eigenvector $\mathbf{w}_1$ corresponding to the **largest eigenvalue $\lambda_1 = \lambda_{\max}$**.

---

## 8.4 Subsequent Components & Explained Variance Ratio

* **Second Principal Component ($\mathbf{w}_2$):** Solves the same objective with the additional constraint that it must be orthogonal to the first component ($\mathbf{w}_2 \perp \mathbf{w}_1$). This yields the eigenvector with the second-largest eigenvalue $\lambda_2$.
* **Explained Variance Ratio:** The fraction of total dataset variance captured by the $k$-th principal component is:
  $$
  \text{Explained Variance Ratio}_k = \frac{\lambda_k}{\sum_{i=1}^{d} \lambda_i} = \frac{\lambda_k}{\text{Tr}(\Sigma)}
  $$

---

> 📖 **Navigation:** [← Previous: Part 07: Eigenvalues & Eigenvectors](./07_eigenvalues_and_eigenvectors.md) | [🏠 Index](./README.md) | [Next: Part 09: Covariance Matrix Complete Calculation →](./09_covariance_matrix.md)
