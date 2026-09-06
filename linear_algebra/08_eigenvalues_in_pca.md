> 📖 **Navigation:** [← Previous: Part 07: Eigenvalues & Eigenvectors (13-Step Derivation)](./07_eigenvalues_and_eigenvectors.md) | [🏠 Index](./README.md) | [Next: Part 09: Covariance Matrix (Complete Hand Calculation) →](./09_covariance_matrix.md)

---

# PART 7 — EIGENVALUES IN PRINCIPAL COMPONENT ANALYSIS (PCA)

---

## 7.1 The Chain of Logic: Data $\to$ Covariance $\to$ Eigenvalues $\to$ Components

```
     RAW DATA (X)          CENTER DATA (X - mu)        COVARIANCE MATRIX (Sigma)
   [x1, y1]                     [x1', y1']                [ Var(x)   Cov(x,y) ]
   [x2, y2]             ──►     [x2', y2']          ──►   [ Cov(x,y) Var(y)   ]
   [x3, y3]                     [x3', y3']                           │
                                                                     ▼
   DIMENSIONALITY REDUCTION      PROJECT DATA              EIGENDECOMPOSITION
   Keep top k components   ◄──   Z = X_centered @ V_k  ◄── Sigma @ v = lambda * v
   (Max variance retained)                                 (Sort lambda1 >= lambda2)
```

1. **Why do we care about eigenvectors in PCA?**
   * PCA seeks the straight line in space along which the data varies the most (maximum variance).
   * The direction of maximum variance is mathematically the **principal eigenvector** of the data's covariance matrix $\Sigma$.
2. **Why do we care about eigenvalues in PCA?**
   * The eigenvalue $\lambda_i$ equals the exact numerical **variance** of the data when projected onto eigenvector $\mathbf{v}_i$.
   * A larger eigenvalue means more information/variance is preserved along that axis.

---

## 7.2 Why Eigenvectors Maximize Variance

Let $\mathbf{u}$ be a unit projection vector ($\|\mathbf{u}\|_2 = 1$). The variance of the projected data points is:

$$
\text{Variance}(\text{Projection}) = \mathbf{u}^T \Sigma \mathbf{u}
$$

To find the direction $\mathbf{u}$ that maximizes this variance, we set up the **Lagrangian optimization**:

$$
\mathcal{L}(\mathbf{u}, \lambda) = \mathbf{u}^T \Sigma \mathbf{u} - \lambda (\mathbf{u}^T \mathbf{u} - 1)
$$

Taking the partial derivative with respect to $\mathbf{u}$ and setting it to $\mathbf{0}$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{u}} = 2\Sigma \mathbf{u} - 2\lambda \mathbf{u} = \mathbf{0} \quad \implies \quad \Sigma \mathbf{u} = \lambda \mathbf{u}
$$

* **The Takeaway:** The condition for maximum variance is *identically* the eigenvalue equation $\Sigma \mathbf{u} = \lambda \mathbf{u}$.

---

> 📖 **Navigation:** [← Previous: Part 07: Eigenvalues & Eigenvectors (13-Step Derivation)](./07_eigenvalues_and_eigenvectors.md) | [🏠 Index](./README.md) | [Next: Part 09: Covariance Matrix (Complete Hand Calculation) →](./09_covariance_matrix.md)
