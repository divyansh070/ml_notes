> 📖 **Navigation:** [← Previous: Part 09: Covariance Matrix Complete Calculation](./09_covariance_matrix.md) | [🏠 Index](./README.md) | [Next: Part 11: Singular Value Decomposition (SVD) →](./11_singular_value_decomposition.md)

---

# PART 10 — COMPLETE PCA WALKTHROUGH FROM SCRATCH

This module provides a complete numerical 9-step hand calculation demonstrating how PCA compresses a 2D dataset down into 1D, projects the points, and reconstructs them while measuring information lost.

---

## 10.1 The 9-Step Numerical PCA Algorithm

We have a 2D dataset with $n = 4$ observations and $d = 2$ features:
$$
X = \begin{bmatrix} 2 & 4 \\ 3 & 6 \\ 5 & 8 \\ 6 & 10 \end{bmatrix}
$$

```
                           THE COMPLETE PCA PIPELINE
    Raw Data (X) ──► Center Data (X_c) ──► Covariance (Σ) ──► Eigendecomposition (λ, v)
                                                                       │
    Reconstruction (X̂) ◄── 1D Projection (z) ◄── Select Top-1 Vector ──┘
```

---

### Step 1: Compute Feature Means
$$
\bar{x}_1 = \frac{2 + 3 + 5 + 6}{4} = \frac{16}{4} = 4.0, \quad \bar{x}_2 = \frac{4 + 6 + 8 + 10}{4} = \frac{28}{4} = 7.0
$$

---

### Step 2: Center the Data ($X_c = X - \bar{X}$)
Subtract the feature means from each column:
$$
X_c = \begin{bmatrix} 2 - 4 & 4 - 7 \\ 3 - 4 & 6 - 7 \\ 5 - 4 & 8 - 7 \\ 6 - 4 & 10 - 7 \end{bmatrix} = \begin{bmatrix} -2 & -3 \\ -1 & -1 \\ 1 & 1 \\ 2 & 3 \end{bmatrix}
$$

---

### Step 3: Compute Sample Covariance Matrix ($\Sigma = \frac{1}{n-1} X_c^T X_c$)
$$
X_c^T X_c = \begin{bmatrix} -2 & -1 & 1 & 2 \\ -3 & -1 & 1 & 3 \end{bmatrix} \begin{bmatrix} -2 & -3 \\ -1 & -1 \\ 1 & 1 \\ 2 & 3 \end{bmatrix} = \begin{bmatrix} 10 & 14 \\ 14 & 20 \end{bmatrix}
$$

Divide by $n - 1 = 3$:
$$
\Sigma = \frac{1}{3} \begin{bmatrix} 10 & 14 \\ 14 & 20 \end{bmatrix} \approx \begin{bmatrix} 3.33 & 4.67 \\ 4.67 & 6.67 \end{bmatrix}
$$

---

### Step 4: Solve for Eigenvalues ($\det(X_c^T X_c - \lambda I) = 0$)
For clean calculation, solve on:

$$
X_c^T X_c = \begin{bmatrix} 10 & 14 \\ 14 & 20 \end{bmatrix}
$$

$$
(10 - \lambda)(20 - \lambda) - (14 \times 14) = \lambda^2 - 30\lambda + (200 - 196) = \lambda^2 - 30\lambda + 4 = 0
$$
$$
\lambda_1 \approx 29.87, \quad \lambda_2 \approx 0.13
$$

Divide by $n-1 = 3$ to get covariance eigenvalues:
$$
\lambda_1(\Sigma) \approx 9.96, \quad \lambda_2(\Sigma) \approx 0.04
$$

---

### Step 5: Find the Dominant Eigenvector ($\mathbf{v}_1$)
Substitute $\lambda_1 = 29.87$ into $(X_c^T X_c - \lambda I)\mathbf{v} = \mathbf{0}$:
$$
\begin{bmatrix} 10 - 29.87 & 14 \\ 14 & 20 - 29.87 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \implies -19.87 v_1 + 14 v_2 = 0 \implies v_2 \approx 1.42 v_1
$$
Normalize to unit length:
$$
\mathbf{v}_1 = \begin{bmatrix} 0.576 \\ 0.817 \end{bmatrix}
$$

---

### Step 6: Compute Explained Variance Ratio
$$
\text{Explained Variance} = \frac{\lambda_1}{\lambda_1 + \lambda_2} = \frac{29.87}{29.87 + 0.13} = \frac{29.87}{30.00} = \mathbf{99.57\%}
$$
*(Keeping just this 1 principal component preserves $99.57\%$ of all information in the dataset!).*

---

### Step 7: Project Data onto 1D Subspace ($z = X_c \mathbf{v}_1$)
$$
z_1 = \begin{bmatrix} -2 & -3 \end{bmatrix} \begin{bmatrix} 0.576 \\ 0.817 \end{bmatrix} = -2(0.576) - 3(0.817) = \mathbf{-3.60}
$$
$$
z_2 = \begin{bmatrix} -1 & -1 \end{bmatrix} \begin{bmatrix} 0.576 \\ 0.817 \end{bmatrix} = -1(0.576) - 1(0.817) = \mathbf{-1.39}
$$
$$
z_3 = \begin{bmatrix} 1 & 1 \end{bmatrix} \begin{bmatrix} 0.576 \\ 0.817 \end{bmatrix} = 1(0.576) + 1(0.817) = \mathbf{+1.39}
$$
$$
z_4 = \begin{bmatrix} 2 & 3 \end{bmatrix} \begin{bmatrix} 0.576 \\ 0.817 \end{bmatrix} = 2(0.576) + 3(0.817) = \mathbf{+3.60}
$$

**1D Compressed Representation:** $\mathbf{z} = [-3.60, -1.39, 1.39, 3.60]^T$.

---

### Step 8: Reconstruct Data Back to 2D ($\hat{X} = \mathbf{z} \mathbf{v}_1^T + \bar{X}$)
For sample 1 ($z_1 = -3.60$):
$$
\hat{\mathbf{x}}_1 = -3.60 \begin{bmatrix} 0.576 \\ 0.817 \end{bmatrix} + \begin{bmatrix} 4.0 \\ 7.0 \end{bmatrix} = \begin{bmatrix} -2.07 \\ -2.94 \end{bmatrix} + \begin{bmatrix} 4.0 \\ 7.0 \end{bmatrix} = \begin{bmatrix} 1.93 \\ 4.06 \end{bmatrix}
$$
*(Original point was $[2.0, 4.0]^T$; reconstruction is $[1.93, 4.06]^T$ — nearly identical!).*

---

### Step 9: What Information Did PCA Discard?

```
                       PCA ORTHOGONAL RESIDUAL DISCARD
                 x2                                          x2
                 │                                           │        ● Original
                 │                                           │       ╱│ (Error distance = λ2)
                 │                                           │      ●─┴─── Reconstructed Point
                 └──┴────────────► x1                        └──┴────────────► x1
```

* **Reconstruction Error:** The residual vector $\mathbf{e} = \mathbf{x} - \hat{\mathbf{x}}$ is strictly orthogonal to $\mathbf{v}_1$.
* The sum of squared reconstruction errors across all points is equal to the discarded eigenvalue: $\sum \|\mathbf{x}_i - \hat{\mathbf{x}}_i\|^2 = \lambda_2 = 0.13$ ($0.43\%$ of total variance).

---

> 📖 **Navigation:** [← Previous: Part 09: Covariance Matrix Complete Calculation](./09_covariance_matrix.md) | [🏠 Index](./README.md) | [Next: Part 11: Singular Value Decomposition (SVD) →](./11_singular_value_decomposition.md)
