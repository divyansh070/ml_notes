> 📖 **Navigation:** [← Previous: Part 10: Complete PCA Walkthrough from Scratch](./10_pca_scratch_walkthrough.md) | [🏠 Index](./README.md) | [Next: Part 12: Orthogonality & Orthonormal Bases →](./12_orthogonality_and_bases.md)

---

# PART 10 — SINGULAR VALUE DECOMPOSITION (SVD)

---

## 10.1 The SVD Equation: $A = U \Sigma V^T$

While eigendecomposition only works on **square** matrices ($N \times N$), **Singular Value Decomposition (SVD)** factorizes **ANY matrix** of shape $(M \times N)$ into three constituent geometric matrices:

$$
A = U \Sigma V^T
$$

```
      A (M x N)              U (M x M)               Sigma (M x N)             V^T (N x N)
   ┌             ┐       ┌                ┐       ┌                 ┐       ┌                ┐
   │             │   =   │ Left Singular  │   ×   │ Singular Values │   ×   │ Right Singular │
   │   DATA      │       │ Vectors        │       │ (Stretch scale) │       │ Vectors        │
   │             │       │ (Orthonormal)  │       │ (Diagonal)      │       │ (Orthonormal)  │
   └             ┘       └                ┘       └                 ┘       └                ┘
```

1. **$U$ (Left Singular Vectors, $M \times M$):** Orthonormal eigenvectors of $A A^T$ ($U^T U = I$). Represents column space basis directions.
2. **$\Sigma$ (Singular Values, $M \times N$):** Diagonal matrix of non-negative singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$ in descending order. Represents scaling/stretch factors along each principal axis.
3. **$V^T$ (Right Singular Vectors Transpose, $N \times N$):** Transpose of the orthonormal eigenvectors of $A^T A$ ($V^T V = I$). Represents row space / principal feature directions.

---

## 10.2 Geometric Interpretation of $U, \Sigma, V^T$

Any linear transformation can be decomposed into three fundamental geometric steps:

$$
\mathbf{x} \quad \xrightarrow{\quad V^T \text{ (Rotate/Reflect)} \quad} \quad \xrightarrow{\quad \Sigma \text{ (Scale axes)} \quad} \quad \xrightarrow{\quad U \text{ (Second Rotation)} \quad} \quad A\mathbf{x}
$$

---

## 10.3 Relationship Between SVD and PCA

For a mean-centered data matrix $X_c$ ($N \times p$):
1. **Covariance Matrix:** $\Sigma_{\text{cov}} = \frac{1}{n-1} X_c^T X_c$.
2. Substitute SVD factorization $X_c = U \Sigma V^T$:

$$
X_c^T X_c = (U \Sigma V^T)^T (U \Sigma V^T) = (V \Sigma^T U^T)(U \Sigma V^T) = V \Sigma^T (U^T U) \Sigma V^T = V \Sigma^2 V^T
$$

*(Since $U$ is orthonormal, $U^T U = I$).*

3. **Core Relationships to Memorize:**
   * The **Right Singular Vectors ($V$)** of $X_c$ are **identical to the Principal Component Directions (Eigenvectors of $\Sigma_{\text{cov}}$)**.
   * The **Singular Values ($\sigma_i$)** relate directly to the Eigenvalues ($\lambda_i$):

$$
\lambda_i = \frac{\sigma_i^2}{n - 1} \quad \iff \quad \sigma_i = \sqrt{(n - 1)\lambda_i}
$$

   * For the unnormalized matrix $A^T A$, singular values satisfy $\sigma_i = \sqrt{\lambda_i(A^T A)}$.
   * *Why modern libraries use SVD instead of Eigendecomposition:* Scikit-Learn's `PCA` uses SVD internally because computing $X_c = U \Sigma V^T$ directly avoids explicitly forming the $X_c^T X_c$ matrix, offering higher numerical precision and avoiding the squaring of condition numbers.

---

> 📖 **Navigation:** [← Previous: Part 10: Complete PCA Walkthrough from Scratch](./10_pca_scratch_walkthrough.md) | [🏠 Index](./README.md) | [Next: Part 12: Orthogonality & Orthonormal Bases →](./12_orthogonality_and_bases.md)
