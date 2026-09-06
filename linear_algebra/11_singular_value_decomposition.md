> 📖 **Navigation:** [← Previous: Part 10: Complete PCA Walkthrough from Scratch](./10_pca_scratch_walkthrough.md) | [🏠 Index](./README.md) | [Next: Part 12: Orthogonality & Orthonormal Bases →](./12_orthogonality_and_bases.md)

---

# PART 11 — SINGULAR VALUE DECOMPOSITION (SVD)

While Eigendecomposition only applies to **square matrices** ($n \times n$), **Singular Value Decomposition (SVD)** factorizes **ANY matrix** of arbitrary shape $(m \times n)$.

---

## 11.1 The Fundamental SVD Equation: $A = U \Sigma V^T$

Every real matrix $A \in \mathbb{R}^{m \times n}$ can be factored into:

$$
A = U \Sigma V^T
$$

```
      A (m x n)              U (m x m)               Sigma (m x n)             V^T (n x n)
   ┌             ┐       ┌                ┐       ┌                 ┐       ┌                ┐
   │             │   =   │ Left Singular  │   ×   │ Singular Values │   ×   │ Right Singular │
   │   DATA      │       │ Vectors        │       │ (Stretch scale) │       │ Vectors        │
   │             │       │ (Orthonormal)  │       │ (Diagonal)      │       │ (Orthonormal)  │
   └             ┘       └                ┘       └                 ┘       └                ┘
```

1. **$U \in \mathbb{R}^{m \times m}$ (Left Singular Vectors):** Orthonormal eigenvectors of $A A^T$ ($U^T U = I$). Basis for output column space.
2. **$\Sigma \in \mathbb{R}^{m \times n}$ (Singular Values):** Diagonal matrix of non-negative stretch factors sorted in descending order: $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$.
3. **$V \in \mathbb{R}^{n \times n}$ (Right Singular Vectors):** Orthonormal eigenvectors of $A^T A$ ($V^T V = I$). Basis for input feature directions.

---

## 11.2 Geometric Sequence: Rotate $\to$ Scale $\to$ Rotate

SVD states that **any linear transformation consists of three fundamental geometric steps**:

$$
\mathbf{x} \quad \xrightarrow{\quad V^T \text{ (Rotate/Align with Principal Axes)} \quad} \quad \xrightarrow{\quad \Sigma \text{ (Stretch along Axes)} \quad} \quad \xrightarrow{\quad U \text{ (Rotate into Output Space)} \quad} \quad A\mathbf{x}
$$

```
     Input Sphere               V^T (Rotation)            Sigma (Stretch)             U (Final Rotation)
          y                           y                           y                           y
        ╭───╮                       ╭───╮                       ╭─────╮                       ╭───╮
        │ ● │        ──────►        │ ● │        ──────►        │  ●  │        ──────►        │ ╱ │
        ╰───╯                       ╰───╯                       ╰─────╯                       ╰───╯
     Unit Circle                 Aligned Axes                 Hyper-Ellipsoid            Final Output Ellipse
```

---

## 11.3 Mathematical Connection to Eigendecomposition & PCA

1. **Relation to $A^T A$ and $A A^T$:**
   $$
   A^T A = (U \Sigma V^T)^T (U \Sigma V^T) = V \Sigma^T (U^T U) \Sigma V^T = V \Sigma^2 V^T
   $$
   $$
   A A^T = (U \Sigma V^T)(U \Sigma V^T)^T = U \Sigma (V^T V) \Sigma^T U^T = U \Sigma^2 U^T
   $$
   * Singular values are the square roots of eigenvalues:
     $$
     \sigma_i = \sqrt{\lambda_i(A^T A)} = \sqrt{\lambda_i(A A^T)}
     $$
2. **Relation to PCA:**
   * For centered data $X_c = U \Sigma V^T$, the **right singular vectors $V$ are EXACTLY the principal component directions** (eigenvectors of covariance $\Sigma_{\text{cov}} = \frac{1}{n-1} X_c^T X_c$).
   * Modern libraries (`sklearn.decomposition.PCA`) compute PCA using SVD on $X_c$ directly, avoiding explicit computation of $X_c^T X_c$ and preventing condition number squaring.

---

## 11.4 Simple Numerical Hand Calculation Example

Let:

$$
A = \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix}
$$

1. Compute $A^T A$:

$$
A^T A = \begin{bmatrix} 9 & 0 \\ 0 & 4 \end{bmatrix} \implies \lambda_1 = 9, \lambda_2 = 4
$$
2. Singular values: $\sigma_1 = \sqrt{9} = 3, \quad \sigma_2 = \sqrt{4} = 2$.
3. $$
\Sigma = \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix}, \quad U = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$


---

## 11.5 Full SVD vs. Compact SVD vs. Truncated SVD

| Variant | Equation Format | Dimensions ($m \ge n$, rank $r$) | Primary Application |
| :--- | :--- | :--- | :--- |
| **Full SVD** | $A = U \Sigma V^T$ | $U (m \times m), \Sigma (m \times n), V (n \times n)$ | Subspace theoretical analysis |
| **Compact (Thin) SVD** | $A = U_r \Sigma_r V_r^T$ | $U_r (m \times r), \Sigma_r (r \times r), V_r (n \times r)$ | Exact loss-free compression |
| **Truncated SVD** | $A \approx U_k \Sigma_k V_k^T$ | $U_k (m \times k), \Sigma_k (k \times k), V_k (n \times k)$ | **PCA, LoRA, LSA NLP topic modeling** |

---

## Advanced / Optional — Do not study until Core track is complete

### A.1 The Four Fundamental Subspaces via SVD
* Column Space $\text{Col}(A) = \text{span}\{\mathbf{u}_1, \dots, \mathbf{u}_r\}$
* Left Nullspace $\text{Null}(A^T) = \text{span}\{\mathbf{u}_{r+1}, \dots, \mathbf{u}_m\}$
* Row Space $\text{Row}(A) = \text{span}\{\mathbf{v}_1, \dots, \mathbf{v}_r\}$
* Nullspace $\text{Null}(A) = \text{span}\{\mathbf{v}_{r+1}, \dots, \mathbf{v}_n\}$

### A.2 The Eckart-Young-Mirsky Theorem (Optimal Rank-k Approximation)
The rank-$k$ truncated SVD $A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$ is the provably optimal rank-$k$ approximation of $A$ under both Frobenius and Spectral norms:
$$
\min_{\text{rank}(B) \le k} \|A - B\|_F = \|A - A_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}
$$
$$
\min_{\text{rank}(B) \le k} \|A - B\|_2 = \|A - A_k\|_2 = \sigma_{k+1}
$$

### A.3 Low-Rank Adaptation (LoRA) in Large Language Models
LoRA factorizes dense weight updates $\Delta W \in \mathbb{R}^{d \times k}$ into low-rank factor matrices $\Delta W = B A$ ($B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$ with $r \ll d$), cutting fine-tuning memory by $>90\%$.

### A.4 Condition Number ($\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}}$) & Optimization Curvature
In gradient descent, ill-conditioned Hessians ($\kappa(H) \gg 1$) create narrow elongated ravines, causing SGD to oscillate severely across ravines rather than moving toward the minimum.

---

> 📖 **Navigation:** [← Previous: Part 10: Complete PCA Walkthrough from Scratch](./10_pca_scratch_walkthrough.md) | [🏠 Index](./README.md) | [Next: Part 12: Orthogonality & Orthonormal Bases →](./12_orthogonality_and_bases.md)
