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

## 10.4 The 4 Fundamental Subspaces from SVD (Strang's Master Picture)

SVD provides explicit orthonormal bases for all **Four Fundamental Subspaces** of any matrix $A \in \mathbb{R}^{m \times n}$ with rank $r$:

```
                         THE SVD FOUR-SUBSPACE CONNECTION
       Input Space R^n                                 Output Space R^m
  ┌─────────────────────────┐                     ┌─────────────────────────┐
  │  Row Space Row(A)       │                     │  Column Space Col(A)    │
  │  span{v_1, ..., v_r}    │  ───── A v_i ─────► │  span{u_1, ..., u_r}    │
  │  Dimension: r           │     = σ_i u_i       │  Dimension: r           │
  ├─────────────────────────┤                     ├─────────────────────────┤
  │  Nullspace Null(A)      │                     │  Left Null Null(A^T)    │
  │  span{v_{r+1}, ..., v_n}│  ───── A v_i ─────► │  span{u_{r+1}, ..., u_m}│
  │  Dimension: n - r       │       = 0           │  Dimension: m - r       │
  └─────────────────────────┘                     └─────────────────────────┘
```

1. **Column Space $\text{Col}(A)$:** First $r$ left singular vectors $\{\mathbf{u}_1, \dots, \mathbf{u}_r\}$.
2. **Left Nullspace $\text{Null}(A^T)$:** Remaining $(m - r)$ left singular vectors $\{\mathbf{u}_{r+1}, \dots, \mathbf{u}_m\}$.
3. **Row Space $\text{Row}(A)$:** First $r$ right singular vectors $\{\mathbf{v}_1, \dots, \mathbf{v}_r\}$.
4. **Nullspace $\text{Null}(A)$:** Remaining $(n - r)$ right singular vectors $\{\mathbf{v}_{r+1}, \dots, \mathbf{v}_n\}$.

---

## 10.5 The Eckart-Young-Mirsky Theorem (Optimal Low-Rank Approximation)

The **Eckart-Young-Mirsky Theorem** is the mathematical foundation of data compression, dimensionality reduction, and Low-Rank Adaptation (LoRA) in LLMs.

### Theorem Statement:
Let $A = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T$ be the SVD of $A$. The **truncated SVD of rank $k < r$**:

$$
A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T
$$

is the **provably optimal rank-$k$ approximation** of matrix $A$ under both the Frobenius Norm and the Spectral Norm:

$$
\min_{\text{rank}(B) \le k} \|A - B\|_F = \|A - A_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}
$$

$$
\min_{\text{rank}(B) \le k} \|A - B\|_2 = \|A - A_k\|_2 = \sigma_{k+1}
$$

```
                  LOW-RANK ADAPTATION (LoRA) VIA SVD FACTORIZATION
          Weight Update ΔW (d x k)               Factored Rank-r Matrices
             ┌               ┐                      ┌   ┐
             │               │          ≈           │ B │  ×  ┌───────────┐
             │               │                      │   │     │     A     │
             └               ┘                      └───┘     └───────────┘
              (d x k params)                     (d x r)         (r x k)
           Full Rank Parameterization          Parameters reduced by >90%!
```

* **ML Applications:**
  * **LoRA (Low-Rank Adaptation):** Freezes pre-trained LLM weights $W_0$ and trains low-rank factor matrices $\Delta W = B A$ where $r \ll d$.
  * **Latent Semantic Analysis (LSA):** Compressing term-document matrices for NLP topic modeling.
  * **Image Compression:** Keeping top $k$ singular values reconstructs $>95\%$ of visual information with a fraction of storage.

---

## 10.6 Condition Number ($\kappa(A)$) & Optimization Geometry

The **Condition Number** $\kappa(A)$ measures the sensitivity of matrix operations to numerical perturbations:

$$
\kappa(A) = \|A\|_2 \|A^{-1}\|_2 = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)} = \frac{\sigma_1}{\sigma_r} \ge 1.0
$$

```
         WELL-CONDITIONED (κ ≈ 1)                 ILL-CONDITIONED (κ >> 1)
           Isotropic Contours                       Highly Elongated Ravine
                  x2                                       x2
                  │                                        │
                ╭─┼─╮                                    ╭─┴────────────────╮
                │ │ │                                    │                  │
               ─┼─●─┼─► x1                              ─┼────────●─────────┼─► x1
                │ │ │                                    │                  │
                ╰─┼─╯                                    ╰─┬────────────────╯
                  │                                        │
         (Fast SGD Convergence)                    (Severe Oscillations & Zig-Zagging)
```

* **Impact on Gradient Descent:** For quadratic optimization $\mathcal{L}(\mathbf{w}) = \frac{1}{2}\mathbf{w}^T H \mathbf{w}$, the convergence rate of Gradient Descent depends directly on the condition number of the Hessian $H$:

$$
\|\mathbf{w}^{(t)} - \mathbf{w}^*\| \le \left( \frac{\kappa(H) - 1}{\kappa(H) + 1} \right)^t \|\mathbf{w}^{(0)} - \mathbf{w}^*\|
$$

* If $\kappa(H) \approx 1$, convergence is instantaneous.
* If $\kappa(H) = 10,000$ (ill-conditioned), convergence slows to a crawl, motivating **Adam**, **Momentum**, and **Batch Normalization** (which precondition the feature space to make $\kappa \approx 1$).

---

## 10.7 Full SVD vs. Compact SVD vs. Truncated SVD

| SVD Variant | Factorization Format | Shapes ($m \ge n$, rank $r$) | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **Full SVD** | $A = U \Sigma V^T$ | $U \in \mathbb{R}^{m \times m}, \Sigma \in \mathbb{R}^{m \times n}, V \in \mathbb{R}^{n \times n}$ | Theoretical proofs, complete subspace geometry. |
| **Compact (Thin) SVD** | $A = U_r \Sigma_r V_r^T$ | $U_r \in \mathbb{R}^{m \times r}, \Sigma_r \in \mathbb{R}^{r \times r}, V_r \in \mathbb{R}^{n \times r}$ | Exact matrix reconstruction without storing zero singular values. |
| **Truncated SVD** | $A \approx U_k \Sigma_k V_k^T$ | $U_k \in \mathbb{R}^{m \times k}, \Sigma_k \in \mathbb{R}^{k \times k}, V_k \in \mathbb{R}^{n \times k}$ ($k < r$) | **LoRA, PCA, TruncatedSVD in Scikit-Learn**, image compression. |

---

> 📖 **Navigation:** [← Previous: Part 10: Complete PCA Walkthrough from Scratch](./10_pca_scratch_walkthrough.md) | [🏠 Index](./README.md) | [Next: Part 12: Orthogonality & Orthonormal Bases →](./12_orthogonality_and_bases.md)
