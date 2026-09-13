> 📖 **Navigation:** [← Previous: Part 18: Positive Definite & Semidefinite Matrices](./18_positive_definite_matrices.md) | [🏠 Index](./README.md) | [Next: Part 20: Moore-Penrose Pseudoinverse (A^+) →](./20_moore_penrose_pseudoinverse.md)

---

# PART 19 — SINGULAR VALUE DECOMPOSITION (SVD)

While eigendecomposition only exists for square matrices, **Singular Value Decomposition (SVD)** factorizes **ANY matrix** of arbitrary shape $m \times n$. SVD is the most powerful factorization in machine learning, powering PCA, Latent Semantic Analysis, recommender systems, and Low-Rank Adaptation (LoRA).

---

## 19.1 The Master SVD Equation: $A = U \Sigma V^T$

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

1. **$U \in \mathbb{R}^{m \times m}$ (Left Singular Vectors):**
   * An orthogonal matrix ($U^T U = I_m$).
   * Its columns $\mathbf{u}_i$ are the orthonormal eigenvectors of the square symmetric matrix $A A^T$.
   * Forms an orthonormal basis for the output space $\mathbb{R}^m$ (column space and left null space).
2. **$\Sigma \in \mathbb{R}^{m \times n}$ (Singular Values):**
   * A diagonal matrix containing non-negative numbers $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$ along its main diagonal, with zeros elsewhere.
   * Singular values measure the stretch factor along each principal direction:

$$
\sigma_i = \sqrt{\lambda_i(A^T A)} = \sqrt{\lambda_i(A A^T)}
$$

3. **$V \in \mathbb{R}^{n \times n}$ (Right Singular Vectors):**
   * An orthogonal matrix ($V^T V = I_n$).
   * Its columns $\mathbf{v}_i$ are the orthonormal eigenvectors of the square symmetric matrix $A^T A$.
   * Forms an orthonormal basis for the input space $\mathbb{R}^n$ (row space and null space).

---

## 19.2 Geometric Meaning: Rotate $\to$ Scale $\to$ Rotate

SVD reveals that **any linear transformation is composed of three elementary geometric actions**:

$$
\mathbf{x} \quad \xrightarrow{\quad V^T \text{ (Rotate/Align)} \quad} \quad \xrightarrow{\quad \Sigma \text{ (Stretch along Axes)} \quad} \quad \xrightarrow{\quad U \text{ (Rotate into Output Space)} \quad} \quad A\mathbf{x}
$$

```
     Input Sphere               V^T (Rotation)            Sigma (Stretch)             U (Final Rotation)
          y                           y                           y                           y
        ╭───╮                       ╭───╮                       ╭─────╮                       ╭───╮
        │ ● │        ──────►        │ ● │        ──────►        │  ●  │        ──────►        │ ╱ │
        ╰───╯                       ╰───╯                       ╰─────╯                       ╰───╯
     Unit Circle                 Aligned Axes                 Hyper-Ellipsoid            Final Output Ellipse
```

* Transformation $A$ maps the unit sphere in $\mathbb{R}^n$ into a hyper-ellipsoid in $\mathbb{R}^m$.
* The **semi-axes of the ellipsoid** have lengths equal to the singular values $\sigma_i$.
* The **directions of the principal axes** are the left singular vectors $\mathbf{u}_i$.

---

## 19.3 Full SVD vs. Compact SVD vs. Truncated SVD

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                                   THE THREE SVD VARIANTS                                  │
├────────────────────┬─────────────────────────────┬────────────────────────────────────────┤
│ Variant            │ Matrix Shapes (rank r)      │ Primary ML Application                 │
├────────────────────┼─────────────────────────────┼────────────────────────────────────────┤
│ 1. Full SVD        │ U (m x m), Σ (m x n), V^T(n x n) │ Subspace theoretical analysis     │
│ 2. Compact (Thin)  │ U_r (m x r), Σ_r (r x r), V_r^T (r x n) │ Exact lossless data storage│
│ 3. Truncated SVD   │ U_k (m x k), Σ_k (k x k), V_k^T (k x n) │ PCA, LoRA, Topic Modeling  │
└────────────────────┴─────────────────────────────┴────────────────────────────────────────┘
```

1. **Full SVD:** Includes all orthogonal basis vectors spanning the entire input and output ambient spaces.
2. **Compact SVD:** Discards the zero singular values and their corresponding columns in $U$ and $V$. It exactly reproduces $A$ without any loss of information:

$$
A = U_r \Sigma_r V_r^T
$$

3. **Truncated SVD (Rank-$k$ Approximation):** Keeps only the top $k < r$ largest singular values:

$$
A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T
$$

---

## 19.4 The Eckart-Young-Mirsky Low-Rank Approximation Theorem

> [!IMPORTANT]
> **The Optimal Compression Theorem:**
> For any rank $k < r$, the truncated SVD matrix $A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$ is the **provably optimal rank-$k$ approximation of $A$** under both Frobenius and Spectral norms:
>
> $$
> \min_{\text{rank}(B) \le k} \|A - B\|_F = \|A - A_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}
> $$
>
> $$
> \min_{\text{rank}(B) \le k} \|A - B\|_2 = \|A - A_k\|_2 = \sigma_{k+1}
> $$
>

The approximation error is simply the energy of the discarded singular values.

---

## 19.5 Small Hand-Computable Numerical Example

Compute the SVD of the rectangular rank-1 matrix:

$$
A = \begin{bmatrix} 3 & 4 \\ 0 & 0 \end{bmatrix} \in \mathbb{R}^{2 \times 2}
$$

### Step 1: Compute $A^T A$ to find Right Singular Vectors $V$ and $\Sigma$

$$
A^T A = \begin{bmatrix} 3 & 0 \\ 4 & 0 \end{bmatrix} \begin{bmatrix} 3 & 4 \\ 0 & 0 \end{bmatrix} =
\begin{bmatrix} 9 & 12 \\ 12 & 16 \end{bmatrix}
$$

Find eigenvalues of $A^T A$:

$$
\det(A^T A - \lambda I) = (9 - \lambda)(16 - \lambda) - 144 = \lambda^2 - 25\lambda + (144 - 144) = \lambda(\lambda - 25) = 0
$$

* $\lambda_1 = 25, \quad \lambda_2 = 0$.
* Singular values:

$$
\sigma_1 = \sqrt{25} = 5, \quad \sigma_2 = \sqrt{0} = 0
$$

$$
\Sigma = \begin{bmatrix} 5 & 0 \\ 0 & 0 \end{bmatrix}
$$

### Step 2: Find Right Singular Vectors $V$
* For $\lambda_1 = 25$:

$$
(A^T A - 25I)\mathbf{v}_1 = \begin{bmatrix} -16 & 12 \\ 12 & -9 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \mathbf{0} \implies -4v_1 + 3v_2 = 0 \implies \mathbf{v}_1 = \begin{bmatrix} 3/5 \\ 4/5 \end{bmatrix}
$$

* For $\lambda_2 = 0$:

$$
(A^T A)\mathbf{v}_2 = \begin{bmatrix} 9 & 12 \\ 12 & 16 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \mathbf{0} \implies 3v_1 + 4v_2 = 0 \implies \mathbf{v}_2 = \begin{bmatrix} -4/5 \\ 3/5 \end{bmatrix}
$$

$$
V = \begin{bmatrix} 3/5 & -4/5 \\ 4/5 & 3/5 \end{bmatrix} \implies V^T = \begin{bmatrix} 3/5 & 4/5 \\ -4/5 & 3/5 \end{bmatrix}
$$

### Step 3: Compute Left Singular Vectors $U$
For non-zero singular value $\sigma_1 = 5$, compute $\mathbf{u}_1 = \frac{1}{\sigma_1} A \mathbf{v}_1$:

$$
\mathbf{u}_1 = \frac{1}{5} \begin{bmatrix} 3 & 4 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} 3/5 \\ 4/5 \end{bmatrix} =
\frac{1}{5} \begin{bmatrix} 9/5 + 16/5 \\ 0 \end{bmatrix} = \frac{1}{5} \begin{bmatrix} 5 \\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
$$

Choose $\mathbf{u}_2$ to complete the orthonormal basis for $\mathbb{R}^2$: $\mathbf{u}_2 = [0, 1]^T$.

$$
U = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$

### Step 4: Verification ($A = U \Sigma V^T$)

$$
U \Sigma = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 5 & 0 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 5 & 0 \\ 0 & 0 \end{bmatrix}
$$

$$
(U \Sigma) V^T = \begin{bmatrix} 5 & 0 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} 3/5 & 4/5 \\ -4/5 & 3/5 \end{bmatrix} =
\begin{bmatrix} 5(3/5) & 5(4/5) \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 3 & 4 \\ 0 & 0 \end{bmatrix} = A
$$

✓ **Verified**.

---

## 19.6 SVD Connection to PCA

For a mean-centered dataset $X_c \in \mathbb{R}^{n \times d}$:

$$
X_c = U \Sigma V^T
$$

Compute the sample covariance matrix:

$$
\Sigma_{\text{cov}} = \frac{1}{n-1} X_c^T X_c = \frac{1}{n-1} (V \Sigma^T U^T)(U \Sigma V^T) = \frac{1}{n-1} V \Sigma^2 V^T = V \left( \frac{\Sigma^2}{n-1} \right) V^T
$$

* **The Takeaway:** The right singular vectors $V$ of centered data $X_c$ are **EXACTLY the principal component directions**!
* **The Variance Connection:** The covariance eigenvalues are $\lambda_i = \frac{\sigma_i^2}{n-1}$.
* **Why Scikit-Learn Uses SVD:** `sklearn.decomposition.PCA` never forms $X_c^T X_c$. It computes SVD directly on $X_c$, cutting condition number in half and avoiding catastrophic precision loss.

---

## 19.7 Why this matters in ML

1. **LoRA (Low-Rank Adaptation):** Fine-tuning foundation models with billions of parameters modifies weights by $\Delta W = B A$ where $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$ ($r \ll d$). SVD provides the theoretical backing for why low-rank updates capture nearly all expressive power.
2. **Latent Semantic Analysis (LSA):** SVD applied to term-document frequency matrices compresses words and documents into a shared conceptual semantic space.
3. **Collaborative Filtering:** Netflix-style recommendation models decompose sparse user-item rating matrices via truncated SVD to discover latent user preferences and movie genres.

---

## 19.8 Common mistakes

* **Confusing Singular Values with Eigenvalues:** Singular values $\sigma_i$ are **always real and non-negative** ($\sigma_i \ge 0$), whereas eigenvalues $\lambda_i$ can be negative or complex.
* **Forgetting that $V^T$ is Transposed in the Equation:** When using numerical packages (`np.linalg.svd`), the third returned argument is already $V^T$ (not $V$). Multiplying $U @ \Sigma @ V$ instead of $U @ \Sigma @ V^T$ is a common coding bug.

---

> 📖 **Navigation:** [← Previous: Part 18: Positive Definite & Semidefinite Matrices](./18_positive_definite_matrices.md) | [🏠 Index](./README.md) | [Next: Part 20: Moore-Penrose Pseudoinverse (A^+) →](./20_moore_penrose_pseudoinverse.md)
