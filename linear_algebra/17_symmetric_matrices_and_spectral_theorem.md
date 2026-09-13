> 📖 **Navigation:** [← Previous: Part 16: Diagonalization](./16_diagonalization.md) | [🏠 Index](./README.md) | [Next: Part 18: Positive Definite & Semidefinite Matrices →](./18_positive_definite_matrices.md)

---

# PART 17 — SYMMETRIC MATRICES & THE SPECTRAL THEOREM

Symmetric matrices ($A = A^T$) are the crowns of linear algebra. In machine learning, sample covariance matrices, loss function Hessians, kernel Gram matrices, and graph Laplacians are all symmetric matrices.

---

## 17.1 The Miracle of Symmetric Matrices

When a real matrix is symmetric ($A = A^T$), three remarkable mathematical guarantees hold:

1. **All Eigenvalues are Strictly Real:** Even if the characteristic polynomial has complex roots in general, for a symmetric matrix, **every eigenvalue is a real number** ($\lambda_i \in \mathbb{R}$).
2. **Eigenvectors Are Mutually Orthogonal:** Eigenvectors corresponding to distinct eigenvalues are **perpendicular to each other** ($\mathbf{q}_i \cdot \mathbf{q}_j = 0$ for $i \neq j$).
3. **Always Diagonalizable:** Symmetric matrices are **never defective**—they always possess a full set of $n$ mutually orthogonal eigenvectors, even when eigenvalues are repeated!

---

## 17.2 The Spectral Theorem (Orthogonal Diagonalization)

> [!IMPORTANT]
> **The Spectral Theorem:**
> Every real symmetric matrix $A \in \mathbb{R}^{n \times n}$ can be **orthogonally diagonalized**:
> $$
> A = Q \Lambda Q^T
> $$
> where $Q = [\mathbf{q}_1 \dots \mathbf{q}_n]$ is an **orthogonal matrix** of normalized eigenvectors ($Q^T Q = I_n$), and $\Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$ is the diagonal matrix of real eigenvalues.

### Geometric Interpretation: Rotate $\to$ Stretch $\to$ Rotate Back
$$
A\mathbf{x} = Q \Lambda Q^T \mathbf{x}
$$
1. **$Q^T$ (Rotate):** Rotates the coordinate system to align the eigenvector axes with the standard coordinate axes.
2. **$\Lambda$ (Stretch):** Stretches or compresses each coordinate independently by its eigenvalue $\lambda_i$.
3. **$Q$ (Rotate Back):** Rotates space back to the original orientation.

---

## 17.3 Spectral Decomposition (Sum of Rank-1 Outer Products)

Multiplying out $A = Q \Lambda Q^T$ column-by-row expands the matrix into a sum of rank-1 projections:

$$
A = \sum_{i=1}^{n} \lambda_i \mathbf{q}_i \mathbf{q}_i^T = \lambda_1 \mathbf{q}_1 \mathbf{q}_1^T + \lambda_2 \mathbf{q}_2 \mathbf{q}_2^T + \dots + \lambda_n \mathbf{q}_n \mathbf{q}_n^T
$$

* Each term $P_i = \mathbf{q}_i \mathbf{q}_i^T$ is a **rank-1 projection matrix** that drops space onto eigenvector axis $\mathbf{q}_i$.
* The symmetric matrix $A$ is literally a weighted sum of orthogonal projections, weighted by its eigenvalues!

---

## 17.4 Complete Worked Numerical Example ($2 \times 2$)

Orthogonally diagonalize the symmetric matrix:
$$
A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}
$$

### Step 1: Solve Characteristic Equation
$$
\det(A - \lambda I) = (3 - \lambda)^2 - (1)(1) = \lambda^2 - 6\lambda + 8 = 0
$$
Factor the quadratic:
$$
(\lambda - 4)(\lambda - 2) = 0 \implies \lambda_1 = 4, \quad \lambda_2 = 2
$$
Both eigenvalues are real numbers $\checkmark$.

---

### Step 2: Find Eigenvectors & Verify Orthogonality
* **For $\lambda_1 = 4$:**
  $$
  (A - 4I)\mathbf{v}_1 = \begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \mathbf{0} \implies -v_1 + v_2 = 0 \implies \mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
  $$
* **For $\lambda_2 = 2$:**
  $$
  (A - 2I)\mathbf{v}_2 = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \mathbf{0} \implies v_1 + v_2 = 0 \implies \mathbf{v}_2 = \begin{bmatrix} -1 \\ 1 \end{bmatrix}
  $$

#### Check Orthogonality:
$$
\mathbf{v}_1 \cdot \mathbf{v}_2 = (1)(-1) + (1)(1) = -1 + 1 = 0 \quad \checkmark
$$
The eigenvectors are strictly perpendicular!

---

### Step 3: Normalize to Unit Length to Form $Q$
* $\|\mathbf{v}_1\|_2 = \sqrt{1^2 + 1^2} = \sqrt{2} \implies \mathbf{q}_1 = \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix}$
* $\|\mathbf{v}_2\|_2 = \sqrt{(-1)^2 + 1^2} = \sqrt{2} \implies \mathbf{q}_2 = \begin{bmatrix} -1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix}$

$$
Q = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}, \quad \Lambda = \begin{bmatrix} 4 & 0 \\ 0 & 2 \end{bmatrix}
$$

---

### Step 4: Verify $A = Q \Lambda Q^T$
$$
Q \Lambda = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 4 & 0 \\ 0 & 2 \end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix} 4 & -2 \\ 4 & 2 \end{bmatrix}
$$

$$
(Q \Lambda) Q^T = \left(\frac{1}{\sqrt{2}} \begin{bmatrix} 4 & -2 \\ 4 & 2 \end{bmatrix}\right) \left(\frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix}\right) =
\frac{1}{2} \begin{bmatrix} 4(1) - 2(-1) & 4(1) - 2(1) \\ 4(1) + 2(-1) & 4(1) + 2(1) \end{bmatrix} =
\frac{1}{2} \begin{bmatrix} 6 & 2 \\ 2 & 6 \end{bmatrix} = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix} = A \quad \checkmark
$$

---

### Step 5: Verify Spectral Decomposition
$$
\lambda_1 \mathbf{q}_1 \mathbf{q}_1^T = 4 \left(\frac{1}{2} \begin{bmatrix} 1 \\ 1 \end{bmatrix} [1, 1]\right) = 2 \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 2 \\ 2 & 2 \end{bmatrix}
$$
$$
\lambda_2 \mathbf{q}_2 \mathbf{q}_2^T = 2 \left(\frac{1}{2} \begin{bmatrix} -1 \\ 1 \end{bmatrix} [-1, 1]\right) = 1 \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}
$$

Summing both rank-1 projection matrices:
$$
\lambda_1 \mathbf{q}_1 \mathbf{q}_1^T + \lambda_2 \mathbf{q}_2 \mathbf{q}_2^T = \begin{bmatrix} 2 + 1 & 2 - 1 \\ 2 - 1 & 2 + 1 \end{bmatrix} = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix} = A \quad \checkmark
$$

---

## 17.5 Why this matters in ML

1. **The Mathematical Guarantee Behind PCA:** The sample covariance matrix $\Sigma = \frac{1}{n-1} X_c^T X_c$ is symmetric ($\Sigma = \Sigma^T$). By the Spectral Theorem, its eigenvalues are guaranteed to be real and its principal component directions are **guaranteed to be mutually perpendicular**.
2. **Hessian Curvature in Convex Optimization:** The matrix of second partial derivatives (the Hessian $H = \nabla^2 \mathcal{L}$) is symmetric by Schwarz's theorem ($\frac{\partial^2 \mathcal{L}}{\partial w_i \partial w_j} = \frac{\partial^2 \mathcal{L}}{\partial w_j \partial w_i}$). The eigenvectors of $H$ give the orthogonal axes of curvature of the loss surface.

---

## 17.6 Common mistakes

* **Assuming All Matrices Have Orthogonal Eigenvectors:** Only symmetric matrices (and normal matrices where $A^T A = A A^T$) are guaranteed to have orthogonal eigenvectors. General matrices can have eigenvectors separated by non-right angles.
* **Forgetting to Normalize Eigenvectors in $Q$:** If you leave eigenvectors as $[1, 1]^T$ without dividing by $\sqrt{2}$, $Q$ will not satisfy $Q^T Q = I$, breaking $Q^{-1} = Q^T$.

---

> 📖 **Navigation:** [← Previous: Part 16: Diagonalization](./16_diagonalization.md) | [🏠 Index](./README.md) | [Next: Part 18: Positive Definite & Semidefinite Matrices →](./18_positive_definite_matrices.md)
