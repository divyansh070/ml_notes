> 📖 **Navigation:** [← Previous: Part 22: Four Fundamental Subspaces (Strang's Big Picture)](./22_four_fundamental_subspaces.md) | [🏠 Index](./README.md) | [Next: Part 24: Positive Definite & Positive Semidefinite Matrices →](./24_positive_definite_matrices.md)

---

# PART 23 — GRAM-SCHMIDT ORTHOGONALIZATION & QR DECOMPOSITION

Orthogonal bases are the numerical bedrock of scientific computing and machine learning. When vectors are mutually orthogonal, computations become decoupled, rounding errors are bounded, and matrix inversion becomes as simple as a transpose.

---

## 23.1 Why We Need Orthogonal Bases

When basis vectors $\mathbf{q}_1, \mathbf{q}_2, \dots, \mathbf{q}_k$ are **orthonormal** ($\mathbf{q}_i \cdot \mathbf{q}_j = 0$ for $i \neq j$ and $\|\mathbf{q}_i\|_2 = 1$):

1. **Instant Coordinates (No Matrix Inversion):**
   Expanding any vector $\mathbf{x}$ in an orthonormal basis requires only dot products:
   $$
   \mathbf{x} = c_1 \mathbf{q}_1 + c_2 \mathbf{q}_2 + \dots + c_k \mathbf{q}_k \quad \text{where } c_i = \mathbf{x} \cdot \mathbf{q}_i
   $$
2. **Trivial Matrix Inversion:**
   Any square orthogonal matrix $Q = [\mathbf{q}_1, \dots, \mathbf{q}_n]$ satisfies:
   $$
   Q^T Q = I \implies Q^{-1} = Q^T
   $$
3. **Length and Angle Preservation:**
   Orthogonal matrices act as rigid rotations and reflections:
   $$
   \|Q\mathbf{x}\|_2 = \|\mathbf{x}\|_2, \qquad (Q\mathbf{x})^T (Q\mathbf{y}) = \mathbf{x}^T \mathbf{y}
   $$

---

## 23.2 Gram-Schmidt Orthogonalization Process

The **Gram-Schmidt Process** takes a linearly independent set of vectors $\mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_k$ and turns them into an orthonormal set $\mathbf{q}_1, \mathbf{q}_2, \dots, \mathbf{q}_k$ spanning the exact same subspace.

```
        VECTOR 1: Keep direction            VECTOR 2: Subtract projection onto q1
                   y                                           y
                   │     a1 = u1                               │     a2
                   │    ╱                                      │    ╱│
                 1 ┼───●                                     1 ┼───● │  u2 = a2 - proj_u1(a2)
                   │  ╱                                        │   │ └──► (Orthogonal to u1!)
                   └──┴─────► x                                └───┴────► x
                      1                                            1
```

### The Three-Step Recipe:
1. **First vector:**
   $$
   \mathbf{u}_1 = \mathbf{a}_1, \qquad \mathbf{q}_1 = \frac{\mathbf{u}_1}{\|\mathbf{u}_1\|_2}
   $$
2. **Second vector:**
   Subtract the projection of $\mathbf{a}_2$ onto $\mathbf{u}_1$:
   $$
   \mathbf{u}_2 = \mathbf{a}_2 - (\mathbf{a}_2 \cdot \mathbf{q}_1)\mathbf{q}_1, \qquad \mathbf{q}_2 = \frac{\mathbf{u}_2}{\|\mathbf{u}_2\|_2}
   $$
3. **General $k$-th vector:**
   Subtract the projections of $\mathbf{a}_k$ onto all previous $\mathbf{q}_1, \dots, \mathbf{q}_{k-1}$:
   $$
   \mathbf{u}_k = \mathbf{a}_k - \sum_{i=1}^{k-1} (\mathbf{a}_k \cdot \mathbf{q}_i)\mathbf{q}_i, \qquad \mathbf{q}_k = \frac{\mathbf{u}_k}{\|\mathbf{u}_k\|_2}
   $$

---

## 23.3 Hand Calculation Example ($2 \times 2$)

Orthonormalize the following two column vectors:

$$
\mathbf{a}_1 =
\begin{bmatrix}
1 \\ 1
\end{bmatrix},
\quad
\mathbf{a}_2 =
\begin{bmatrix}
1 \\ 0
\end{bmatrix}
$$

### Step 1: Compute $\mathbf{u}_1$ and $\mathbf{q}_1$
* $\mathbf{u}_1 = \mathbf{a}_1 = [1, 1]^T$
* $\|\mathbf{u}_1\|_2 = \sqrt{1^2 + 1^2} = \sqrt{2}$
* Normalized $\mathbf{q}_1$:
  $$
  \mathbf{q}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix}
  $$

### Step 2: Compute $\mathbf{u}_2$ and $\mathbf{q}_2$
* Project $\mathbf{a}_2$ onto $\mathbf{q}_1$:
  $$
  \mathbf{a}_2 \cdot \mathbf{q}_1 = (1)\left(\frac{1}{\sqrt{2}}\right) + (0)\left(\frac{1}{\sqrt{2}}\right) = \frac{1}{\sqrt{2}}
  $$
* Subtract projection:
  $$
  \mathbf{u}_2 = \mathbf{a}_2 - (\mathbf{a}_2 \cdot \mathbf{q}_1)\mathbf{q}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix} - \frac{1}{\sqrt{2}} \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \end{bmatrix} - \begin{bmatrix} 1/2 \\ 1/2 \end{bmatrix} = \begin{bmatrix} 1/2 \\ -1/2 \end{bmatrix}
  $$
* Check orthogonality: $\mathbf{u}_1 \cdot \mathbf{u}_2 = (1)(1/2) + (1)(-1/2) = 0 \quad \checkmark$
* Normalize $\mathbf{u}_2$:
  $$
  \|\mathbf{u}_2\|_2 = \sqrt{(1/2)^2 + (-1/2)^2} = \sqrt{1/4 + 1/4} = \sqrt{1/2} = \frac{1}{\sqrt{2}}
  $$
  $$
  \mathbf{q}_2 = \frac{\mathbf{u}_2}{\|\mathbf{u}_2\|_2} = \sqrt{2} \begin{bmatrix} 1/2 \\ -1/2 \end{bmatrix} = \begin{bmatrix} 1/\sqrt{2} \\ -1/\sqrt{2} \end{bmatrix}
  $$

---

## 23.4 QR Factorization ($A = QR$)

Matrix representation of Gram-Schmidt yields the **QR Decomposition**:

$$
A = QR
$$

where:
* $Q \in \mathbb{R}^{m \times n}$ has **orthonormal columns** ($Q^T Q = I$).
* $R \in \mathbb{R}^{n \times n}$ is an **upper-triangular matrix** containing the projection dot products:
  $$
  R_{ij} = \mathbf{q}_i^T \mathbf{a}_j \quad (R_{ij} = 0 \text{ for } i > j)
  $$

### Constructing $Q$ and $R$ from the Hand Example:
1. Matrix $Q$:
   $$
   Q = \begin{bmatrix} \mathbf{q}_1 & \mathbf{q}_2 \end{bmatrix} = \begin{bmatrix} 1/\sqrt{2} & 1/\sqrt{2} \\ 1/\sqrt{2} & -1/\sqrt{2} \end{bmatrix}
   $$
2. Matrix $R = Q^T A$:
   * $R_{11} = \mathbf{q}_1^T \mathbf{a}_1 = \frac{1}{\sqrt{2}}(1) + \frac{1}{\sqrt{2}}(1) = \sqrt{2}$
   * $R_{12} = \mathbf{q}_1^T \mathbf{a}_2 = \frac{1}{\sqrt{2}}(1) + \frac{1}{\sqrt{2}}(0) = 1/\sqrt{2}$
   * $R_{21} = 0$
   * $R_{22} = \mathbf{q}_2^T \mathbf{a}_2 = \frac{1}{\sqrt{2}}(1) - \frac{1}{\sqrt{2}}(0) = 1/\sqrt{2}$
   $$
   R = \begin{bmatrix} \sqrt{2} & 1/\sqrt{2} \\ 0 & 1/\sqrt{2} \end{bmatrix}
   $$
3. Verify $QR = A$:
   $$
   QR = \begin{bmatrix} 1/\sqrt{2} & 1/\sqrt{2} \\ 1/\sqrt{2} & -1/\sqrt{2} \end{bmatrix} \begin{bmatrix} \sqrt{2} & 1/\sqrt{2} \\ 0 & 1/\sqrt{2} \end{bmatrix} = \begin{bmatrix} 1 + 0 & 1/2 + 1/2 \\ 1 + 0 & 1/2 - 1/2 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix} = A \quad \checkmark
   $$

---

## 23.5 Why QR Solves Least Squares Stably

In Ordinary Least Squares, the Normal Equations are $X^T X \mathbf{w} = X^T \mathbf{y}$.

Substitute $X = QR$:

$$
(QR)^T (QR) \mathbf{w} = (QR)^T \mathbf{y} \implies R^T (Q^T Q) R \mathbf{w} = R^T Q^T \mathbf{y}
$$

Since $Q^T Q = I$:

$$
R^T R \mathbf{w} = R^T Q^T \mathbf{y} \implies R \mathbf{w} = Q^T \mathbf{y}
$$

### Why Every ML Library Uses QR Instead of $(X^T X)^{-1}$:
1. **Back-Substitution:** Because $R$ is upper-triangular, $R\mathbf{w} = Q^T \mathbf{y}$ is solved directly by backward substitution in $O(n^2)$ time with zero explicit matrix inversion.
2. **Condition Number Squaring Avoidance:**
   * Forming $X^T X$ squares the condition number: $\kappa(X^T X) = \kappa(X)^2$.
   * If $\kappa(X) = 10^4$ (moderate multicollinearity), then $\kappa(X^T X) = 10^8$. In IEEE single precision (7 digits), $(X^T X)$ becomes numerically singular!
   * QR solves the system with condition number $\kappa(X)$, preserving all available precision.

> [!TIP]
> **Common Interview Question:** *"Why does `np.linalg.lstsq` or `scipy.linalg.lstsq` use QR / SVD instead of computing $(X^T X)^{-1} X^T \mathbf{y}$?"*
> **Answer:** Computing $X^T X$ squares the condition number ($\kappa(X^T X) = \kappa(X)^2$), which causes catastrophic loss of floating-point precision and potential false singularity. QR decomposition solves $R\mathbf{w} = Q^T \mathbf{y}$ via back-substitution with condition number $\kappa(X)$, ensuring maximum numerical stability.

---

## 23.6 Advanced / Optional — Do not study until Core track is complete

### Classical Gram-Schmidt (CGS) vs Modified Gram-Schmidt (MGS)
In floating-point arithmetic, standard Gram-Schmidt (CGS) suffers from severe loss of orthogonality when columns are nearly collinear.
* **Modified Gram-Schmidt (MGS):** Projects each subsequent vector against the *already updated* orthogonal vectors rather than the original vectors, restoring numerical orthogonality.
* **Householder Reflections:** In production LAPACK/BLAS routines, QR is computed via Householder reflections ($H = I - 2\mathbf{v}\mathbf{v}^T$), which avoids all loss of orthogonality.

---

> 📖 **Navigation:** [← Previous: Part 22: Four Fundamental Subspaces (Strang's Big Picture)](./22_four_fundamental_subspaces.md) | [🏠 Index](./README.md) | [Next: Part 24: Positive Definite & Positive Semidefinite Matrices →](./24_positive_definite_matrices.md)
