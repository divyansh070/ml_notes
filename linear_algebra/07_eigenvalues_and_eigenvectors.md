> 📖 **Navigation:** [← Previous: Part 06: Linear Independence & Matrix Rank](./06_linear_independence_and_rank.md) | [🏠 Index](./README.md) | [Next: Part 08: Eigenvalues in Principal Component Analysis (PCA) →](./08_eigenvalues_in_pca.md)

---

# PART 6 — EIGENVALUES & EIGENVECTORS (13-STEP MASTER DERIVATION)

---

## 6.1 The Fundamental Equation: $Av = \lambda v$

When a matrix $A$ multiplies most vectors $\mathbf{x}$, it both **rotates** and **stretches** them.

However, for square matrices, there can exist special directions $\mathbf{v}$ where the matrix **ONLY STRETCHES or SHRINKS the vector, without changing its direction at all!**

$$
A \mathbf{v} = \lambda \mathbf{v}
$$

```
        TYPICAL VECTOR (Rotated & Stretched)            EIGENVECTOR (Only Stretched by Lambda!)
                 y                                                y
                 │        A @ x                                   │            A @ v = lambda * v
                 │       ╱                                        │           ╱
                 │      ╱                                         │          ╱
                 │     ● ◄── Rotated                              │         ●
                 │    ╱                                           │        ╱
                 │   ● x                                          │       ● v (Same direction!)
                 └───┴────────► x                                 └───────┴────────► x
```

* $\mathbf{v} \neq \mathbf{0}$ is the **Eigenvector** (the invariant direction).
* $\lambda \in \mathbb{R}$ is the **Eigenvalue** (the scalar stretch factor).

> [!NOTE]
> **Mathematical Rigor Note:** For an arbitrary square matrix, eigenvalues and eigenvectors may be real or complex. For the **real symmetric matrices** commonly encountered in PCA and Machine Learning (such as covariance matrices $\Sigma = \Sigma^T$), the Spectral Theorem guarantees that all eigenvalues and eigenvectors are strictly **real**, and eigenvectors corresponding to distinct eigenvalues are mutually **orthogonal**.

---

## 6.2 Complete 13-Step Hand Calculation on a 2x2 Matrix

We will solve for all eigenvalues and eigenvectors of:

$$
A =
\begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}
$$

---

### Step 1: Write the Fundamental Definition

$$
A \mathbf{v} = \lambda \mathbf{v}
$$

### Step 2: Move Everything to One Side

$$
A \mathbf{v} - \lambda \mathbf{v} = \mathbf{0} \quad \implies \quad (A - \lambda I) \mathbf{v} = \mathbf{0}
$$

### Step 3: Understand Why $\det(A - \lambda I) = 0$ is Required
If $(A - \lambda I)$ had an inverse, we could multiply both sides by $(A - \lambda I)^{-1}$:

$$
\mathbf{v} = (A - \lambda I)^{-1} \mathbf{0} = \mathbf{0}
$$

This would only give the trivial solution $\mathbf{v} = \mathbf{0}$. To find a **non-zero** eigenvector $\mathbf{v} \neq \mathbf{0}$, the matrix $(A - \lambda I)$ **MUST be non-invertible (singular)**. Therefore, its determinant must be zero:

$$
\det(A - \lambda I) = 0
$$

### Step 4: Construct the Matrix $(A - \lambda I)$

$$
A - \lambda I =
\begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix} - \lambda
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix} =
\begin{bmatrix}
2 - \lambda & 1 \\
1 & 2 - \lambda
\end{bmatrix}
$$

### Step 5: Compute the Determinant

$$
\det(A - \lambda I) = (2 - \lambda)(2 - \lambda) - (1 \times 1) = (2 - \lambda)^2 - 1
$$

### Step 6: Form the Characteristic Polynomial

$$
(4 - 4\lambda + \lambda^2) - 1 = 0 \quad \implies \quad \lambda^2 - 4\lambda + 3 = 0
$$

### Step 7: Solve for the Eigenvalues ($\lambda$)
Factor the quadratic equation:

$$
(\lambda - 3)(\lambda - 1) = 0 \quad \implies \quad \lambda_1 = 3, \quad \lambda_2 = 1
$$

*We have found our two eigenvalues!*

---

### Step 8: Find Eigenvector 1 for $\lambda_1 = 3$
Substitute $\lambda_1 = 3$ into $(A - \lambda I)\mathbf{v} = \mathbf{0}$:

$$
\begin{bmatrix}
2 - 3 & 1 \\
1 & 2 - 3
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix} =
\begin{bmatrix}
0 \\
0
\end{bmatrix}
\quad \implies \quad
\begin{bmatrix}
-1 & 1 \\
1 & -1
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix} =
\begin{bmatrix}
0 \\
0
\end{bmatrix}
$$

### Step 9: Solve the Simultaneous Equations for $\mathbf{v}_1$

$$
-1v_1 + 1v_2 = 0 \implies v_2 = v_1
$$

Any non-zero vector where $v_1 = v_2$ is an eigenvector. Choose $v_1 = 1 \implies v_2 = 1$:

$$
\mathbf{v}_1 =
\begin{bmatrix}
1 \\
1
\end{bmatrix}
\quad \quad \text{Normalized (Unit Length): } \mathbf{u}_1 =
\begin{bmatrix}
1/\sqrt{2} \\
1/\sqrt{2}
\end{bmatrix}
$$

---

### Step 10: Find Eigenvector 2 for $\lambda_2 = 1$
Substitute $\lambda_2 = 1$ into $(A - \lambda I)\mathbf{v} = \mathbf{0}$:

$$
\begin{bmatrix}
2 - 1 & 1 \\
1 & 2 - 1
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix} =
\begin{bmatrix}
0 \\
0
\end{bmatrix}
\quad \implies \quad
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix} =
\begin{bmatrix}
0 \\
0
\end{bmatrix}
$$

$$
1v_1 + 1v_2 = 0 \implies v_2 = -v_1
$$

Choose $v_1 = 1 \implies v_2 = -1$:

$$
\mathbf{v}_2 =
\begin{bmatrix}
1 \\
-1
\end{bmatrix}
\quad \quad \text{Normalized (Unit Length): } \mathbf{u}_2 =
\begin{bmatrix}
1/\sqrt{2} \\
-1/\sqrt{2}
\end{bmatrix}
$$

---

### Step 11: Verify the Result ($A \mathbf{v} = \lambda \mathbf{v}$)
1. **Check $\mathbf{v}_1$ with $\lambda_1 = 3$:**

$$
A \mathbf{v}_1 =
\begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}
\begin{bmatrix}
1 \\
1
\end{bmatrix} =
\begin{bmatrix}
2(1) + 1(1) \\
1(1) + 2(1)
\end{bmatrix} =
\begin{bmatrix}
3 \\
3
\end{bmatrix}
= 3
\begin{bmatrix}
1 \\
1
\end{bmatrix}
= \lambda_1 \mathbf{v}_1 \quad \checkmark
$$

2. **Check $\mathbf{v}_2$ with $\lambda_2 = 1$:**

$$
A \mathbf{v}_2 =
\begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}
\begin{bmatrix}
1 \\
-1
\end{bmatrix} =
\begin{bmatrix}
2(1) + 1(-1) \\
1(1) + 2(-1)
\end{bmatrix} =
\begin{bmatrix}
1 \\
-1
\end{bmatrix}
= 1
\begin{bmatrix}
1 \\
-1
\end{bmatrix}
= \lambda_2 \mathbf{v}_2 \quad \checkmark
$$

---

### Step 12: Verify Orthogonality of Eigenvectors
Because matrix $A$ is symmetric ($A = A^T$), its eigenvectors for distinct eigenvalues **must be orthogonal**:

$$
\mathbf{v}_1 \cdot \mathbf{v}_2 = (1 \times 1) + (1 \times -1) = 1 - 1 = 0 \quad \checkmark
$$

## 6.3 Geometric Meaning of Eigenvectors & Eigenvalues

### Step 13: Geometric Interpretation
* Any vector along the line $y = x$ ($\mathbf{v}_1$) is stretched by a factor of $3$.
* Any vector along the perpendicular line $y = -x$ ($\mathbf{v}_2$) is stretched by a factor of $1$ (unchanged in length).
* All other vectors in 2D space get rotated toward the dominant eigenvector direction $\mathbf{v}_1$.

---

> 📖 **Navigation:** [← Previous: Part 06: Linear Independence & Matrix Rank](./06_linear_independence_and_rank.md) | [🏠 Index](./README.md) | [Next: Part 08: Eigenvalues in Principal Component Analysis (PCA) →](./08_eigenvalues_in_pca.md)
