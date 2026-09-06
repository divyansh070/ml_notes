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

## 6.4 The Spectral Theorem & Orthogonal Diagonalization

The **Spectral Theorem** is the crowning jewel of symmetric matrix theory in machine learning.

### The Spectral Theorem Statement
For any real symmetric matrix $A = A^T \in \mathbb{R}^{n \times n}$:
1. All $n$ eigenvalues $\lambda_1, \lambda_2, \dots, \lambda_n$ are strictly **real numbers** ($\lambda_i \in \mathbb{R}$).
2. Eigenvectors corresponding to distinct eigenvalues are mutually **orthogonal**.
3. There exists an orthonormal matrix $Q = [\mathbf{q}_1, \dots, \mathbf{q}_n]$ ($Q^T Q = I$) such that $A$ can be **orthogonally diagonalized**:

$$
A = Q \Lambda Q^T = \sum_{i=1}^{n} \lambda_i \mathbf{q}_i \mathbf{q}_i^T
$$

```
                    SPECTRAL DECOMPOSITION (Outer Product Form)
           A = λ_1 (q_1 q_1^T) + λ_2 (q_2 q_2^T) + ... + λ_n (q_n q_n^T)
               └──────┬──────┘   └──────┬──────┘
                  Rank-1 Matrix     Rank-1 Matrix
                   (Weight λ_1)      (Weight λ_2)
```

* **Why this is fundamental in ML:** Every covariance matrix $\Sigma = \frac{1}{n-1}X_c^TX_c$ is symmetric, guaranteeing that PCA finds a complete, orthogonal coordinate system of variance axes!

---

## 6.5 Spectral Radius ($\rho(A)$) & System Stability in ML

The **Spectral Radius** $\rho(A)$ of a square matrix $A$ is the maximum absolute value of its eigenvalues:

$$
\rho(A) = \max_{1 \le i \le n} |\lambda_i|
$$

```
                         SPECTRAL RADIUS & STABILITY
    ┌──────────────────────────┬──────────────────────────────────────────┐
    │ Condition                │ System Behavior as k -> ∞                │
    ├──────────────────────────┼──────────────────────────────────────────┤
    │ ρ(A) < 1.0               │ lim A^k = 0 (Asymptotically Stable /     │
    │                          │ Vanishing Dynamics)                      │
    ├──────────────────────────┼──────────────────────────────────────────┤
    │ ρ(A) = 1.0               │ Steady State Convergence (Markov Chains, │
    │                          │ PageRank)                                │
    ├──────────────────────────┼──────────────────────────────────────────┤
    │ ρ(A) > 1.0               │ lim ||A^k|| = ∞ (Exploding Dynamics /    │
    │                          │ Instability)                             │
    └──────────────────────────┴──────────────────────────────────────────┘
```

* **Application to Recurrent Neural Networks (RNNs):** In an unrolled RNN with recurrent weight matrix $W_{\text{rec}}$, hidden state propagation involves powers $W_{\text{rec}}^t$. If $\rho(W_{\text{rec}}) > 1$, gradients explode exponentially; if $\rho(W_{\text{rec}}) < 1$, gradients vanish exponentially.

---

## 6.6 The Power Iteration Algorithm (Finding Principal Eigenvectors)

**Power Iteration** is the foundational iterative numerical algorithm used to find the dominant eigenvalue and its corresponding eigenvector (the algorithm behind Google's PageRank):

```
                        POWER ITERATION ALGORITHM
       1. Initialize random unit vector: x_0 (||x_0||_2 = 1)
       2. Repeat until convergence:
              x_{k+1} = (A x_k) / ||A x_k||_2
       3. Dominant Eigenvalue (Rayleigh Quotient):
              λ_1 ≈ x_k^T A x_k
```

### Why It Works (Mathematical Convergence Proof):
Express the initial guess $\mathbf{x}_0$ as a linear combination of the orthonormal eigenvectors of $A$: $\mathbf{x}_0 = c_1 \mathbf{q}_1 + c_2 \mathbf{q}_2 + \dots + c_n \mathbf{q}_n$.
Applying matrix $A$ $k$ times:

$$
A^k \mathbf{x}_0 = c_1 \lambda_1^k \mathbf{q}_1 + c_2 \lambda_2^k \mathbf{q}_2 + \dots + c_n \lambda_n^k \mathbf{q}_n = \lambda_1^k \left[ c_1 \mathbf{q}_1 + c_2 \left(\frac{\lambda_2}{\lambda_1}\right)^k \mathbf{q}_2 + \dots + c_n \left(\frac{\lambda_n}{\lambda_1}\right)^k \mathbf{q}_n \right]
$$

Since $|\lambda_1| > |\lambda_2| \ge \dots \ge |\lambda_n|$, the ratio terms $\left(\frac{\lambda_i}{\lambda_1}\right)^k \to 0$ as $k \to \infty$. Normalizing at each step leaves purely the principal eigenvector $\mathbf{q}_1$!

---

## 6.7 Essential Eigenvalue Properties & Algebraic Identities

```
                    MASTER EIGENVALUE ALGEBRAIC IDENTITIES
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Matrix Operation         │ Resulting Eigenvalues                       │
  ├──────────────────────────┼─────────────────────────────────────────────┤
  │ Matrix Power A^k         │ λ_i^k                                       │
  │ Matrix Inverse A^-1      │ 1 / λ_i  (if λ_i ≠ 0)                       │
  │ Shifted Matrix (A + c I) │ λ_i + c                                     │
  │ Scaled Matrix (c A)      │ c λ_i                                       │
  │ Matrix Transpose A^T     │ λ_i (Identical eigenvalues!)                │
  │ Similar Matrix P^-1 A P  │ λ_i (Eigenvalues are similarity invariants!) │
  └──────────────────────────┴─────────────────────────────────────────────┘
```

* **Gershgorin Circle Theorem:** Every eigenvalue of $A \in \mathbb{C}^{n \times n}$ lies within at least one Gershgorin disk $D(A_{ii}, R_i)$ in the complex plane, where center is the diagonal entry $A_{ii}$ and radius is the off-diagonal absolute row sum $R_i = \sum_{j \neq i} |A_{ij}|$.

---

> 📖 **Navigation:** [← Previous: Part 06: Linear Independence & Matrix Rank](./06_linear_independence_and_rank.md) | [🏠 Index](./README.md) | [Next: Part 08: Eigenvalues in Principal Component Analysis (PCA) →](./08_eigenvalues_in_pca.md)
