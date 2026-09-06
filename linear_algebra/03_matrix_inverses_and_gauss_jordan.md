> 📖 **Navigation:** [← Previous: Part 02: Matrices & Matrix Operations](./02_matrices_and_operations.md) | [🏠 Index](./README.md) | [Next: Part 04: Determinants & Geometric Scaling →](./04_determinants.md)

---

# PART 3 — MATRIX INVERSES & GAUSSIAN ELIMINATION

The **inverse** of a square matrix $A \in \mathbb{R}^{n \times n}$, denoted $A^{-1}$, is the matrix that "undoes" the linear transformation of $A$, returning space to its original coordinates.

---

## 3.1 What is an Inverse Matrix?

Just as multiplying by $\frac{1}{5}$ undoes multiplying by $5$ for scalars ($5 \times \frac{1}{5} = 1$), multiplying by $A^{-1}$ undoes matrix $A$:

$$
A A^{-1} = A^{-1} A = I
$$

```
                         THE INVERSE TRANSFORMATION
             x ────── Multiply by A ─────► y = Ax
               ◄── Multiply by A^-1 ─────
```

---

## 3.2 The 2x2 Inverse Shortcut Formula

For a $2 \times 2$ matrix:

$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
$$


$$
A^{-1} = \frac{1}{\det(A)} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
$$

### Step-by-Step Hand Calculation
Let:

$$
A = \begin{bmatrix} 4 & 7 \\ 2 & 6 \end{bmatrix}
$$

1. Determinant: $\det(A) = (4 \times 6) - (7 \times 2) = 24 - 14 = 10$.
2. Swap main diagonals ($4 \leftrightarrow 6$) and negate off-diagonals ($7 \to -7, 2 \to -2$):
   $$
   \text{adj}(A) = \begin{bmatrix} 6 & -7 \\ -2 & 4 \end{bmatrix}
   $$
3. Multiply by $\frac{1}{10}$:
   $$
   A^{-1} = \frac{1}{10} \begin{bmatrix} 6 & -7 \\ -2 & 4 \end{bmatrix} = \begin{bmatrix} 0.6 & -0.7 \\ -0.2 & 0.4 \end{bmatrix}
   $$
4. **Verification ($A A^{-1} = I$):**
   $$
   \begin{bmatrix} 4 & 7 \\ 2 & 6 \end{bmatrix} \begin{bmatrix} 0.6 & -0.7 \\ -0.2 & 0.4 \end{bmatrix} = \begin{bmatrix} 4(0.6)+7(-0.2) & 4(-0.7)+7(0.4) \\ 2(0.6)+6(-0.2) & 2(-0.7)+6(0.4) \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \quad \checkmark
   $$

---

## 3.3 Gaussian Elimination & Augmented Matrices

For matrices of size $3 \times 3$ and larger, the $2 \times 2$ swap shortcut does not work. We use **Gaussian Elimination** on an **Augmented Matrix**.

```
    ┌─────────────────────────┐             ┌─────────────────────────┐
    │  [ A ]   │   [ b ]      │  ────────►  │  [ U ]   │   [ c ]      │
    │  System  │   Target     │   Row Ops   │  Upper   │   New        │
    │  Matrix  │   Vector     │             │Triangular│  Target      │
    └─────────────────────────┘             └─────────────────────────┘
```

### The Three Allowed Elementary Row Operations (EROs):
1. **Row Swap ($R_i \leftrightarrow R_j$):** Interchange two equations.
2. **Scalar Multiplication ($R_i \leftarrow c R_i$):** Multiply an equation by non-zero scalar $c \neq 0$.
3. **Row Addition ($R_i \leftarrow R_i + c R_j$):** Add a multiple of one equation to another.

> [!NOTE]
> **Why Row Operations Preserve the Exact Solution Set:**
> Each elementary row operation corresponds to multiplying on the left by an invertible **elementary matrix** $E_k$. Since $E_k$ is invertible, $E_k A\mathbf{x} = E_k \mathbf{b}$ has the exact same solution vector $\mathbf{x}$ as $A\mathbf{x} = \mathbf{b}$.

---

## 3.4 Gauss-Jordan Matrix Inversion ($[A \mid I] \implies [I \mid A^{-1}]$)

To invert a matrix $A$, we set up the augmented matrix $[A \mid I]$ and row-reduce the left side to $I$. The right side simultaneously becomes $A^{-1}$:

$$
[A \mid I] \xrightarrow{\text{Elementary Row Operations}} [I \mid A^{-1}]
$$

### Complete 3x3 Worked Hand Calculation Example
Invert matrix $A$:
$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 1 & 4 \\ 5 & 6 & 0 \end{bmatrix}
$$

#### Step 1: Set up Augmented Matrix $[A \mid I]$
$$
\left[\begin{array}{ccc|ccc}
1 & 2 & 3 & 1 & 0 & 0 \\
0 & 1 & 4 & 0 & 1 & 0 \\
5 & 6 & 0 & 0 & 0 & 1
\end{array}\right]
$$

#### Step 2: Eliminate below diagonal in Column 1 ($R_3 \leftarrow R_3 - 5R_1$)
$$
\left[\begin{array}{ccc|ccc}
1 & 2 & 3 & 1 & 0 & 0 \\
0 & 1 & 4 & 0 & 1 & 0 \\
0 & -4 & -15 & -5 & 0 & 1
\end{array}\right]
$$

#### Step 3: Eliminate below diagonal in Column 2 ($R_3 \leftarrow R_3 + 4R_2$)
$$
\left[\begin{array}{ccc|ccc}
1 & 2 & 3 & 1 & 0 & 0 \\
0 & 1 & 4 & 0 & 1 & 0 \\
0 & 0 & 1 & -5 & 4 & 1
\end{array}\right]
$$

#### Step 4: Eliminate above diagonal in Column 3 ($R_2 \leftarrow R_2 - 4R_3$, $R_1 \leftarrow R_1 - 3R_3$)
$$
\left[\begin{array}{ccc|ccc}
1 & 2 & 0 & 16 & -12 & -3 \\
0 & 1 & 0 & 20 & -15 & -4 \\
0 & 0 & 1 & -5 & 4 & 1
\end{array}\right]
$$

#### Step 5: Eliminate above diagonal in Column 2 ($R_1 \leftarrow R_1 - 2R_2$)
$$
\left[\begin{array}{ccc|ccc}
1 & 0 & 0 & -24 & 18 & 5 \\
0 & 1 & 0 & 20 & -15 & -4 \\
0 & 0 & 1 & -5 & 4 & 1
\end{array}\right]
$$

#### Result:
$$
A^{-1} = \begin{bmatrix} -24 & 18 & 5 \\ 20 & -15 & -4 \\ -5 & 4 & 1 \end{bmatrix}
$$

---

## 3.5 Why We Usually DON'T Explicitly Compute $A^{-1}$ in Production ML

> [!IMPORTANT]
> **Production ML Anti-Pattern: Never write `x = np.linalg.inv(A) @ b`!**
> In production numerical software (PyTorch, SciPy, LAPACK, BLAS), solving $A\mathbf{x} = \mathbf{b}$ **NEVER computes $A^{-1}$ explicitly**. Always use `np.linalg.solve(A, b)`.

### Why Explicit Inversion Fails in Practice:
1. **Computational Speed:**
   * Computing $A^{-1}$ explicitly and multiplying $A^{-1}\mathbf{b}$ requires $\approx \frac{4}{3} n^3$ FLOPs.
   * Factoring $A = LU$ and solving via forward/backward substitution requires only $\approx \frac{2}{3} n^3$ FLOPs (**$2\times$ faster!**).
2. **Numerical Stability & Catastrophic Cancellation:**
   * Explicit matrix inversion amplifies floating-point round-off errors.
   * In linear regression $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$, explicitly forming and inverting $(X^T X)$ **squares the condition number**: $\kappa(X^T X) = (\kappa(X))^2$, causing learned weights to destabilize under multicollinearity. Modern libraries use **QR Decomposition** instead.

---

## 3.6 When Does an Inverse NOT Exist? (The Invertible Matrix Theorem)

For an $n \times n$ matrix $A$, the following conditions are **all mathematically equivalent**:

```
                         THE INVERTIBLE MATRIX EQUIVALENCE WEB
                              ┌────────────────────────┐
                              │  Matrix A is Invertible│
                              │      (Non-Singular)    │
                              └───────────┬────────────┘
                  ┌───────────────────────┼───────────────────────┐
                  │                       │                       │
                  ▼                       ▼                       ▼
        ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
        │   det(A) ≠ 0     │    │   rank(A) = n    │    │  Columns/Rows are│
        │ (Non-zero Volume)│    │   (Full Rank)    │    │  Lin. Independent│
        └──────────────────┘    └──────────────────┘    └──────────────────┘
                  │                       │                       │
                  ▼                       ▼                       ▼
        ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
        │  Null(A) = {0}   │    │  All Eigenvalues │    │  Ax = b has a    │
        │ (Trivial Kernel) │    │      λ_i ≠ 0     │    │ Unique Solution  │
        └──────────────────┘    └──────────────────┘    └──────────────────┘
```

* If $\det(A) = 0$, the transformation squashes $n$-dimensional space into a lower dimension (destroying information). You cannot divide by zero ($\frac{1}{\det(A)}$), so no inverse exists.

---

## Advanced / Optional — Do not study until Core track is complete

### A.1 The 3x3 Adjugate / Cofactor Matrix Inverse
The general formula is $A^{-1} = \frac{1}{\det(A)} C^T$, where $C$ is the matrix of cofactors ($C_{ij} = (-1)^{i+j} M_{ij}$).
For matrix:

$$
A = \begin{bmatrix} 2 & 1 & 1 \\ 3 & 2 & 1 \\ 2 & 1 & 2 \end{bmatrix}
$$

1. Determinant: $\det(A) = 1$.
2. Minors ($M_{ij}$) computed by crossing out row $i$ and column $j$:
   $$
   M = \begin{bmatrix} 3 & 4 & -1 \\ 1 & 2 & 0 \\ -1 & -1 & 1 \end{bmatrix}
   $$
3. Apply checkerboard sign pattern to get Cofactor Matrix $C$:
   $$
   C = \begin{bmatrix} 3 & -4 & -1 \\ -1 & 2 & 0 \\ -1 & 1 & 1 \end{bmatrix}
   $$
4. Transpose to get Adjugate $\text{adj}(A) = C^T$:
   $$
   \text{adj}(A) = \begin{bmatrix} 3 & -1 & -1 \\ -4 & 2 & 1 \\ -1 & 0 & 1 \end{bmatrix} \implies A^{-1} = \begin{bmatrix} 3 & -1 & -1 \\ -4 & 2 & 1 \\ -1 & 0 & 1 \end{bmatrix}
   $$

### A.2 The Sherman-Morrison Formula (Rank-1 Inverse Update)
In online streaming ML and Kalman filters, updating an inverse after adding rank-1 perturbation $\mathbf{u}\mathbf{v}^T$ takes $\mathcal{O}(n^2)$ rather than $\mathcal{O}(n^3)$:
$$
(A + \mathbf{u}\mathbf{v}^T)^{-1} = A^{-1} - \frac{A^{-1} \mathbf{u} \mathbf{v}^T A^{-1}}{1 + \mathbf{v}^T A^{-1} \mathbf{u}}
$$

### A.3 The Woodbury Matrix Identity
Generalizes low-rank updates for $U \in \mathbb{R}^{n \times k}, C \in \mathbb{R}^{k \times k}, V \in \mathbb{R}^{k \times n}$:
$$
(A + U C V)^{-1} = A^{-1} - A^{-1} U (C^{-1} + V A^{-1} U)^{-1} V A^{-1}
$$
*(Enables fast inversion in Gaussian Processes by inverting $k \times k$ inducing points rather than $n \times n$ data).*

### A.4 Block Matrix Inversion & The Schur Complement
For partitioned matrix:

$$
M = \begin{bmatrix} A & B \\ C & D \end{bmatrix}
$$

the Schur complement of $D$ is $S = A - B D^{-1} C$. In Multivariate Gaussians, the conditional covariance is $\Sigma_{1|2} = \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}$.

### A.5 Left Inverses vs. Right Inverses
* **Left Inverse (Tall $m > n$, full column rank):** $A_{\text{left}}^{-1} = (A^T A)^{-1} A^T \implies A_{\text{left}}^{-1} A = I_n$ (Ordinary Least Squares).
* **Right Inverse (Wide $m < n$, full row rank):** $A_{\text{right}}^{-1} = A^T (A A^T)^{-1} \implies A A_{\text{right}}^{-1} = I_m$ (Minimum-norm interpolator).

---

> 📖 **Navigation:** [← Previous: Part 02: Matrices & Matrix Operations](./02_matrices_and_operations.md) | [🏠 Index](./README.md) | [Next: Part 04: Determinants & Geometric Scaling →](./04_determinants.md)
