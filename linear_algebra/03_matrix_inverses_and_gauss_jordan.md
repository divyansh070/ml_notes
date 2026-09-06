> 📖 **Navigation:** [← Previous: Part 02: Matrices & Matrix Operations](./02_matrices_and_operations.md) | [🏠 Index](./README.md) | [Next: Part 04: Determinants & Geometric Scaling →](./04_determinants.md)

---

## 2.7 Matrix Inverse (2x2 Hand Derivation & Invertibility)

The **inverse** $A^{-1}$ of a square matrix $A$ is the matrix that undoes the transformation of $A$:

$$
A A^{-1} = A^{-1} A = I
$$

### 2x2 Inverse Formula:
For:

$$
A =
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
$$

$$
A^{-1} = \frac{1}{\det(A)}
\begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}
= \frac{1}{ad - bc}
\begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}
$$

### Step-by-Step Hand Calculation
Let:

$$
A =
\begin{bmatrix}
4 & 7 \\
2 & 6
\end{bmatrix}
$$

1. Calculate determinant:

$$
\det(A) = (4 \times 6) - (7 \times 2) = 24 - 14 = 10
$$

2. Swap diagonal elements ($4 \leftrightarrow 6$) and negate off-diagonal elements ($7 \to -7, 2 \to -2$):

$$
\text{Adj}(A) =
\begin{bmatrix}
6 & -7 \\
-2 & 4
\end{bmatrix}
$$

3. Multiply by $\frac{1}{\det(A)}$:

$$
A^{-1} = \frac{1}{10}
\begin{bmatrix}
6 & -7 \\
-2 & 4
\end{bmatrix} =
\begin{bmatrix}
0.6 & -0.7 \\
-0.2 & 0.4
\end{bmatrix}
$$

4. **Verify by multiplication ($A A^{-1} = I$):**

$$
A A^{-1} =
\begin{bmatrix}
4 & 7 \\
2 & 6
\end{bmatrix}
\begin{bmatrix}
0.6 & -0.7 \\
-0.2 & 0.4
\end{bmatrix} =
\begin{bmatrix}
4(0.6)+7(-0.2) & 4(-0.7)+7(0.4) \\
2(0.6)+6(-0.2) & 2(-0.7)+6(0.4)
\end{bmatrix} =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

### When Does an Inverse NOT Exist?
* A matrix is **singular (non-invertible)** if and only if **$\det(A) = 0$**.
* If $\det(A) = 0$, the transformation squashes space into a lower dimension (e.g. 2D plane into a 1D line), destroying information. You cannot divide by zero ($\frac{1}{0}$), so no inverse exists.

---

### 2.7.1 — Why the 2x2 Shortcut Doesn't Generalize

The quick $2 \times 2$ formula $\left(A^{-1} = \frac{1}{ad-bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}\right)$ relies on swapping elements and negating off-diagonals. While this is an algebraic shortcut of the **Cofactor / Adjugate method** ($A^{-1} = \frac{1}{\det(A)} \text{adj}(A)$), it **fails to provide a practical hand-computation method for larger matrices ($3 \times 3, 4 \times 4, N \times N$)**.

```
                COMPUTATIONAL COMPLEXITY: COFACTOR VS. GAUSS-JORDAN

    Matrix Size (N)       Cofactor / Adjugate O(N!)       Gauss-Jordan O(N³)
    ────────────────────────────────────────────────────────────────────────
         2 × 2                 2 operations                    8 operations   (2x2 Shortcut wins!)
         3 × 3                36 operations                   27 operations   (Gauss-Jordan takes over)
         4 × 4               576 operations                   64 operations
        10 × 10        ~3,628,800 operations               1,000 operations   (Gauss-Jordan ~3,600x faster)
        20 × 20        ~2.4 × 10¹⁸ operations              8,000 operations   (Cofactor impossible!)
```

* **The Combinatorial Explosion of Cofactors:** Building the adjugate matrix for an $N \times N$ matrix requires evaluating $N^2$ individual $(N-1) \times (N-1)$ sub-determinants. Laplace expansion grows factorially as $\mathcal{O}(N!)$. For a $4 \times 4$ matrix, you would need to calculate $16$ separate $3 \times 3$ determinants!
* **The Solution:** For general $N \times N$ matrices, **Gauss-Jordan Elimination** reduces the computational cost to polynomial time $\mathcal{O}(N^3)$, making it the universal standard for hand calculations and systematic algebraic reduction.

---

### 2.7.2 — General Method: Gauss-Jordan Elimination (Augmented Matrix)

The **Gauss-Jordan Elimination** method inverts an $N \times N$ matrix $A$ by solving the matrix equation $A X = I$ for $X = A^{-1}$ using simultaneous Elementary Row Operations (EROs).

#### The Augmented Matrix Concept: $[A \mid I] \implies [I \mid A^{-1}]$

1. **Construct the Augmented Matrix:** Write the target matrix $A$ on the left and the identity matrix $I$ of the same dimension on the right inside a single partitioned block matrix:
   $$
   [A \mid I] = \left[\begin{array}{ccc|ccc}
   a_{11} & a_{12} & a_{13} & 1 & 0 & 0 \\
   a_{21} & a_{22} & a_{23} & 0 & 1 & 0 \\
   a_{31} & a_{32} & a_{33} & 0 & 0 & 1
   \end{array}\right]
   $$
2. **Apply Row Operations:** Systematically apply elementary row operations to transform the left side $A$ into the Reduced Row Echelon Form (RREF), which is the identity matrix $I$.
3. **Extract the Inverse:** Every row operation is mathematically equivalent to multiplying on the left by an elementary matrix $E_k$. When the left side becomes the identity $I = (E_k \cdots E_2 E_1) A$, the right side simultaneously accumulates $(E_k \cdots E_2 E_1) I = A^{-1}$:
   $$
   [A \mid I] \xrightarrow{\text{Elementary Row Operations}} [I \mid A^{-1}]
   $$

```
    ┌─────────────────────────┐             ┌─────────────────────────┐
    │  [ A ]   │   [ I ]      │  ────────►  │  [ I ]   │  [ A⁻¹ ]     │
    │  Target  │   Identity   │    Row Ops  │ Identity │   Result     │
    └─────────────────────────┘             └─────────────────────────┘
```

#### The Three Allowed Elementary Row Operations:

| Operation | Mathematical Effect | Standard Notation | Example |
| :--- | :--- | :--- | :--- |
| **1. Row Swap** | Interchange two rows to position a non-zero pivot | $R_i \leftrightarrow R_j$ | $R_1 \leftrightarrow R_2$ |
| **2. Scalar Multiplication** | Multiply an entire row by a non-zero constant $c \neq 0$ | $R_i \leftarrow c R_i$ | $R_1 \leftarrow \frac{1}{2} R_1$ |
| **3. Row Addition** | Add a scalar multiple of row $j$ to row $i$ | $R_i \leftarrow R_i + c R_j$ | $R_2 \leftarrow R_2 - 2R_1$ |

---

#### Complete 3x3 Worked Hand Calculation Example

Let us invert the following invertible $3 \times 3$ matrix $A$:

$$
A =
\begin{bmatrix}
1 & 2 & 3 \\
0 & 1 & 4 \\
5 & 6 & 0
\end{bmatrix}
$$

---

##### Step 1: Set Up the Augmented Matrix $[A \mid I]$

$$
[A \mid I] =
\left[\begin{array}{ccc|ccc}
1 & 2 & 3 & 1 & 0 & 0 \\
0 & 1 & 4 & 0 & 1 & 0 \\
5 & 6 & 0 & 0 & 0 & 1
\end{array}\right]
$$

---

##### Step 2: Clear Column 1 Below the Main Diagonal
The pivot in Row 1, Column 1 is already $1$. Row 2 already has a $0$ in Column 1. Eliminate the $5$ in Row 3 using $R_3 \leftarrow R_3 - 5R_1$:

$$
\text{New } R_3 = [5 - 5(1), \quad 6 - 5(2), \quad 0 - 5(3) \quad \mid \quad 0 - 5(1), \quad 0 - 5(0), \quad 1 - 5(0)]
$$

$$
\text{New } R_3 = [0, \quad -4, \quad -15 \quad \mid \quad -5, \quad 0, \quad 1]
$$

$$
\left[\begin{array}{ccc|ccc}
1 & 2 & 3 & 1 & 0 & 0 \\
0 & 1 & 4 & 0 & 1 & 0 \\
0 & -4 & -15 & -5 & 0 & 1
\end{array}\right]
$$

---

##### Step 3: Clear Column 2 Below the Main Diagonal
The pivot in Row 2, Column 2 is already $1$. Eliminate the $-4$ in Row 3 using $R_3 \leftarrow R_3 + 4R_2$:

$$
\text{New } R_3 = [0 + 4(0), \quad -4 + 4(1), \quad -15 + 4(4) \quad \mid \quad -5 + 4(0), \quad 0 + 4(1), \quad 1 + 4(0)]
$$

$$
\text{New } R_3 = [0, \quad 0, \quad 1 \quad \mid \quad -5, \quad 4, \quad 1]
$$

$$
\left[\begin{array}{ccc|ccc}
1 & 2 & 3 & 1 & 0 & 0 \\
0 & 1 & 4 & 0 & 1 & 0 \\
0 & 0 & 1 & -5 & 4 & 1
\end{array}\right]
$$

*(The left matrix is now in Upper Triangular / Row Echelon Form! Next, we back-substitute upwards to achieve Reduced Row Echelon Form).*

---

##### Step 4: Clear Column 3 Above the Main Diagonal
The pivot in Row 3, Column 3 is $1$.

* **Eliminate $4$ in Row 2 using $R_2 \leftarrow R_2 - 4R_3$:**
  $$
  \text{New } R_2 = [0 - 4(0), \quad 1 - 4(0), \quad 4 - 4(1) \quad \mid \quad 0 - 4(-5), \quad 1 - 4(4), \quad 0 - 4(1)]
  $$
  $$
  \text{New } R_2 = [0, \quad 1, \quad 0 \quad \mid \quad 20, \quad -15, \quad -4]
  $$

* **Eliminate $3$ in Row 1 using $R_1 \leftarrow R_1 - 3R_3$:**
  $$
  \text{New } R_1 = [1 - 3(0), \quad 2 - 3(0), \quad 3 - 3(1) \quad \mid \quad 1 - 3(-5), \quad 0 - 3(4), \quad 0 - 3(1)]
  $$
  $$
  \text{New } R_1 = [1, \quad 2, \quad 0 \quad \mid \quad 16, \quad -12, \quad -3]
  $$

The augmented matrix is now:

$$
\left[\begin{array}{ccc|ccc}
1 & 2 & 0 & 16 & -12 & -3 \\
0 & 1 & 0 & 20 & -15 & -4 \\
0 & 0 & 1 & -5 & 4 & 1
\end{array}\right]
$$

---

##### Step 5: Clear Column 2 Above the Main Diagonal
Eliminate the $2$ in Row 1 using $R_1 \leftarrow R_1 - 2R_2$:

$$
\text{New } R_1 = [1 - 2(0), \quad 2 - 2(1), \quad 0 - 2(0) \quad \mid \quad 16 - 2(20), \quad -12 - 2(-15), \quad -3 - 2(-4)]
$$

$$
\text{New } R_1 = [1, \quad 0, \quad 0 \quad \mid \quad 16 - 40, \quad -12 + 30, \quad -3 + 8] = [1, \quad 0, \quad 0 \quad \mid \quad -24, \quad 18, \quad 5]
$$

The augmented matrix is fully reduced to $[I \mid A^{-1}]$:

$$
\left[\begin{array}{ccc|ccc}
1 & 0 & 0 & -24 & 18 & 5 \\
0 & 1 & 0 & 20 & -15 & -4 \\
0 & 0 & 1 & -5 & 4 & 1
\end{array}\right]
$$

---

##### Step 6: Extract the Inverse Matrix $A^{-1}$

$$
A^{-1} =
\begin{bmatrix}
-24 & 18 & 5 \\
20 & -15 & -4 \\
-5 & 4 & 1
\end{bmatrix}
$$

---

##### Step 7: Verification Step ($A A^{-1} = I$)
Multiply the original matrix $A$ by the derived $A^{-1}$ to confirm the identity matrix $I$:

$$
A A^{-1} =
\begin{bmatrix}
1 & 2 & 3 \\
0 & 1 & 4 \\
5 & 6 & 0
\end{bmatrix}
\begin{bmatrix}
-24 & 18 & 5 \\
20 & -15 & -4 \\
-5 & 4 & 1
\end{bmatrix}
$$

* **Row 1:**
  * Row 1 $\cdot$ Col 1: $1(-24) + 2(20) + 3(-5) = -24 + 40 - 15 = \mathbf{1}$
  * Row 1 $\cdot$ Col 2: $1(18) + 2(-15) + 3(4) = 18 - 30 + 12 = \mathbf{0}$
  * Row 1 $\cdot$ Col 3: $1(5) + 2(-4) + 3(1) = 5 - 8 + 3 = \mathbf{0}$
* **Row 2:**
  * Row 2 $\cdot$ Col 1: $0(-24) + 1(20) + 4(-5) = 0 + 20 - 20 = \mathbf{0}$
  * Row 2 $\cdot$ Col 2: $0(18) + 1(-15) + 4(4) = 0 - 15 + 16 = \mathbf{1}$
  * Row 2 $\cdot$ Col 3: $0(5) + 1(-4) + 4(1) = 0 - 4 + 4 = \mathbf{0}$
* **Row 3:**
  * Row 3 $\cdot$ Col 1: $5(-24) + 6(20) + 0(-5) = -120 + 120 + 0 = \mathbf{0}$
  * Row 3 $\cdot$ Col 2: $5(18) + 6(-15) + 0(4) = 90 - 90 + 0 = \mathbf{0}$
  * Row 3 $\cdot$ Col 3: $5(5) + 6(-4) + 0(1) = 25 - 24 + 0 = \mathbf{1}$

$$
A A^{-1} =
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix} = I_3 \quad \checkmark
$$

---

### 2.7.3 — The Cofactor / Adjugate Method for NxN (For Understanding, Not Hand Computation)

The general analytical formula for the inverse of any square matrix $A \in \mathbb{R}^{n \times n}$ is given by:

$$
A^{-1} = \frac{1}{\det(A)} \text{adj}(A) = \frac{1}{\det(A)} C^T
$$

where $\det(A) \neq 0$, $C$ is the **Cofactor Matrix**, and $\text{adj}(A) = C^T$ is the **Adjugate Matrix** (the transpose of the cofactor matrix).

#### 1. Definitions: Minors and Cofactors
* **Minor ($M_{ij}$):** The determinant of the $(n-1) \times (n-1)$ submatrix left after deleting row $i$ and column $j$ from $A$.
* **Cofactor ($C_{ij}$):** The minor scaled by an alternating sign factor based on row and column index parity:
  $$
  C_{ij} = (-1)^{i+j} M_{ij}
  $$

```
            THE CHECKERBOARD SIGN PATTERN: (-1)^(i+j)

                   ┌                 ┐
                   │  +   -   +   -  │
                   │  -   +   -   +  │
                   │  +   -   +   -  │
                   │  -   +   -   +  │
                   └                 ┘
```

#### 2. Proof that the 2x2 Formula is a Special Case of the Adjugate Formula
Let $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$. We can derive the $2 \times 2$ shortcut rigorously from first principles:
1. **Find all 4 Minors ($1 \times 1$ determinants):**
   * $M_{11} = |d| = d$
   * $M_{12} = |c| = c$
   * $M_{21} = |b| = b$
   * $M_{22} = |a| = a$
2. **Apply the Checkerboard Signs to form the Cofactor Matrix $C$:**
   $$
   C =
   \begin{bmatrix}
   +(-1)^{1+1} M_{11} & (-1)^{1+2} M_{12} \\
   (-1)^{2+1} M_{21} & (-1)^{2+2} M_{22}
   \end{bmatrix} =
   \begin{bmatrix}
   d & -c \\
   -b & a
   \end{bmatrix}
   $$
3. **Transpose the Cofactor Matrix to obtain the Adjugate $\text{adj}(A) = C^T$:**
   $$
   \text{adj}(A) = C^T =
   \begin{bmatrix}
   d & -b \\
   -c & a
   \end{bmatrix}
   $$
4. **Multiply by $\frac{1}{\det(A)}$:**
   $$
   A^{-1} = \frac{1}{ad - bc}
   \begin{bmatrix}
   d & -b \\
   -c & a
   \end{bmatrix}
   $$
*The famous $2 \times 2$ shortcut is simply Cramer's adjugate formula in its lowest-dimensional form!*

---

### 2.7.4 — Practical Reality: How Computers Actually Invert Matrices

> [!IMPORTANT]
> **Production ML Reality: Never Invert Matrices Explicitly!**
> In machine learning libraries (NumPy, SciPy, PyTorch, LAPACK, BLAS), solving a linear system $A\mathbf{x} = \mathbf{b}$ **NEVER computes $A^{-1}$ directly**. 
> * Writing `x = np.linalg.inv(A) @ b` in Python is an industry anti-pattern because it is both **computationally slower** and **numerically unstable**.
> * Production solvers use `np.linalg.solve(A, b)` under the hood via **LU Decomposition with Partial Pivoting ($PA = LU$)** or **QR Decomposition ($A = QR$)**.

#### Why Explicit Inversion Fails in Hardware:

1. **Numerical Instability & Catastrophic Cancellation:** Computing matrix inverses using determinants or standard elimination amplifies round-off errors in IEEE 754 64-bit floating point arithmetic. A matrix with a large condition number $\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}}$ produces massive numerical noise when inverted directly.
2. **Computational Inefficiency:**
   * Computing $A^{-1}$ explicitly and multiplying $A^{-1}\mathbf{b}$ requires $\approx \frac{4}{3} N^3$ floating-point operations (FLOPs).
   * Factoring $A = LU$ and solving via forward/backward substitution requires only $\approx \frac{2}{3} N^3$ FLOPs (a **$2\times$ speedup**).
3. **Connection to Normal Equations in ML:**
   * In Ordinary Least Squares regression $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$, explicitly computing $(X^T X)^{-1}$ squares the condition number: $\kappa(X^T X) = (\kappa(X))^2$. If $X$ is even slightly ill-conditioned, $(X^T X)^{-1}$ destroys predictive accuracy. Production libraries solve regression via **QR Decomposition** without ever computing an inverse (see [Section 13.4 — Why $(X^T X)^{-1}$ Fails in Practice](#134-why-xt-x-1-fails-in-practice-condition-number-qr-solvers)).

---

### 2.7.5 — When Is a Matrix NOT Invertible (Beyond just det = 0)

In technical interviews and theoretical mathematics, invertibility connects across every core concept in Linear Algebra. For an $n \times n$ square matrix $A$, the following statements are **all mathematically equivalent** (The Invertible Matrix Theorem):

```
                        THE INVERTIBLE MATRIX EQUIVALENCE WEB

                             ┌────────────────────────┐
                             │  Matrix A is Invertible│
                             │      (Non-Singular)    │
                             └───────────┬────────────┘
                                         │
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

#### Core Conditions of Non-Invertibility:
1. **$\det(A) = 0$ (Zero Determinant):** The transformation squashes geometric space from $n$ dimensions down into a lower-dimensional subspace (e.g. 3D space flattened into a 2D plane or 1D line), compressing volume to $0$.
2. **$\text{rank}(A) < n$ (Rank Deficient):** The matrix does not have full rank (see [Section 5.3 — Matrix Rank](#53-matrix-rank-column-rank-row-rank)). The true number of independent dimensions of information is strictly less than the number of features.
3. **Linearly Dependent Columns/Rows:** At least one column or row can be written as an exact linear combination of the others (see [Section 5.2 — Linear Dependence vs. Independence](#52-linear-dependence-vs-independence-hand-check)). In machine learning, this corresponds to **perfect multicollinearity** (redundant feature columns).
4. **Non-Trivial Null Space ($\text{Null}(A) \neq \{\mathbf{0}\}$):** There exist non-zero vectors $\mathbf{x} \neq \mathbf{0}$ such that $A\mathbf{x} = \mathbf{0}$. Because multiple distinct input vectors map to the exact same output origin $\mathbf{0}$, the transformation destroys information and no mathematical function can undo it.
5. **Zero Eigenvalue ($\exists \lambda = 0$):** If $A$ has an eigenvalue $\lambda = 0$, then $A\mathbf{v} = 0\mathbf{v} = \mathbf{0}$ for its corresponding eigenvector $\mathbf{v} \neq \mathbf{0}$, proving the matrix collapses space along direction $\mathbf{v}$ (see [Section 6.1 — Fundamental Eigenvalue Equation](#61-the-fundamental-equation-av-lambda-v)).

---

### 2.7.6 — The Sherman-Morrison Formula (Rank-1 Inverse Update)

In real-time machine learning (e.g. streaming data, online regression, Kalman filtering), data points arrive one by one. Inverting an updated matrix $(A + \mathbf{u}\mathbf{v}^T)$ from scratch costs $\mathcal{O}(n^3)$ operations. 

The **Sherman-Morrison Formula** computes the exact inverse of a rank-1 perturbed matrix in $\mathcal{O}(n^2)$ matrix-vector operations:

$$
(A + \mathbf{u}\mathbf{v}^T)^{-1} = A^{-1} - \frac{A^{-1} \mathbf{u} \mathbf{v}^T A^{-1}}{1 + \mathbf{v}^T A^{-1} \mathbf{u}}
$$

*(Condition: Valid as long as the scalar denominator $1 + \mathbf{v}^T A^{-1} \mathbf{u} \neq 0$).*

```
                    SHERMAN-MORRISON RANK-1 UPDATE SPEEDUP
   Full Inversion from Scratch:               Sherman-Morrison Online Update:
        (A + u v^T)^-1                                  A^-1 known
              │                                              │
              ▼                                              ▼
          O(n^3) FLOPs                                  O(n^2) FLOPs
  (e.g., n=1000 -> 1,000,000,000 ops)             (e.g., n=1000 -> 1,000,000 ops)
                                                  >>> 1000x FASTER! <<<
```

#### Step-by-Step Hand Calculation Example
Let $A = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix} \implies A^{-1} = \begin{bmatrix} 1/2 & 0 \\ 0 & 1/2 \end{bmatrix}$, with update vectors $\mathbf{u} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$ and $\mathbf{v} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$.

1. Compute $A^{-1} \mathbf{u}$:

$$
A^{-1} \mathbf{u} = \begin{bmatrix} 1/2 & 0 \\ 0 & 1/2 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 1/2 \\ 1 \end{bmatrix}
$$

2. Compute $\mathbf{v}^T A^{-1}$:

$$
\mathbf{v}^T A^{-1} = \begin{bmatrix} 3 & 1 \end{bmatrix} \begin{bmatrix} 1/2 & 0 \\ 0 & 1/2 \end{bmatrix} = \begin{bmatrix} 3/2 & 1/2 \end{bmatrix}
$$

3. Compute scalar denominator $1 + \mathbf{v}^T A^{-1} \mathbf{u}$:

$$
1 + \begin{bmatrix} 3 & 1 \end{bmatrix} \begin{bmatrix} 1/2 \\ 1 \end{bmatrix} = 1 + (3/2 + 1) = 1 + 2.5 = 3.5 = \frac{7}{2}
$$

4. Compute outer product matrix $(A^{-1} \mathbf{u})(\mathbf{v}^T A^{-1})$:

$$
\begin{bmatrix} 1/2 \\ 1 \end{bmatrix} \begin{bmatrix} 3/2 & 1/2 \end{bmatrix} = \begin{bmatrix} 3/4 & 1/4 \\ 3/2 & 1/2 \end{bmatrix}
$$

5. Assemble $(A + \mathbf{u}\mathbf{v}^T)^{-1}$:

$$
(A + \mathbf{u}\mathbf{v}^T)^{-1} = \begin{bmatrix} 1/2 & 0 \\ 0 & 1/2 \end{bmatrix} - \frac{2}{7} \begin{bmatrix} 3/4 & 1/4 \\ 3/2 & 1/2 \end{bmatrix} = \begin{bmatrix} 1/2 - 3/14 & 0 - 1/14 \\ 0 - 3/7 & 1/2 - 1/7 \end{bmatrix} = \begin{bmatrix} 2/7 & -1/14 \\ -3/7 & 5/14 \end{bmatrix}
$$

* **ML Role:** Recursive Least Squares (RLS) adaptive filters, quasi-Newton optimization (BFGS Hessian updates).

---

### 2.7.7 — The Woodbury Matrix Identity (Low-Rank Inversion Update)

The **Woodbury Formula** generalizes Sherman-Morrison to rank-$k$ updates where $U \in \mathbb{R}^{n \times k}, C \in \mathbb{R}^{k \times k}, V \in \mathbb{R}^{k \times n}$:

$$
(A + U C V)^{-1} = A^{-1} - A^{-1} U (C^{-1} + V A^{-1} U)^{-1} V A^{-1}
$$

> [!TIP]
> **Why Woodbury Powers Gaussian Processes & Kernel Methods:**
> * If $A$ is $N \times N$ with $N = 1,000,000$ (massive dataset) and we approximate the kernel using $k = 100$ inducing points:
> * Inverting $(A + UCV)$ directly requires inverting an $(N \times N)$ matrix $\to \mathcal{O}(N^3) = 10^{18}$ FLOPs (impossible!).
> * Using Woodbury, we only invert $(C^{-1} + VA^{-1}U)$, which is a tiny **$(k \times k) = (100 \times 100)$ matrix** $\to \mathcal{O}(k^3) = 10^6$ FLOPs!

---

### 2.7.8 — Block Matrix Inversion & The Schur Complement

For a partitioned $2 \times 2$ block matrix $M$:

$$
M =
\begin{bmatrix}
A & B \\
C & D
\end{bmatrix}
$$

Assuming $D$ is invertible, the **Schur Complement of $D$** in $M$ is defined as:

$$
S_D = A - B D^{-1} C
$$

The exact block inverse is:

$$
\begin{bmatrix}
A & B \\
C & D
\end{bmatrix}^{-1}
=
\begin{bmatrix}
S_D^{-1} & -S_D^{-1} B D^{-1} \\
-D^{-1} C S_D^{-1} & D^{-1} + D^{-1} C S_D^{-1} B D^{-1}
\end{bmatrix}
$$

* **Multivariate Gaussian Conditional Distribution:** For jointly Gaussian vectors $\begin{bmatrix} \mathbf{x}_1 \\ \mathbf{x}_2 \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{bmatrix}, \begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{bmatrix} \right)$, the conditional distribution $p(\mathbf{x}_1 \mid \mathbf{x}_2)$ has covariance matrix equal to the **Schur complement**:

$$
\Sigma_{1|2} = \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}
$$

---

### 2.7.9 — Left Inverses vs. Right Inverses (Non-Square Matrices)

Rectangular matrices ($M \times N$) do not have standard two-sided inverses. However, depending on their rank:

```
                  LEFT INVERSE vs. RIGHT INVERSE
        Tall Matrix (M > N)                 Wide Matrix (M < N)
      Full Column Rank (rank=N)             Full Row Rank (rank=M)
          ┌       ┐                             ┌                   ┐
          │       │                             │                   │
          │   A   │                             └                   ┘
          │       │                                    A
          └       ┘                            A_right^-1 = A^T (A A^T)^-1
     A_left^-1 = (A^T A)^-1 A^T                       A A_right^-1 = I_M
            A_left^-1 A = I_N                   (Minimum Norm Solution)
      (Ordinary Least Squares)
```

1. **Left Inverse (Tall Matrices, $M > N$, $\text{rank}=N$):**

$$
A_{\text{left}}^{-1} = (A^T A)^{-1} A^T \implies A_{\text{left}}^{-1} A = (A^T A)^{-1} (A^T A) = I_N
$$

   * *Role in ML:* The exact OLS regression normal equation weight solver $\mathbf{w} = A_{\text{left}}^{-1} \mathbf{y}$!

2. **Right Inverse (Wide Matrices, $M < N$, $\text{rank}=M$):**

$$
A_{\text{right}}^{-1} = A^T (A A^T)^{-1} \implies A A_{\text{right}}^{-1} = (A A^T)(A A^T)^{-1} = I_M
$$

   * *Role in ML:* Finds the underdetermined solution $\mathbf{x} = A_{\text{right}}^{-1} \mathbf{b}$ that has the **minimum Euclidean norm** $\|\mathbf{x}\|_2$ (foundation of deep learning over-parameterized interpolation).

---

> 📖 **Navigation:** [← Previous: Part 02: Matrices & Matrix Operations](./02_matrices_and_operations.md) | [🏠 Index](./README.md) | [Next: Part 04: Determinants & Geometric Scaling →](./04_determinants.md)
