> 📖 **Navigation:** [← Previous: Part 03: Systems of Linear Equations (Ax = b)](./03_systems_of_linear_equations.md) | [🏠 Index](./README.md) | [Next: Part 05: Matrix Inverses & Invertibility →](./05_matrix_inverses.md)

---

# PART 4 — GAUSSIAN ELIMINATION & GAUSS-JORDAN ELIMINATION

To solve linear systems and invert matrices systematically, we use systematic row reduction algorithms. There are two primary variants: **Gaussian Elimination** and **Gauss-Jordan Elimination**.

---

## 4.1 Gaussian Elimination vs. Gauss-Jordan Elimination

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           THE TWO ELIMINATION PATHS                         │
│                                                                             │
│   Augmented Matrix [ A | b ]                                                │
│              │                                                              │
│              ▼ Forward Elimination (Eliminate entries below pivots)         │
│   Row Echelon Form (REF)                                                    │
│        /           \                                                        │
│       /             \                                                       │
│      ▼               ▼ Backward Elimination (Eliminate above pivots + normalize)
│  Back-Substitution   Reduced Row Echelon Form (RREF) [ I | x ]              │
│      │               │                                                      │
│      ▼               ▼                                                      │
│  Solution x      Solution x (Read off directly)                             │
│                                                                             │
│  GAUSSIAN ELIMINATION                GAUSS-JORDAN ELIMINATION               │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **Gaussian Elimination:**
   * **Stage 1 (Forward Elimination):** Uses row operations to eliminate all entries below the main diagonal, converting the matrix into **Row Echelon Form (REF)** (upper-triangular structure).
   * **Stage 2 (Back-Substitution):** Solves the last variable first, then substitutes backwards up the equations to find preceding variables.
   * *Computational Cost:* Requires $\approx \frac{2}{3} n^3$ floating-point operations (FLOPs).

2. **Gauss-Jordan Elimination:**
   * Continues row operations past REF.
   * Normalizes every pivot to $1$ and eliminates all entries **above** each pivot as well, producing **Reduced Row Echelon Form (RREF)**.
   * The solution vector $\mathbf{x}$ is read directly off the right-hand column with zero substitution.
   * *Computational Cost:* Requires $\approx n^3$ FLOPs (more operations for solving systems, but essential for matrix inversion).

---

## 4.2 Complete Worked Example: Solving a System via Both Methods

Solve the $3 \times 3$ system:

$$
\begin{aligned}
x_1 + 2x_2 + x_3 &= 8 \\
2x_1 + 5x_2 + 4x_3 &= 21 \\
3x_1 + 7x_2 + 6x_3 &= 31
\end{aligned}
$$

### Initial Augmented Matrix:

$$
\left[\begin{array}{ccc|c}
1 & 2 & 1 & 8 \\
2 & 5 & 4 & 21 \\
3 & 7 & 6 & 31
\end{array}\right]
$$

---

### Phase 1: Forward Elimination (Gaussian Elimination to REF)

#### Step 1: Eliminate Column 1 below Pivot 1 ($a_{11} = 1$)
* $R_2 \leftarrow R_2 - 2R_1$:

$$
[2, 5, 4 \mid 21] - 2[1, 2, 1 \mid 8] = [0, 1, 2 \mid 5]
$$

* $R_3 \leftarrow R_3 - 3R_1$:

$$
[3, 7, 6 \mid 31] - 3[1, 2, 1 \mid 8] = [0, 1, 3 \mid 7]
$$

Augmented matrix becomes:

$$
\left[\begin{array}{ccc|c}
1 & 2 & 1 & 8 \\
0 & 1 & 2 & 5 \\
0 & 1 & 3 & 7
\end{array}\right]
$$

#### Step 2: Eliminate Column 2 below Pivot 2 ($a_{22} = 1$)
* $R_3 \leftarrow R_3 - R_2$:

$$
[0, 1, 3 \mid 7] - [0, 1, 2 \mid 5] = [0, 0, 1 \mid 2]
$$

$$
\left[\begin{array}{ccc|c}
1 & 2 & 1 & 8 \\
0 & 1 & 2 & 5 \\
0 & 0 & 1 & 2
\end{array}\right] \quad \leftarrow \mathbf{Row\ Echelon\ Form\ (REF)}
$$

---

### Gaussian Path: Back-Substitution
From the REF matrix:
1. **From Row 3:**

$$
x_3 = 2
$$

2. **Substitute $x_3 = 2$ into Row 2:**

$$
x_2 + 2(2) = 5 \implies x_2 + 4 = 5 \implies x_2 = 1
$$

3. **Substitute $x_2 = 1, x_3 = 2$ into Row 1:**

$$
x_1 + 2(1) + 1(2) = 8 \implies x_1 + 4 = 8 \implies x_1 = 4
$$

**Solution:** $\mathbf{x} = [4, 1, 2]^T$.

---

### Gauss-Jordan Path: Continue to RREF

Starting from the REF matrix:

$$
\left[\begin{array}{ccc|c}
1 & 2 & 1 & 8 \\
0 & 1 & 2 & 5 \\
0 & 0 & 1 & 2
\end{array}\right]
$$

#### Step 3: Eliminate entries above Pivot 3 ($a_{33} = 1$)
* $R_2 \leftarrow R_2 - 2R_3$:

$$
[0, 1, 2 \mid 5] - 2[0, 0, 1 \mid 2] = [0, 1, 0 \mid 1]
$$

* $R_1 \leftarrow R_1 - R_3$:

$$
[1, 2, 1 \mid 8] - [0, 0, 1 \mid 2] = [1, 2, 0 \mid 6]
$$

Matrix becomes:

$$
\left[\begin{array}{ccc|c}
1 & 2 & 0 & 6 \\
0 & 1 & 0 & 1 \\
0 & 0 & 1 & 2
\end{array}\right]
$$

#### Step 4: Eliminate entries above Pivot 2 ($a_{22} = 1$)
* $R_1 \leftarrow R_1 - 2R_2$:

$$
[1, 2, 0 \mid 6] - 2[0, 1, 0 \mid 1] = [1, 0, 0 \mid 4]
$$

$$
\left[\begin{array}{ccc|c}
1 & 0 & 0 & 4 \\
0 & 1 & 0 & 1 \\
0 & 0 & 1 & 2
\end{array}\right] \quad \leftarrow \mathbf{Reduced\ Row\ Echelon\ Form\ (RREF)}
$$

The solution $\mathbf{x} = [4, 1, 2]^T$ is read directly off the target column.

---

## 4.3 Matrix Inversion via Gauss-Jordan ($[A \mid I] \implies [I \mid A^{-1}]$)

The most important practical application of Gauss-Jordan elimination is computing the **matrix inverse** $A^{-1}$.

### The Mathematical Principle:
Augment matrix $A$ with the identity matrix $I$:

$$
[A \mid I]
$$

Perform elementary row operations until the left block becomes $I$. The right block will automatically be transformed into $A^{-1}$:

$$
[A \mid I] \xrightarrow{\text{Elementary Row Operations}} [I \mid A^{-1}]
$$

> [!NOTE]
> **Why does this work?**
> Each row operation is mathematically equivalent to multiplying on the left by an elementary matrix $E_k$. The entire elimination sequence represents multiplying by product matrix $E = E_k \dots E_2 E_1$:
>
> $$
> E [A \mid I] = [EA \mid EI] = [EA \mid E]
> $$
>
> If the left block becomes the identity matrix ($EA = I$), then by definition **$E = A^{-1}$**! Therefore, the right block becomes $EI = E = A^{-1}$.

---

## 4.4 Complete Worked Numerical Example: Inverting a $3 \times 3$ Matrix

Invert matrix $A$:

$$
A = \begin{bmatrix}
1 & 2 & 3 \\
0 & 1 & 4 \\
5 & 6 & 0
\end{bmatrix}
$$

### Step 1: Set up the Augmented Matrix $[A \mid I]$

$$
\left[\begin{array}{ccc|ccc}
1 & 2 & 3 & 1 & 0 & 0 \\
0 & 1 & 4 & 0 & 1 & 0 \\
5 & 6 & 0 & 0 & 0 & 1
\end{array}\right]
$$

### Step 2: Eliminate below Pivot 1 in Column 1 ($R_3 \leftarrow R_3 - 5R_1$)

$$
[5, 6, 0 \mid 0, 0, 1] - 5[1, 2, 3 \mid 1, 0, 0] = [0, -4, -15 \mid -5, 0, 1]
$$

$$
\left[\begin{array}{ccc|ccc}
1 & 2 & 3 & 1 & 0 & 0 \\
0 & 1 & 4 & 0 & 1 & 0 \\
0 & -4 & -15 & -5 & 0 & 1
\end{array}\right]
$$

### Step 3: Eliminate below Pivot 2 in Column 2 ($R_3 \leftarrow R_3 + 4R_2$)

$$
[0, -4, -15 \mid -5, 0, 1] + 4[0, 1, 4 \mid 0, 1, 0] = [0, 0, 1 \mid -5, 4, 1]
$$

$$
\left[\begin{array}{ccc|ccc}
1 & 2 & 3 & 1 & 0 & 0 \\
0 & 1 & 4 & 0 & 1 & 0 \\
0 & 0 & 1 & -5 & 4 & 1
\end{array}\right]
$$

### Step 4: Eliminate above Pivot 3 in Column 3
* $R_2 \leftarrow R_2 - 4R_3$:

$$
[0, 1, 4 \mid 0, 1, 0] - 4[0, 0, 1 \mid -5, 4, 1] = [0, 1, 0 \mid 20, -15, -4]
$$

* $R_1 \leftarrow R_1 - 3R_3$:

$$
[1, 2, 3 \mid 1, 0, 0] - 3[0, 0, 1 \mid -5, 4, 1] = [1, 2, 0 \mid 16, -12, -3]
$$

$$
\left[\begin{array}{ccc|ccc}
1 & 2 & 0 & 16 & -12 & -3 \\
0 & 1 & 0 & 20 & -15 & -4 \\
0 & 0 & 1 & -5 & 4 & 1
\end{array}\right]
$$

### Step 5: Eliminate above Pivot 2 in Column 2 ($R_1 \leftarrow R_1 - 2R_2$)

$$
[1, 2, 0 \mid 16, -12, -3] - 2[0, 1, 0 \mid 20, -15, -4] = [1, 0, 0 \mid -24, 18, 5]
$$

$$
\left[\begin{array}{ccc|ccc}
1 & 0 & 0 & -24 & 18 & 5 \\
0 & 1 & 0 & 20 & -15 & -4 \\
0 & 0 & 1 & -5 & 4 & 1
\end{array}\right]
$$

### Final Result:
The left block is $I_3$, so the right block is $A^{-1}$:

$$
A^{-1} = \begin{bmatrix}
-24 & 18 & 5 \\
20 & -15 & -4 \\
-5 & 4 & 1
\end{bmatrix}
$$

### Step 6: Numerical Verification ($A A^{-1} = I$)

$$
A A^{-1} = \begin{bmatrix}
1 & 2 & 3 \\
0 & 1 & 4 \\
5 & 6 & 0
\end{bmatrix}
\begin{bmatrix}
-24 & 18 & 5 \\
20 & -15 & -4 \\
-5 & 4 & 1
\end{bmatrix}
=
\begin{bmatrix}
-24 + 40 - 15 & 18 - 30 + 12 & 5 - 8 + 3 \\
0 + 20 - 20 & 0 - 15 + 16 & 0 - 4 + 4 \\
-120 + 120 + 0 & 90 - 90 + 0 & 25 - 24 + 0
\end{bmatrix}
=
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}
$$

✓ **Verified**.

---

## 4.5 Why this matters in ML

1. **LU Decomposition is Gaussian Elimination in Matrix Form:** When machine learning libraries solve $A\mathbf{x} = \mathbf{b}$ (`torch.linalg.solve` or `scipy.linalg.solve`), they perform Gaussian elimination factored as $A = L U$ (where $L$ records the row operations and $U$ is the upper-triangular REF). Solving $L\mathbf{y} = \mathbf{b}$ and $U\mathbf{x} = \mathbf{y}$ takes only $\mathcal{O}(n^2)$ back-substitution steps.
2. **Checking Invertibility in Real Time:** If row reduction ever produces a row of all zeros on the left side of $[A \mid I]$, the matrix has $\det(A) = 0$ (rank deficient) and cannot be inverted.

---

## 4.6 Common mistakes

* **Dividing by a Near-Zero Pivot:** In floating-point computation, dividing by a tiny pivot $a_{ii} \approx 0$ causes catastrophic numerical overflow. Professional solvers use **partial pivoting** (swapping rows to place the largest available entry on the pivot position).
* **Forgetting Operations on the Augmented Identity Matrix:** When inverting $[A \mid I]$, every row operation applied to the left side must be simultaneously applied to the right side.
* **Sign Errors in Row Replacements:** Computing $R_i - c R_j$ with negative numbers is the #1 source of paper-and-pencil exam errors. Always write out the scalar multiplication explicitly.

---

> 📖 **Navigation:** [← Previous: Part 03: Systems of Linear Equations (Ax = b)](./03_systems_of_linear_equations.md) | [🏠 Index](./README.md) | [Next: Part 05: Matrix Inverses & Invertibility →](./05_matrix_inverses.md)
