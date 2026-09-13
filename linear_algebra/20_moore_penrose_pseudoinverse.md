> 📖 **Navigation:** [← Previous: Part 19: Singular Value Decomposition (SVD)](./19_singular_value_decomposition.md) | [🏠 Index](./README.md) | [Next: Part 21: Linear Algebra for ML Synthesis →](./21_linear_algebra_for_ml_synthesis.md)

---

# PART 20 — MOORE-PENROSE PSEUDOINVERSE ($A^+$)

What do you do when a matrix is not square or not invertible, but you still need to solve $A\mathbf{x} = \mathbf{b}$? The **Moore-Penrose Pseudoinverse** $A^+$ generalizes matrix inversion to all rectangular, singular, and rank-deficient matrices.

---

## 20.1 Why Ordinary Inverses Fail in Real ML

The standard two-sided inverse $A^{-1}$ exists if and only if $A$ is square ($n \times n$) and non-singular ($\det(A) \neq 0$). In machine learning:

1. **Overdetermined Systems ($m > n$, e.g. 10,000 samples and 10 features):** No exact solution exists because $\mathbf{y} \notin \text{Col}(X)$.
2. **Underdetermined Systems ($m < n$, e.g. 50 samples and 5,000 gene features):** Infinitely many exact interpolating solutions exist.
3. **Rank-Deficient Matrices ($\det(X^T X) = 0$):** Multicollinear or duplicated features make the normal equations uninvertible.

---

## 20.2 The Universal SVD Definition of $A^+$

For ANY matrix $A \in \mathbb{R}^{m \times n}$ of arbitrary shape and rank, with Singular Value Decomposition $A = U \Sigma V^T$:

$$
A^+ = V \Sigma^+ U^T \in \mathbb{R}^{n \times m}
$$

where $\Sigma^+ \in \mathbb{R}^{n \times m}$ is formed by:
1. Transposing $\Sigma$.
2. Inverting every strictly non-zero singular value: $\sigma_i \to \frac{1}{\sigma_i}$.
3. Leaving all zero singular values as zero ($0 \to 0$).

```
        Sigma (m x n)                            Sigma^+ (n x m)
   ┌                      ┐                 ┌                      ┐
   │ σ_1   0     0      0 │                 │ 1/σ_1   0      0     │
   │  0   σ_2    0      0 │  ─────────────► │   0    1/σ_2   0     │
   │  0    0     0      0 │                 │   0     0      0     │
   └                      ┘                 │   0     0      0     │
                                            └                      ┘
```

---

## 20.3 When Can We Use Closed-Form Shortcut Formulas?

> [!WARNING]
> **Data Science Exam Trap: Never use $A^+ = (A^T A)^{-1} A^T$ as the universal definition!**
> That formula requires $A^T A$ to be invertible, which fails whenever $A$ is wide ($m < n$) or rank-deficient.

The valid formulas depend strictly on the rank and shape of $A$:

```
┌───────────────────────────────────────┬───────────────────────────────────┬────────────────────────────────────────┐
│ Matrix Condition                      │ Valid Pseudoinverse Formula       │ Inverse Property                       │
├───────────────────────────────────────┼───────────────────────────────────┼────────────────────────────────────────┤
│ 1. Square Invertible (m = n, full rank)│ A^+ = A^-1                        │ Two-sided: A A^+ = A^+ A = I           │
│ 2. Full Column Rank (m ≥ n, rank n)   │ A^+ = (A^T A)^-1 A^T              │ Left Inverse: A^+ A = I_n              │
│ 3. Full Row Rank (m ≤ n, rank m)      │ A^+ = A^T (A A^T)^-1              │ Right Inverse: A A^+ = I_m             │
│ 4. Rank-Deficient (rank < min(m, n))  │ A^+ = V Σ^+ U^T (SVD ONLY!)       │ Neither A^T A nor A A^T invertible     │
└───────────────────────────────────────┴───────────────────────────────────┴────────────────────────────────────────┘
```

---

## 20.4 Geometric Meaning: The Best Possible Solution

In every possible situation, the pseudoinverse solution $\mathbf{x}^* = A^+ \mathbf{b}$ gives the **mathematically optimal answer**:

1. **Overdetermined Systems ($m > n$, 0 exact solutions):**

$$
\mathbf{x}^* = A^+ \mathbf{b} \quad \text{minimizes the sum of squared errors } \|\mathbf{b} - A\mathbf{x}\|_2^2
$$

   *(It reproduces the Ordinary Least Squares projection).*
2. **Underdetermined Systems ($m < n$, $\infty$ exact solutions):**

$$
\mathbf{x}^* = A^+ \mathbf{b} \quad \text{selects the UNIQUE solution with the MINIMUM Euclidean norm } \|\mathbf{x}\|_2
$$

   *(It finds the weights that fit the data perfectly while keeping parameters as close to zero as possible).*

---

## 20.5 Complete Worked Numerical Example

Find the pseudoinverse of $A = [1, 2]^T \in \mathbb{R}^{2 \times 1}$.

### Method 1: Using the Full Column Rank Formula
* $m = 2, n = 1$. $\text{rank}(A) = 1 = n$ (Full column rank).
* $A^T A = [1, 2] [1, 2]^T = 1^2 + 2^2 = 5$.
* $(A^T A)^{-1} = \frac{1}{5} = 0.2$.
* Compute $A^+ = (A^T A)^{-1} A^T$:

$$
A^+ = \frac{1}{5} [1, 2] = \begin{bmatrix} 0.2 & 0.4 \end{bmatrix} \in \mathbb{R}^{1 \times 2}
$$

### Verification of Left Inverse Property ($A^+ A = I_1$):

$$
A^+ A = \begin{bmatrix} 0.2 & 0.4 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = (0.2)(1) + (0.4)(2) = 0.2 + 0.8 = 1.0 = I_1
$$

✓ **Verified**.

### Checking the Projection Matrix $P = A A^+$:

$$
A A^+ = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \begin{bmatrix} 0.2 & 0.4 \end{bmatrix} = \begin{bmatrix} 0.2 & 0.4 \\ 0.4 & 0.8 \end{bmatrix}
$$

*(Notice that $A A^+ \neq I_2$; it is the orthogonal projection matrix onto the column space of $A$!).*

---

## 20.6 The Four Moore-Penrose Conditions

A matrix $A^+$ is the unique Moore-Penrose pseudoinverse of $A$ if and only if it satisfies all four conditions:
1. $A A^+ A = A$ (Transforms column space consistently).
2. $A^+ A A^+ = A^+$ (Acts as a valid weak inverse).
3. $(A A^+)^T = A A^+$ ($A A^+$ is the symmetric projector onto $\text{Col}(A)$).
4. $(A^+ A)^T = A^+ A$ ($A^+ A$ is the symmetric projector onto $\text{Row}(A)$).

---

## 20.7 Why this matters in ML

1. **Overparameterized Deep Networks:** Modern neural networks have more parameters than data points ($m < n$). Gradient descent initialized near zero naturally converges to the **minimum-norm interpolating solution** $\mathbf{w} = X^+ \mathbf{y}$, which acts as an implicit regularizer preventing overfitting.
2. **Solving Collinear Linear Models:** When features are collinear, `np.linalg.pinv(X) @ y` safely computes the minimum-norm OLS weights without crashing due to singular matrices.

---

## 20.8 Common mistakes

* **Using $(X^T X)^{-1} X^T$ When Columns Are Collinear:** If columns are dependent, $\det(X^T X) = 0$, so $(X^T X)^{-1}$ does not exist! You must use the SVD-based pseudoinverse.
* **Assuming $A A^+ = I$ for Rectangular Matrices:** $A A^+$ is an $m \times m$ matrix. If $m > n$, it is a **projection matrix**, never the identity matrix ($A A^+ = P \neq I$).

---

> 📖 **Navigation:** [← Previous: Part 19: Singular Value Decomposition (SVD)](./19_singular_value_decomposition.md) | [🏠 Index](./README.md) | [Next: Part 21: Linear Algebra for ML Synthesis →](./21_linear_algebra_for_ml_synthesis.md)
