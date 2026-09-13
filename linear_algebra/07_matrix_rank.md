> 📖 **Navigation:** [← Previous: Part 06: Determinants](./06_determinants.md) | [🏠 Index](./README.md) | [Next: Part 08: Linear Independence, Span, Basis & Dimension →](./08_linear_independence_span_basis.md)

---

# PART 7 — MATRIX RANK

The **rank** of a matrix is the number of true, non-redundant geometric dimensions contained in its data. In machine learning, matrix rank determines whether feature sets suffer from multicollinearity and whether model parameters can be uniquely identified.

---

## 7.1 Definitions of Matrix Rank

For any matrix $A \in \mathbb{R}^{m \times n}$:

1. **Column Rank:** The maximum number of linearly independent column vectors in $A$.
2. **Row Rank:** The maximum number of linearly independent row vectors in $A$.
3. **The Fundamental Rank Theorem:** For EVERY matrix, row rank equals column rank:
   $$
   \operatorname{rank}(A) = \text{Column Rank} = \text{Row Rank} \le \min(m, n)
   $$

```
                         GEOMETRIC MEANING OF MATRIX RANK
    A matrix transformation A @ x maps input space into output space.
    Matrix Rank = The NUMBER OF TRUE DIMENSIONS that survive the transformation!
```

---

## 7.2 Determining Rank via Echelon Pivots

The most reliable computational method to find rank is **Gaussian Elimination**:

> [!IMPORTANT]
> **Rank equals the Number of Pivots:**
> Reduce matrix $A$ to Row Echelon Form (REF).
> $$
> \operatorname{rank}(A) = \text{Total number of non-zero rows (pivots) in REF}
> $$
> Elementary row operations never change the row rank or column rank of a matrix.

---

## 7.3 Full Rank vs. Rank-Deficient Matrices

1. **Full Column Rank ($\operatorname{rank}(A) = n \le m$):**
   * All $n$ columns are linearly independent.
   * No information is lost in the input feature space.
   * $A^T A$ is an $n \times n$ non-singular, invertible matrix.
2. **Full Row Rank ($\operatorname{rank}(A) = m \le n$):**
   * All $m$ rows are linearly independent.
   * The columns span the entire output space $\mathbb{R}^m$.
   * $A A^T$ is an $m \times m$ non-singular, invertible matrix.
3. **Full Rank Square Matrix ($\operatorname{rank}(A) = n$ for $n \times n$):**
   * Both full column rank and full row rank.
   * $\det(A) \neq 0 \iff A^{-1}$ exists.
4. **Rank-Deficient Matrix ($\operatorname{rank}(A) < \min(m, n)$):**
   * Redundant columns or rows exist.
   * The transformation collapses space into a lower dimension.
   * For square matrices, $\det(A) = 0$ (singular).

---

## 7.4 Complete Worked Numerical Examples

### Example 1: Full Rank $3 \times 3$ Matrix
$$
A = \begin{bmatrix}
1 & 2 & 1 \\
2 & 5 & 4 \\
1 & 1 & 0
\end{bmatrix}
$$

#### Row Reduction to REF:
1. $R_2 \leftarrow R_2 - 2R_1$: $[2, 5, 4] - 2[1, 2, 1] = [0, 1, 2]$
2. $R_3 \leftarrow R_3 - R_1$: $[1, 1, 0] - [1, 2, 1] = [0, -1, -1]$
   $$
   \begin{bmatrix} 1 & 2 & 1 \\ 0 & 1 & 2 \\ 0 & -1 & -1 \end{bmatrix}
   $$
3. $R_3 \leftarrow R_3 + R_2$: $[0, -1, -1] + [0, 1, 2] = [0, 0, 1]$
   $$
   \begin{bmatrix} 1 & 2 & 1 \\ 0 & 1 & 2 \\ 0 & 0 & 1 \end{bmatrix}
   $$
* There are **3 pivots** (entries $(1,1), (2,2), (3,3)$).
* **$\operatorname{rank}(A) = 3$ (Full Rank).** $\det(A) \neq 0$, all 3 columns are independent.

---

### Example 2: Rank-Deficient $3 \times 3$ Matrix
$$
B = \begin{bmatrix}
1 & 2 & 3 \\
2 & 4 & 6 \\
1 & 1 & 1
\end{bmatrix}
$$

#### Row Reduction to REF:
1. $R_2 \leftarrow R_2 - 2R_1$: $[2, 4, 6] - 2[1, 2, 3] = [0, 0, 0]$
2. $R_3 \leftarrow R_3 - R_1$: $[1, 1, 1] - [1, 2, 3] = [0, -1, -2]$
   $$
   \begin{bmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \\ 0 & -1 & -2 \end{bmatrix}
   $$
3. Swap $R_2 \leftrightarrow R_3$:
   $$
   \begin{bmatrix} 1 & 2 & 3 \\ 0 & -1 & -2 \\ 0 & 0 & 0 \end{bmatrix}
   $$
* There are only **2 pivots** (Row 3 is completely zero).
* **$\operatorname{rank}(B) = 2 < 3$ (Rank-Deficient).**
* Column 3 is a linear combination of Column 1 and Column 2 ($2 \times \text{Col } 2 - \text{Col } 1 = \text{Col } 3$).
* $\det(B) = 0$, non-invertible.

---

## 7.5 How Rank Governs Solutions to $A\mathbf{x} = \mathbf{b}$

Comparing $\operatorname{rank}(A)$ to the number of variables $n$:

| Matrix Dimensions | Rank | System Type | Solvability of $A\mathbf{x} = \mathbf{b}$ |
| :--- | :--- | :--- | :--- |
| **Square ($n \times n$)** | $r = n$ | Invertible | Unique solution for **every** $\mathbf{b}$ |
| **Square ($n \times n$)** | $r < n$ | Singular | 0 or $\infty$ solutions (depends on $\mathbf{b}$) |
| **Tall ($m > n$)** | $r = n$ | Overdetermined (Full Col Rank) | 0 or 1 solution (Least squares gives unique $\mathbf{x}_{\text{LS}}$) |
| **Wide ($m < n$)** | $r = m$ | Underdetermined (Full Row Rank) | $\infty$ solutions for every $\mathbf{b}$ |

---

## 7.6 Why this matters in ML

1. **Multicollinearity & Invertibility:** In linear regression $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$, the Gram matrix $X^T X$ has shape $d \times d$.
   $$
   \operatorname{rank}(X^T X) = \operatorname{rank}(X)
   $$
   If two features are perfectly collinear (e.g., `sqft` and `sqmeters`), $\operatorname{rank}(X) < d$. Consequently, $\operatorname{rank}(X^T X) < d \implies \det(X^T X) = 0$, making it **mathematically impossible to invert**!
2. **The Dummy Variable Trap:** When one-hot encoding a categorical feature with $K$ categories, including all $K$ binary columns alongside an intercept column $\mathbf{1}$ forces $\sum_{k=1}^K \mathbf{x}_k = \mathbf{1}$. This exact linear dependence reduces rank by 1. One category must always be dropped.
3. **Low-Rank Matrix Approximations (LoRA & Compression):** A weight matrix $W \in \mathbb{R}^{d \times k}$ with rank $r \ll \min(d, k)$ can be factored into $W = B A$ where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$. This cuts memory and compute from $\mathcal{O}(d \cdot k)$ to $\mathcal{O}(r(d + k))$.

---

## 7.7 Common mistakes

* **Confusing Matrix Dimensions with Rank:** An $8 \times 3$ matrix does NOT have rank 8. The rank is bounded by $\min(m, n) = \min(8, 3) = 3$.
* **Thinking Row Operations Change Rank:** Applying elementary row operations reshapes rows, but never destroys or creates pivots. Rank is strictly invariant under row operations.
* **Assuming Non-Zero Determinant on Non-Square Matrices:** Determinants are only defined for square matrices. For rectangular matrices, check rank via pivots or singular values.

---

> 📖 **Navigation:** [← Previous: Part 06: Determinants](./06_determinants.md) | [🏠 Index](./README.md) | [Next: Part 08: Linear Independence, Span, Basis & Dimension →](./08_linear_independence_span_basis.md)
