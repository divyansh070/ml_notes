> 📖 **Navigation:** [← Previous: Part 24: Positive Definite & Positive Semidefinite Matrices](./24_positive_definite_matrices.md) | [🏠 Index](./README.md) | [Next: Part 26: Linear Transformations & Change of Basis →](./26_linear_transformations_and_change_of_basis.md)

---

# PART 25 — MOORE-PENROSE PSEUDOINVERSE ($A^+$)

What do you do when a matrix is not square or not invertible, but you still need to solve $A\mathbf{x} = \mathbf{b}$? The **Moore-Penrose Pseudoinverse** $A^+$ generalizes matrix inversion to all rectangular, singular, and rank-deficient matrices.

---

## 25.1 Why Ordinary Inverses Fail in Real ML

The standard inverse $A^{-1}$ exists if and only if $A$ is **square** ($n \times n$) and **full rank** ($\det(A) \neq 0$). In machine learning:

1. **Rectangular Feature Matrices ($m \neq n$):**
   * Overdetermined ($m > n$, e.g., 10,000 samples and 10 features): No exact solution exists because $\mathbf{y} \notin C(X)$.
   * Underdetermined ($m < n$, e.g., 50 samples and 500 gene features): Infinitely many exact solutions exist.
2. **Singular Matrices ($\det(A) = 0$):**
   * Multicollinear or duplicate features cause $X^T X$ to be non-invertible.

---

## 25.2 What the Pseudoinverse Does Geometrically

The pseudoinverse $A^+ \mathbf{b}$ gives the **optimal solution** in every possible scenario:

1. **Overdetermined Systems ($m > n$):**
   $$
   \mathbf{x}_{\text{LS}} = A^+ \mathbf{b} \quad \text{minimizes the squared residual } \|\mathbf{b} - A\mathbf{x}\|_2^2
   $$
2. **Underdetermined Systems ($m < n$):**
   $$
   \mathbf{x}_{\text{min-norm}} = A^+ \mathbf{b} \quad \text{selects the unique solution with the smallest Euclidean norm } \|\mathbf{x}\|_2
   $$

---

## 25.3 Practical Formulas for $A^+$

Depending on the rank of $A \in \mathbb{R}^{m \times n}$:

### 1. Full Column Rank ($\text{rank}(A) = n \le m$ — Left Inverse)
When columns are linearly independent, $A^T A$ is invertible:

$$
A^+ = (A^T A)^{-1} A^T
$$

* Note: $A^+ A = (A^T A)^{-1} (A^T A) = I_n$ (acts as a left inverse).

### 2. Full Row Rank ($\text{rank}(A) = m \le n$ — Right Inverse)
When rows are linearly independent, $A A^T$ is invertible:

$$
A^+ = A^T (A A^T)^{-1}
$$

* Note: $A A^+ = (A A^T) (A A^T)^{-1} = I_m$ (acts as a right inverse).

### 3. Universal Formula via SVD (Any Shape, Any Rank)
For ANY matrix $A$ with SVD $A = U \Sigma V^T$:

$$
A^+ = V \Sigma^+ U^T
$$

where $\Sigma^+$ is formed by transposing $\Sigma$ and inverting all non-zero singular values:

$$
\sigma_i \to \frac{1}{\sigma_i} \quad (\text{if } \sigma_i > 0), \qquad 0 \to 0
$$

---

## 25.4 Complete Step-by-Step Hand Calculation

Find the pseudoinverse of the column vector $A = [1, 2]^T \in \mathbb{R}^{2 \times 1}$:

### Step 1: Check Rank
* $m = 2, n = 1$. Rank $r = 1 = n$ (full column rank).
* Use the formula $A^+ = (A^T A)^{-1} A^T$.

### Step 2: Compute $A^T A$
$$
A^T A = \begin{bmatrix} 1 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = 1^2 + 2^2 = 5
$$

### Step 3: Compute Scalar Inverse and Multiply by $A^T$
$$
(A^T A)^{-1} = \frac{1}{5} = 0.2
$$
$$
A^+ = \frac{1}{5} \begin{bmatrix} 1 & 2 \end{bmatrix} = \begin{bmatrix} 0.2 & 0.4 \end{bmatrix}
$$

### Step 4: Verify Properties
* Left inverse check:
  $$
  A^+ A = \begin{bmatrix} 0.2 & 0.4 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = 0.2(1) + 0.4(2) = 0.2 + 0.8 = 1.0 = I_1 \quad \checkmark
  $$
* Least-squares fit for $\mathbf{b} = [3, 1]^T$:
  $$
  \hat{x} = A^+ \mathbf{b} = \begin{bmatrix} 0.2 & 0.4 \end{bmatrix} \begin{bmatrix} 3 \\ 1 \end{bmatrix} = 0.2(3) + 0.4(1) = 0.6 + 0.4 = 1.0
  $$
  * Model prediction: $A\hat{x} = [1, 2]^T(1.0) = [1, 2]^T$.
  * Error: $\mathbf{e} = [3, 1]^T - [1, 2]^T = [2, -1]^T$.
  * Orthogonality check: $\mathbf{e} \cdot A = 2(1) + (-1)(2) = 0 \quad \checkmark$.

---

## 25.5 ML Connection: Ridge Regularization Limit

In Ridge Regression, the regularized solution is:

$$
\mathbf{w}_{\text{ridge}} = (X^T X + \alpha I)^{-1} X^T \mathbf{y}
$$

When $X^T X$ is singular and multiple solutions achieve minimal training error, taking the limit as $\alpha \to 0^+$ yields:

$$
\lim_{\alpha \to 0^+} (X^T X + \alpha I)^{-1} X^T \mathbf{y} = X^+ \mathbf{y}
$$

Ridge regression smoothly converges to the **minimum-norm pseudoinverse solution**, preventing model weights from exploding along collinear directions!

---

## 25.6 Advanced / Optional — Do not study until Core track is complete

### The 4 Formal Moore-Penrose Algebraic Conditions
A matrix $A^+$ is the unique pseudoinverse of $A$ if and only if it satisfies all 4 conditions:
1. $A A^+ A = A$ ($A A^+$ acts as identity on $C(A)$)
2. $A^+ A A^+ = A^+$ ($A^+$ is consistent)
3. $(A A^+)^T = A A^+$ ($A A^+$ is the symmetric orthogonal projection matrix onto $C(A)$)
4. $(A^+ A)^T = A^+ A$ ($A^+ A$ is the symmetric orthogonal projection matrix onto $C(A^T)$)

---

> 📖 **Navigation:** [← Previous: Part 24: Positive Definite & Positive Semidefinite Matrices](./24_positive_definite_matrices.md) | [🏠 Index](./README.md) | [Next: Part 26: Linear Transformations & Change of Basis →](./26_linear_transformations_and_change_of_basis.md)
