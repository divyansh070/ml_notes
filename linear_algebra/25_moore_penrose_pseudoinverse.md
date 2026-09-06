> 📖 **Navigation:** [← Previous: Part 24: Positive Definite & Positive Semidefinite Matrices](./24_positive_definite_matrices.md) | [🏠 Index](./README.md) | [Next: Part 26: Linear Transformations & Change of Basis →](./26_linear_transformations_and_change_of_basis.md)

---

# PART 25 — MOORE-PENROSE PSEUDOINVERSE

---

## 25.1 Why Ordinary Inverse Fails

The standard matrix inverse $A^{-1}$ only exists if $A$ is **square** ($n \times n$) and **full rank** ($\det(A) \neq 0$). It fails when:
1. Matrix is **rectangular** ($m \times n$, e.g., $1000$ samples $\times 10$ features).
2. Matrix is **square but singular** ($\det(A) = 0$, redundant features).

---

## 25.2 What the Pseudoinverse Does

The **Moore-Penrose Pseudoinverse** $A^+$ is the unique matrix satisfying the 4 Moore-Penrose conditions:
1. $A A^+ A = A$
2. $A^+ A A^+ = A^+$
3. $(A A^+)^T = A A^+$
4. $(A^+ A)^T = A^+ A$

* **Intuition:** $A^+$ provides the "best possible inverse":
  * For **overdetermined systems** ($m \gt n$, more equations than unknowns), $A^+ \mathbf{b}$ gives the least-squares solution minimizing $\|\mathbf{y} - X\mathbf{w}\|_2^2$.
  * For **underdetermined systems** ($m \lt n$, infinitely many solutions), $A^+ \mathbf{b}$ picks the unique solution with the minimum $L_2$ norm ($\|\mathbf{w}\|_2$).

---

## 25.3 Least Squares Connection

$$
\mathbf{w}_{\text{LS}} = A^+ \mathbf{b}
$$

* When $A$ has full column rank ($\text{rank}(A) = n$), the pseudoinverse formula is:

$$
A^+ = (A^T A)^{-1} A^T
$$

* When $A$ has full row rank ($\text{rank}(A) = m$), the right-inverse formula is:

$$
A^+ = A^T (A A^T)^{-1}
$$

* **Universal SVD Formulation:** For ANY matrix of any rank, SVD $A = U \Sigma V^T$ gives:

$$
A^+ = V \Sigma^+ U^T
$$

  where $\Sigma^+$ transposes $\Sigma$ and inverts all non-zero singular values ($\sigma_i \to 1/\sigma_i$).

---

## 25.4 Complete Step-by-Step Hand Calculation

Find the pseudoinverse of the rectangular column vector $A = [1, 2]^T \in \mathbb{R}^{2 \times 1}$:
1. Since $A$ has full column rank ($r = 1 = n$), use $A^+ = (A^T A)^{-1} A^T$:
2. Compute $A^T A$:

$$
A^T A =
\begin{bmatrix}
1 & 2
\end{bmatrix}
\begin{bmatrix}
1 \\
2
\end{bmatrix} = 1^2 + 2^2 = 5
$$

3. Invert scalar $(A^T A)^{-1} = \frac{1}{5} = 0.2$.
4. Multiply by $A^T$:

$$
A^+ = \frac{1}{5}
\begin{bmatrix}
1 & 2
\end{bmatrix} =
\begin{bmatrix}
0.2 & 0.4
\end{bmatrix}
$$

5. **Verify Moore-Penrose Property** ($A^+ A = I_1$):

$$
A^+ A =
\begin{bmatrix}
0.2 & 0.4
\end{bmatrix}
\begin{bmatrix}
1 \\
2
\end{bmatrix} = 0.2(1) + 0.4(2) = 0.2 + 0.8 = 1.0 = I_1 \quad \checkmark
$$

6. **Solve Least Squares Problem:**
   Let target vector $\mathbf{b} = [3, 1]^T$. Find best scalar $x$:

$$
\hat{x} = A^+ \mathbf{b} =
\begin{bmatrix}
0.2 & 0.4
\end{bmatrix}
\begin{bmatrix}
3 \\
1
\end{bmatrix} = 0.2(3) + 0.4(1) = 0.6 + 0.4 = \mathbf{1.0}
$$

   * Fitted point: $A\hat{x} = [1, 2]^T(1) = [1, 2]^T$.
   * Residual error: $\mathbf{e} = \mathbf{b} - A\hat{x} = [3-1, 1-2]^T = [2, -1]^T$.
   * Orthogonality check: $\mathbf{e} \cdot A = 2(1) + (-1)(2) = 0 \quad \checkmark$.

---

## 25.5 ML Applications

* **Linear Regression:** Directly computes $\mathbf{w} = X^+ \mathbf{y}$ even if $X$ contains collinear columns.
* **Ridge Regularization Connection:** As regularization $\alpha \to 0$, Ridge solution $(X^T X + \alpha I)^{-1} X^T \mathbf{y}$ converges smoothly to the minimum-norm pseudoinverse solution $X^+ \mathbf{y}$.

> [!TIP]
> **Common Interview Question:** *"When is the formula $A^+ = (A^T A)^{-1} A^T$ valid, and what should you use if $A$ does not have full column rank?"*
> **Answer:** The normal-equation formula $(A^T A)^{-1} A^T$ is valid only when $A$ has full column rank ($A^T A$ is non-singular and invertible). When $A$ is rank-deficient or has collinear columns, you must use the universal SVD formulation $A^+ = V \Sigma^+ U^T$, which inverts only the non-zero singular values ($1/\sigma_i$) and sets zero singular values to zero.

> [!WARNING]
> **Common Mistake:** Assuming $A^+ A = I$ is always the full identity matrix. For a rectangular matrix with $m \gt n$, $A^+ A = I_n$ (an $n \times n$ identity), but $A A^+ \neq I_m$ — instead, $A A^+$ is an orthogonal projection matrix onto the Column Space $C(A)$.

---

> 📖 **Navigation:** [← Previous: Part 24: Positive Definite & Positive Semidefinite Matrices](./24_positive_definite_matrices.md) | [🏠 Index](./README.md) | [Next: Part 26: Linear Transformations & Change of Basis →](./26_linear_transformations_and_change_of_basis.md)
