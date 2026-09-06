> 📖 **Navigation:** [← Previous: Part 05: Systems of Linear Equations (Ax = b)](./05_systems_of_linear_equations.md) | [🏠 Index](./README.md) | [Next: Part 07: Eigenvalues & Eigenvectors (13-Step Derivation) →](./07_eigenvalues_and_eigenvectors.md)

---

# PART 5 — LINEAR INDEPENDENCE & MATRIX RANK

---

## 5.1 Linear Combinations & Span

* **Linear Combination:** A vector $\mathbf{v}$ formed by multiplying vectors by scalars and adding them:

$$
\mathbf{v} = c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_k \mathbf{v}_k
$$

* **Span:** The entire set of all possible vectors that can be reached by all linear combinations of $\lbrace\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\rbrace$.

---

## 5.2 Linear Dependence vs. Independence (Hand Check)

A set of vectors is **Linearly Dependent** if at least one vector can be written as a linear combination of the others (redundant information). Otherwise, they are **Linearly Independent**.

### Hand Example A: Linearly Dependent Vectors
Let $\mathbf{v}_1 = [1, 2]^T$ and $\mathbf{v}_2 = [2, 4]^T$.
* *Test:* $\mathbf{v}_2 = 2 \mathbf{v}_1$.
* Both vectors lie along the exact same line in 2D space. Their span is only a 1D line, NOT the full 2D plane!
* Determinant check:

$$
\det
\begin{bmatrix}
1 & 2 \\
2 & 4
\end{bmatrix}
= 1(4) - 2(2) = 0 \implies \text{Dependent}
$$

### Hand Example B: Linearly Independent Vectors
Let $\mathbf{v}_1 = [1, 2]^T$ and $\mathbf{v}_2 = [2, 3]^T$.
* *Test:* Can we solve $c_1 [1, 2]^T + c_2 [2, 3]^T = [0, 0]^T$ with non-zero scalars?
  * $c_1 + 2c_2 = 0 \implies c_1 = -2c_2$
  * $2(-2c_2) + 3c_2 = 0 \implies -c_2 = 0 \implies c_2 = 0, c_1 = 0$.
* The only solution is the trivial solution ($c_1=0, c_2=0$).
* Determinant check:

$$
\det
\begin{bmatrix}
1 & 2 \\
2 & 3
\end{bmatrix}
= 1(3) - 2(2) = 3 - 4 = -1 \neq 0 \implies \text{Independent}
$$

---

## 5.3 Matrix Rank (Column Rank & Row Rank)

* **Rank of a Matrix ($\text{rank}(A)$):** The maximum number of linearly independent column vectors (or row vectors) in $A$.
* **Full Rank:** An $M \times N$ matrix is full rank if $\text{rank}(A) = \min(M, N)$.
* *Rank-Nullity Theorem Intuition:* $\text{Rank} = \text{True Dimensionality of Information}$. If a dataset has 10 columns but rank is 3, only 3 independent features exist; the other 7 are linear duplicates.

---

## 5.4 Connection to Multicollinearity & Singular Models

In Linear Regression:

$$
\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}
$$

* If feature 2 is redundant with feature 1 (e.g. $x_2 = 2.5 x_1$, like height in inches vs height in feet):
  * The columns of $X$ are **linearly dependent**.
  * $\text{rank}(X^T X) \lt p \implies \det(X^T X) = 0$.
  * $(X^T X)$ is **singular (non-invertible)**.
  * Ordinary Least Squares fails with a singular matrix error.
* *The Math Fix:* **Ridge Regularization ($L_2$)** adds $\alpha I$ to $(X^T X)$:

$$
\mathbf{w} = (X^T X + \alpha I)^{-1} X^T \mathbf{y}
$$

  Adding $\alpha \gt 0$ to the diagonal guarantees that $\det(X^T X + \alpha I) \gt 0$, making the matrix invertible and stable!

---

> 📖 **Navigation:** [← Previous: Part 05: Systems of Linear Equations (Ax = b)](./05_systems_of_linear_equations.md) | [🏠 Index](./README.md) | [Next: Part 07: Eigenvalues & Eigenvectors (13-Step Derivation) →](./07_eigenvalues_and_eigenvectors.md)
