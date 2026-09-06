> 📖 **Navigation:** [← Previous: Part 03: Matrix Inverses, Gauss-Jordan & Adjugate](./03_matrix_inverses_and_gauss_jordan.md) | [🏠 Index](./README.md) | [Next: Part 05: Systems of Linear Equations (Ax = b) →](./05_systems_of_linear_equations.md)

---

# PART 3 — DETERMINANTS

---

## 3.1 The 2x2 Determinant Formula

For a $2 \times 2$ matrix:

$$
A =
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
$$

$$
\det(A) = |A| = ad - bc
$$

### Hand Calculation Example
Let:

$$
A =
\begin{bmatrix}
5 & 3 \\
2 & 4
\end{bmatrix}
$$

$$
\det(A) = (5 \times 4) - (3 \times 2) = 20 - 6 = 14
$$

---

## 3.2 Geometric Meaning: Area & Volume Scaling

The determinant represents the **factor by which a linear transformation scales area (in 2D) or volume (in 3D)**.

```
       ORIGINAL UNIT SQUARE (Area = 1)               TRANSFORMED PARALLELOGRAM (Area = det(A))
                y                                                 y
                │                                                 │           ● (a+b, c+d)
              1 ┼───● (0,1)                                       │          ╱ ╱
                │   │                                       c+d ──┼─────────● ╱
                │   │                                           c ┼───●    ╱ ╱
                └───┴────── x                                     └───┴────┴────── x
                    1                                                 b    a
```

* **Unit Square:** The unit square formed by basis vectors $\mathbf{i} = [1, 0]^T$ and $\mathbf{j} = [0, 1]^T$ has an initial area of $1 \times 1 = 1$.
* **After Transformation:**

$$
A =
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
$$

  * $\mathbf{i}$ moves to $[a, c]^T$
  * $\mathbf{j}$ moves to $[b, d]^T$
  * The resulting parallelogram has an exact geometric area equal to $|ad - bc| = \det(A)$.
* **Negative Determinant ($\det(A) \lt 0$):** Indicates the transformation flipped the orientation of space (like looking at an image in a mirror).

---

## 3.3 Why Determinants Matter in ML (Invertibility & Singularities)

| Determinant Value | Geometric Meaning | Algebraic Meaning | Machine Learning Impact |
| :--- | :--- | :--- | :--- |
| **$\det(A) \neq 0$** | Area/volume is scaled by non-zero factor; no dimensions lost. | Matrix is **full rank** and **invertible** ($A^{-1}$ exists). | Linear Regression normal equation $(X^T X)^{-1}$ has a unique, stable solution. |
| **$\det(A) = 0$** | Transformation collapses 2D space into a 1D line or 0D point. | Matrix is **singular / degenerate** ($A^{-1}$ does NOT exist). | **Multicollinearity bug:** Redundant features cause $(X^T X)$ to have determinant 0; OLS fails. |

---

## 3.4 Gilbert Strang's 10 Fundamental Determinant Properties

In MIT 18.06, Prof. Gilbert Strang establishes the 10 foundational mathematical axioms and rules governing determinants:

```
                    STRANG'S 10 DETERMINANT RULES
  1. det(I) = 1 (Unit cube has volume 1)
  2. Row swap reverses sign: det(P A) = -det(A) (for odd permutation)
  3. Linearity in each individual row: det(c * row_i) = c * det(A)
     -> Corollary: For n x n matrix, det(c A) = c^n det(A)
  4. Two identical rows -> det(A) = 0
  5. Row operations (Row_i - k * Row_j) leave det(A) UNCHANGED
  6. Row of all zeros -> det(A) = 0
  7. Triangular / Diagonal matrix: det(A) = d_1 * d_2 * ... * d_n (product of pivots)
  8. A is Invertible <==> det(A) ≠ 0
  9. Multiplicative Rule: det(AB) = det(A) * det(B)
     -> Corollary: det(A^-1) = 1 / det(A)
  10. Transpose Rule: det(A^T) = det(A)
```

> [!WARNING]
> **Classic Test Trap: $\det(cA) \neq c \det(A)$!**
> For an $n \times n$ matrix $A$ scaled by scalar $c$:
> 
> $$
> \det(c A) = c^n \det(A)
> $$
> 
> * *Hand Proof:* Scaling the whole matrix scales all $n$ rows by $c$. By Rule 3, factoring out $c$ from each of the $n$ rows yields $c^n$.

---

## 3.5 Determinant as the Product of Eigenvalues

For any square matrix $A \in \mathbb{R}^{n \times n}$ with eigenvalues $\lambda_1, \lambda_2, \dots, \lambda_n$:

$$
\det(A) = \prod_{i=1}^{n} \lambda_i = \lambda_1 \times \lambda_2 \times \dots \times \lambda_n
$$

* *Why this is true:* The characteristic polynomial is $p(\lambda) = \det(A - \lambda I) = (-1)^n \prod (\lambda - \lambda_i)$. Setting $\lambda = 0$ gives $\det(A) = \prod \lambda_i$.
* *Singularity connection:* If any single eigenvalue $\lambda_i = 0$, then $\det(A) = 0 \implies$ matrix is singular!

---

## 3.6 The Jacobian Determinant & Change of Variables in ML

When transforming a continuous random vector $\mathbf{x} \sim p_X(\mathbf{x})$ through an invertible, differentiable mapping $\mathbf{y} = f(\mathbf{x})$, probability density transforms via the **Jacobian Determinant**:

$$
p_Y(\mathbf{y}) = p_X(f^{-1}(\mathbf{y})) \cdot \left| \det \left( \frac{\partial f^{-1}(\mathbf{y})}{\partial \mathbf{y}} \right) \right| = p_X(\mathbf{x}) \cdot \left| \det \left( \frac{\partial f(\mathbf{x})}{\partial \mathbf{x}} \right) \right|^{-1}
$$

```
                   NORMALIZING FLOW VOLUME TRANSFORMATION
                 p_X(x)                                  p_Y(y)
               (Base Gaussian)                         (Complex Target)
                   ╭───╮                                   ╭─╮   ╭─╮
                   │   │             y = f(x)              │ │   │ │
                 ──┴───┴──          ─────────►           ──┴─┴───┴─┴──
                 Unit Volume                         Scaled by |det(J)|
```

* **ML Application (Generative Flow Models / RealNVP):** Normalizing Flows design neural network layers with **triangular Jacobian matrices**. Because the determinant of a triangular matrix is simply the product of its diagonal entries (Rule 7), computing $|\det(J)|$ requires only $\mathcal{O}(n)$ operations instead of $\mathcal{O}(n^3)$, making exact log-likelihood training tractable!

---

## 3.7 Cramer's Rule for Solving Linear Systems

For an invertible system $A\mathbf{x} = \mathbf{b}$, **Cramer's Rule** provides an explicit analytical formula for each unknown variable:

$$
x_i = \frac{\det(A_i)}{\det(A)}
$$

Where $A_i$ is the matrix formed by replacing the $i$-th column of $A$ with the target vector $\mathbf{b}$.

### Hand Calculation Example
Solve:

$$
\begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 5 \\ 5 \end{bmatrix}
$$

1. $\det(A) = (2 \times 3) - (1 \times 1) = 6 - 1 = 5$.
2. Replace Column 1 with $\mathbf{b}$: $A_1 = \begin{bmatrix} 5 & 1 \\ 5 & 3 \end{bmatrix} \implies \det(A_1) = (5 \times 3) - (1 \times 5) = 15 - 5 = 10$.
3. Replace Column 2 with $\mathbf{b}$: $A_2 = \begin{bmatrix} 2 & 5 \\ 1 & 5 \end{bmatrix} \implies \det(A_2) = (2 \times 5) - (5 \times 1) = 10 - 5 = 5$.
4. Solve:

$$
x_1 = \frac{\det(A_1)}{\det(A)} = \frac{10}{5} = 2, \quad x_2 = \frac{\det(A_2)}{\det(A)} = \frac{5}{5} = 1
$$

---

> 📖 **Navigation:** [← Previous: Part 03: Matrix Inverses, Gauss-Jordan & Adjugate](./03_matrix_inverses_and_gauss_jordan.md) | [🏠 Index](./README.md) | [Next: Part 05: Systems of Linear Equations (Ax = b) →](./05_systems_of_linear_equations.md)
