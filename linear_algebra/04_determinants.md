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

> 📖 **Navigation:** [← Previous: Part 03: Matrix Inverses, Gauss-Jordan & Adjugate](./03_matrix_inverses_and_gauss_jordan.md) | [🏠 Index](./README.md) | [Next: Part 05: Systems of Linear Equations (Ax = b) →](./05_systems_of_linear_equations.md)
