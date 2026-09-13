> 📖 **Navigation:** [← Previous: Part 05: Matrix Inverses & Invertibility](./05_matrix_inverses.md) | [🏠 Index](./README.md) | [Next: Part 07: Matrix Rank →](./07_matrix_rank.md)

---

# PART 6 — DETERMINANTS

The **determinant** of a square matrix $A \in \mathbb{R}^{n \times n}$, denoted $\det(A)$ or $|A|$, is a scalar that measures the **factor by which the linear transformation scales area, volume, or hyper-volume**.

---

## 6.1 Geometric Meaning: Area & Volume Scaling

A matrix transformation maps the standard unit coordinate hypercube into a parallelepiped formed by the matrix's column vectors:

```
     ORIGINAL UNIT SQUARE (Area = 1)               TRANSFORMED PARALLELOGRAM (Area = |det A|)
               y                                                 y
               │                                                 │           ● (a+b, c+d)
             1 ┼───● (0,1)                                       │          ╱ ╱
               │   │                                       c+d ──┼─────────● ╱
               │   │                                           c ┼───●    ╱ ╱
               └───┴────── x                                     └───┴────┴────── x
                   1                                                 b    a
```

* **2D Area Scaling:** The unit square (area $= 1$) transformed by $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$ becomes a parallelogram with geometric area $= |\det(A)| = |ad - bc|$.
* **3D Volume Scaling:** For a $3 \times 3$ matrix, $|\det(A)|$ is the volume of the parallelepiped spanned by its three column vectors.
* **Negative Determinant ($\det(A) < 0$):** Indicates an **orientation reversal** (a reflection across an axis, like looking in a mirror).
* **Zero Determinant ($\det(A) = 0$):** Indicates **dimensional collapse**. The transformation squashes space into a lower dimension (e.g., a 2D plane into a 1D line or point; 3D space into a 2D flat sheet). Volume collapses to $0$, information is permanently destroyed, and the matrix is **singular (non-invertible)**.

---

## 6.2 Determinant of a $2 \times 2$ Matrix

For:

$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
$$

$$
\det(A) = ad - bc
$$

### Example:

$$
A = \begin{bmatrix} 5 & 2 \\ 3 & 4 \end{bmatrix} \implies \det(A) = (5)(4) - (2)(3) = 20 - 6 = 14
$$

*(Any 2D region transformed by $A$ has its area multiplied by $14$).*

---

## 6.3 Determinant of a $3 \times 3$ Matrix

For a $3 \times 3$ matrix:

$$
A = \begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{bmatrix}
$$

### 1. Laplace Cofactor Expansion (Along Any Row or Column)
We expand along Row 1 using cofactors $C_{ij} = (-1)^{i+j} M_{ij}$:

$$
\det(A) = a_{11} C_{11} + a_{12} C_{12} + a_{13} C_{13} = a_{11} M_{11} - a_{12} M_{12} + a_{13} M_{13}
$$

Expanding the $2 \times 2$ minors:

$$
\det(A) = a_{11} \det\begin{bmatrix} a_{22} & a_{23} \\ a_{32} & a_{33} \end{bmatrix}
- a_{12} \det\begin{bmatrix} a_{21} & a_{23} \\ a_{31} & a_{33} \end{bmatrix}
+ a_{13} \det\begin{bmatrix} a_{21} & a_{22} \\ a_{31} & a_{32} \end{bmatrix}
$$

> [!TIP]
> **Expansion Strategy:** You can expand along **ANY row or column** and obtain the exact same determinant. Always choose the row or column with the **most zeros** to minimize calculations!

### 2. Sarrus' Rule (Diagonal Shortcut for $3 \times 3$)
Add the products of the three forward diagonals and subtract the products of the three backward diagonals:

$$
\det(A) = (a_{11}a_{22}a_{33} + a_{12}a_{23}a_{31} + a_{13}a_{21}a_{32}) - (a_{13}a_{22}a_{31} + a_{11}a_{23}a_{32} + a_{12}a_{21}a_{33})
$$

---

## 6.4 Determinants via Elementary Row Operations

For matrices of size $4 \times 4$ and larger, cofactor expansion takes $\mathcal{O}(n!)$ steps. In computational practice, determinants are calculated via Gaussian elimination in $\mathcal{O}(n^3)$ using three simple rules:

1. **Row Swap ($R_i \leftrightarrow R_j$):** Multiplies determinant by $-1$:

$$
\det(P A) = -\det(A)
$$

2. **Scalar Row Scaling ($R_i \leftarrow k R_i$):** Multiplies determinant by $k$:

$$
\det(\text{Scaled}) = k \det(A)
$$

3. **Row Addition ($R_i \leftarrow R_i + c R_j$):** **Leaves determinant UNCHANGED!**

$$
\det(\text{Row Added}) = \det(A)
$$

### Triangular Reduction:
Reducing $A$ to an upper-triangular matrix $U$ using only row additions (and $s$ row swaps):

$$
\det(A) = (-1)^s \prod_{i=1}^{n} u_{ii} = (-1)^s (u_{11} u_{22} \dots u_{nn})
$$

The determinant of any triangular matrix is simply the **product of its diagonal entries**.

---

## 6.5 Complete Worked Numerical Example ($3 \times 3$)

Compute the determinant of:

$$
A = \begin{bmatrix}
1 & 2 & 4 \\
3 & 8 & 14 \\
2 & 6 & 13
\end{bmatrix}
$$

### Method 1: Laplace Cofactor Expansion (Row 1)

$$
\det(A) = 1 \det\begin{bmatrix} 8 & 14 \\ 6 & 13 \end{bmatrix} - 2 \det\begin{bmatrix} 3 & 14 \\ 2 & 13 \end{bmatrix} + 4 \det\begin{bmatrix} 3 & 8 \\ 2 & 6 \end{bmatrix}
$$

* $M_{11} = (8)(13) - (14)(6) = 104 - 84 = 20$
* $M_{12} = (3)(13) - (14)(2) = 39 - 28 = 11$
* $M_{13} = (3)(6) - (8)(2) = 18 - 16 = 2$

$$
\det(A) = 1(20) - 2(11) + 4(2) = 20 - 22 + 8 = \mathbf{6}
$$

### Method 2: Gaussian Elimination to Triangular Form
1. Eliminate Column 1:
   * $R_2 \leftarrow R_2 - 3R_1$ (det unchanged): $[3, 8, 14] - 3[1, 2, 4] = [0, 2, 2]$
   * $R_3 \leftarrow R_3 - 2R_1$ (det unchanged): $[2, 6, 13] - 2[1, 2, 4] = [0, 2, 5]$

$$
\begin{bmatrix} 1 & 2 & 4 \\ 0 & 2 & 2 \\ 0 & 2 & 5 \end{bmatrix}
$$

2. Eliminate Column 2:
   * $R_3 \leftarrow R_3 - R_2$ (det unchanged): $[0, 2, 5] - [0, 2, 2] = [0, 0, 3]$

$$
U = \begin{bmatrix} 1 & 2 & 4 \\ 0 & 2 & 2 \\ 0 & 0 & 3 \end{bmatrix}
$$

3. Multiply diagonal pivots:

$$
\det(A) = (1) \times (2) \times (3) = \mathbf{6}
$$

✓ **Verified**.

---

## 6.6 Master Determinant Properties

1. **Identity:** $\det(I_n) = 1$.
2. **Product Rule:** $\det(AB) = \det(A)\det(B)$.
3. **Inverse Rule:** $\det(A^{-1}) = \frac{1}{\det(A)}$.
4. **Transpose Invariance:** $\det(A^T) = \det(A)$.
5. **Scalar Scaling:** For an $n \times n$ matrix, $\det(c A) = c^n \det(A)$ *(Each of the $n$ rows factors out a scalar $c$)*.
6. **Eigenvalues Connection:** The determinant equals the product of all eigenvalues:

$$
\det(A) = \prod_{i=1}^{n} \lambda_i = \lambda_1 \lambda_2 \dots \lambda_n
$$

7. **Singularity Test:** $\det(A) = 0 \iff A$ is singular $\iff \text{rank}(A) < n \iff A^{-1}$ does not exist.

---

## 6.7 Why this matters in ML

1. **Normalizing Flows & Generative Models:** When transforming probability distributions $\mathbf{y} = f(\mathbf{x})$, the change of variables formula scales probability densities by the absolute value of the **Jacobian determinant**:

$$
p_Y(\mathbf{y}) = p_X(f^{-1}(\mathbf{y})) \cdot \left| \det\left( \frac{\partial f^{-1}(\mathbf{y})}{\partial \mathbf{y}} \right) \right|
$$

   Normalizing Flows (e.g., RealNVP) design neural network layers with triangular Jacobians so the determinant is computed in $\mathcal{O}(n)$ time as the product of diagonal elements.
2. **Multivariate Gaussian Distributions:** The density of $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$ includes the term $\frac{1}{\sqrt{(2\pi)^d \det(\Sigma)}}$. If features are collinear, $\det(\Sigma) = 0$, causing division by zero!

---

## 6.8 Common mistakes

* **Assuming $\det(A + B) = \det(A) + \det(B)$:** Determinants are **NOT linear under addition**! For example, $\det(I) = 1$, but $\det(I + I) = \det(2I) = 2^n \neq 1 + 1$.
* **Scaling Error ($\det(c A) \neq c \det(A)$):** Scaling an $n \times n$ matrix by $2$ multiplies the volume by $2^n$, not $2$.
* **Confusing Determinant with Matrix Norm:** A norm $\|A\|$ measures size and is always non-negative. A determinant $\det(A)$ measures volume scaling and can be negative.

---

> 📖 **Navigation:** [← Previous: Part 05: Matrix Inverses & Invertibility](./05_matrix_inverses.md) | [🏠 Index](./README.md) | [Next: Part 07: Matrix Rank →](./07_matrix_rank.md)
