> 📖 **Navigation:** [← Previous: Part 04: Gaussian & Gauss-Jordan Elimination](./04_gaussian_and_gauss_jordan_elimination.md) | [🏠 Index](./README.md) | [Next: Part 06: Determinants →](./06_determinants.md)

---

# PART 5 — MATRIX INVERSES & INVERTIBILITY

The **inverse** of a square matrix $A \in \mathbb{R}^{n \times n}$, denoted $A^{-1}$, is the unique transformation that "undoes" the effect of $A$, returning space to its original coordinates:

$$
A A^{-1} = A^{-1} A = I_n
$$

---

## 5.1 The Four Types of Inverses in Linear Algebra

A common misconception is that non-square matrices cannot have inverses. In linear algebra and machine learning, we distinguish four distinct categories of inverses:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               THE MATRIX INVERSE SPECTRUM                               │
├───────────────────────┬──────────────────────────────────┬──────────────────────────────┤
│ Inverse Type          │ Matrix Shape & Rank              │ Defining Identity            │
├───────────────────────┼──────────────────────────────────┼──────────────────────────────┤
│ 1. Two-Sided Inverse  │ Square (n x n), Full Rank        │ A A^-1 = A^-1 A = I_n        │
│ 2. Left Inverse       │ Tall (m > n), Full Column Rank   │ A_left^-1 A = I_n            │
│ 3. Right Inverse      │ Wide (m < n), Full Row Rank      │ A A_right^-1 = I_m           │
│ 4. Pseudoinverse (A+) │ ANY Shape (m x n), ANY Rank      │ Universal via SVD            │
└───────────────────────┴──────────────────────────────────┴──────────────────────────────┘
```

1. **Two-Sided Inverse ($A^{-1}$):** Exists if and only if $A$ is **square** ($n \times n$) and **non-singular** ($\det(A) \neq 0$).
2. **Left Inverse ($A_{\text{left}}^{-1}$):** For tall matrices with independent columns ($m > n$, rank $n$). The left inverse is:

$$
A_{\text{left}}^{-1} = (A^T A)^{-1} A^T \implies A_{\text{left}}^{-1} A = (A^T A)^{-1} (A^T A) = I_n
$$

   *(This is the exact Ordinary Least Squares regression formula!)*
3. **Right Inverse ($A_{\text{right}}^{-1}$):** For wide matrices with independent rows ($m < n$, rank $m$). The right inverse is:

$$
A_{\text{right}}^{-1} = A^T (A A^T)^{-1} \implies A A_{\text{right}}^{-1} = (A A^T) (A A^T)^{-1} = I_m
$$

   *(This is the minimum-norm interpolator in overparameterized models).*
4. **Moore-Penrose Pseudoinverse ($A^+$):** Generalizes inversion to any rectangular or rank-deficient matrix via Singular Value Decomposition (covered fully in [Part 20](./20_moore_penrose_pseudoinverse.md)).

---

## 5.2 Why We Solve Systems Instead of Inverting in ML

Mathematically, if $A\mathbf{x} = \mathbf{b}$, multiplying both sides by $A^{-1}$ gives $\mathbf{x} = A^{-1}\mathbf{b}$.

> [!IMPORTANT]
> **Production ML Rule: Never write `x = np.linalg.inv(A) @ b`! Always use `x = np.linalg.solve(A, b)`.**
> 1. **Speed:** Computing $A^{-1}$ and multiplying takes $\approx \frac{4}{3} n^3$ FLOPs. Gaussian elimination / LU factorization takes $\approx \frac{2}{3} n^3$ FLOPs (**$2\times$ faster**).
> 2. **Numerical Stability:** Inverting directly squares the matrix condition number: $\kappa(A^T A) = (\kappa(A))^2$, causing catastrophic amplification of floating-point roundoff errors.

---

## 5.3 Derivation of the $2 \times 2$ Inverse Formula

Let $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$. We seek matrix $X = \begin{bmatrix} x_1 & x_2 \\ x_3 & x_4 \end{bmatrix}$ such that $A X = I_2$:

$$
\begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} x_1 & x_2 \\ x_3 & x_4 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$

This expands into two independent linear systems:

### System 1 (Column 1):

$$
\begin{aligned}
a x_1 + b x_3 &= 1 \quad \times d \implies a d x_1 + b d x_3 = d \\
c x_1 + d x_3 &= 0 \quad \times b \implies b c x_1 + b d x_3 = 0
\end{aligned}
$$

Subtracting the second equation from the first:

$$
(a d - b c) x_1 = d \implies x_1 = \frac{d}{a d - b c}
$$

Substitute $x_1$ back into $c x_1 + d x_3 = 0$:

$$
d x_3 = -c x_1 = \frac{-c d}{a d - b c} \implies x_3 = \frac{-c}{a d - b c}
$$

### System 2 (Column 2):

$$
\begin{aligned}
a x_2 + b x_4 &= 0 \quad \times d \implies a d x_2 + b d x_4 = 0 \\
c x_2 + d x_4 &= 1 \quad \times b \implies b c x_2 + b d x_4 = b
\end{aligned}
$$

Subtracting the second equation from the first:

$$
(a d - b c) x_2 = -b \implies x_2 = \frac{-b}{a d - b c}
$$

Substitute $x_2$ back into $a x_2 + b x_4 = 0$:

$$
b x_4 = -a x_2 = \frac{a b}{a d - b c} \implies x_4 = \frac{a}{a d - b c}
$$

### Assembling the Result:
Factoring out the common scalar denominator $\det(A) = ad - bc$:

$$
A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
$$

* **The Rule:** Swap the main diagonal entries ($a \leftrightarrow d$), negate the off-diagonal entries ($b \to -b, c \to -c$), and divide by $\det(A) = ad - bc$.

---

## 5.4 Inverse of a Matrix Using the Adjugate Method

For matrices of size $3 \times 3$ and larger, the **Adjugate (or Cofactor) Method** provides the general analytical formula for matrix inversion.

### The Pipeline of Concepts:
```
  Matrix A ──► Minors (M_ij) ──► Cofactors (C_ij) ──► Cofactor Matrix C ──► Transpose: adj(A) = C^T ──► A^-1 = (1/det A) adj(A)
```

1. **Minor of an Element ($M_{ij}$):** The determinant of the $(n-1) \times (n-1)$ submatrix left after crossing out Row $i$ and Column $j$ of matrix $A$.
2. **Cofactor ($C_{ij}$):** The minor signed by the checkerboard pattern:

$$
C_{ij} = (-1)^{i+j} M_{ij}
$$

3. **Cofactor Matrix ($C$):** The $n \times n$ matrix containing all cofactors $C_{ij}$.
4. **Adjugate Matrix ($\text{adj}(A)$):** The **transpose** of the cofactor matrix:

$$
\text{adj}(A) = C^T
$$

5. **The Final Inverse Formula:**

$$
A^{-1} = \frac{1}{\det(A)} \text{adj}(A) = \frac{1}{\det(A)} C^T
$$

---

### Why $A \text{adj}(A) = \det(A) I$ (Why We Transpose the Cofactor Matrix)

Consider the matrix product $A C^T$. The entry at Row $i$, Column $j$ is the dot product of Row $i$ of $A$ and Row $j$ of $C$:

$$
(A C^T)_{ij} = \sum_{k=1}^{n} a_{ik} C_{jk}
$$

* **When $i = j$ (Diagonal Entries):**

$$
\sum_{k=1}^{n} a_{ik} C_{ik} = a_{i1} C_{i1} + a_{i2} C_{i2} + \dots + a_{in} C_{in} = \det(A)
$$

  This is literally the definition of Laplace cofactor expansion of $\det(A)$ along Row $i$!
* **When $i \neq j$ (Off-Diagonal Entries):**

$$
\sum_{k=1}^{n} a_{ik} C_{jk} = 0
$$

  This corresponds to evaluating the determinant of a matrix whose $j$-th row has been replaced by its $i$-th row. Because this hypothetical matrix has two identical rows ($i$ and $j$), its determinant is **identically zero**!

Therefore, every diagonal entry of $A C^T$ is $\det(A)$ and every off-diagonal entry is $0$:

$$
A C^T = A \text{adj}(A) = \begin{bmatrix}
\det(A) & 0 & \dots & 0 \\
0 & \det(A) & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & \det(A)
\end{bmatrix} = \det(A) I_n
$$

Dividing both sides by $\det(A)$ gives:

$$
A \left( \frac{1}{\det(A)} \text{adj}(A) \right) = I_n \implies A^{-1} = \frac{1}{\det(A)} \text{adj}(A)
$$

■

> [!NOTE]
> **Terminology: Adjugate vs. Adjoint**
> In classical linear algebra, $C^T$ is named the **adjugate matrix**. In older textbooks, it is sometimes called the "adjoint". However, in modern mathematics and quantum mechanics, "adjoint" refers to the complex conjugate transpose ($A^* = \overline{A}^T$). We strictly use **adjugate** to eliminate ambiguity.

---

## 5.5 Complete $3 \times 3$ Worked Numerical Example

Invert the matrix:

$$
A = \begin{bmatrix}
2 & 1 & 1 \\
3 & 2 & 1 \\
2 & 1 & 2
\end{bmatrix}
$$

### Step 1: Compute All 9 Minors ($M_{ij}$)
Cross out row $i$ and column $j$, then compute the $2 \times 2$ determinant:
* $M_{11} = \det\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} = (2)(2) - (1)(1) = 3$
* $M_{12} = \det\begin{bmatrix} 3 & 1 \\ 2 & 2 \end{bmatrix} = (3)(2) - (1)(2) = 4$
* $M_{13} = \det\begin{bmatrix} 3 & 2 \\ 2 & 1 \end{bmatrix} = (3)(1) - (2)(2) = -1$
* $M_{21} = \det\begin{bmatrix} 1 & 1 \\ 1 & 2 \end{bmatrix} = (1)(2) - (1)(1) = 1$
* $M_{22} = \det\begin{bmatrix} 2 & 1 \\ 2 & 2 \end{bmatrix} = (2)(2) - (1)(2) = 2$
* $M_{23} = \det\begin{bmatrix} 2 & 1 \\ 2 & 1 \end{bmatrix} = (2)(1) - (1)(2) = 0$
* $M_{31} = \det\begin{bmatrix} 1 & 1 \\ 2 & 1 \end{bmatrix} = (1)(1) - (1)(2) = -1$
* $M_{32} = \det\begin{bmatrix} 2 & 1 \\ 3 & 1 \end{bmatrix} = (2)(1) - (1)(3) = -1$
* $M_{33} = \det\begin{bmatrix} 2 & 1 \\ 3 & 2 \end{bmatrix} = (2)(2) - (1)(3) = 1$

### Step 2: Apply Checkerboard Signs to Form Cofactors ($C_{ij} = (-1)^{i+j} M_{ij}$)
The sign pattern is $\begin{bmatrix} + & - & + \\ - & + & - \\ + & - & + \end{bmatrix}$:
* $C_{11} = +3$
* $C_{12} = -4$
* $C_{13} = -1$
* $C_{21} = -1$
* $C_{22} = +2$
* $C_{23} = -0 = 0$
* $C_{31} = +(-1) = -1$
* $C_{32} = -(-1) = +1$
* $C_{33} = +1$

### Step 3: Assemble Cofactor Matrix $C$

$$
C = \begin{bmatrix}
3 & -4 & -1 \\
-1 & 2 & 0 \\
-1 & 1 & 1
\end{bmatrix}
$$

### Step 4: Transpose $C$ to get the Adjugate Matrix $\text{adj}(A) = C^T$

$$
\text{adj}(A) = C^T = \begin{bmatrix}
3 & -1 & -1 \\
-4 & 2 & 1 \\
-1 & 0 & 1
\end{bmatrix}
$$

### Step 5: Compute the Determinant $\det(A)$
Using cofactor expansion along Row 1:

$$
\det(A) = a_{11} C_{11} + a_{12} C_{12} + a_{13} C_{13} = 2(3) + 1(-4) + 1(-1) = 6 - 4 - 1 = 1
$$

### Step 6: Assemble the Final Inverse
Because $\det(A) = 1$:

$$
A^{-1} = \frac{1}{1} \begin{bmatrix}
3 & -1 & -1 \\
-4 & 2 & 1 \\
-1 & 0 & 1
\end{bmatrix}
=
\begin{bmatrix}
3 & -1 & -1 \\
-4 & 2 & 1 \\
-1 & 0 & 1
\end{bmatrix}
$$

### Step 7: Verification ($A A^{-1} = I$)

$$
A A^{-1} = \begin{bmatrix}
2 & 1 & 1 \\
3 & 2 & 1 \\
2 & 1 & 2
\end{bmatrix}
\begin{bmatrix}
3 & -1 & -1 \\
-4 & 2 & 1 \\
-1 & 0 & 1
\end{bmatrix}
=
\begin{bmatrix}
6 - 4 - 1 & -2 + 2 + 0 & -2 + 1 + 1 \\
9 - 8 - 1 & -3 + 4 + 0 & -3 + 2 + 1 \\
6 - 4 - 2 & -2 + 2 + 0 & -2 + 1 + 2
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

## 5.6 Why this matters in ML

1. **Analytical vs. Numerical Inverses:** While production libraries use Gaussian/LU elimination ($\mathcal{O}(n^3)$) for computation, the adjugate formula $A^{-1} = \frac{1}{\det(A)} \text{adj}(A)$ is vital for symbolic derivations and proving properties of parameter estimates in statistics and ML.
2. **Left Inverses in OLS:** The normal equation solution $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$ is precisely the **left inverse** of the feature design matrix $X$ applied to $\mathbf{y}$.

---

## 5.7 Common mistakes

* **Forgetting to Transpose the Cofactor Matrix:** Computing $A^{-1} = \frac{1}{\det(A)} C$ instead of $\frac{1}{\det(A)} C^T$. The transpose is required for the row-on-column orthogonality cancellation!
* **Forgetting Checkerboard Signs:** Missing the negative signs on $C_{12}, C_{21}, C_{23}, C_{32}$ when going from minors to cofactors.
* **Assuming Non-Square Matrices Have No Inverses:** Tall matrices with independent columns have left inverses; wide matrices with independent rows have right inverses.

---

> 📖 **Navigation:** [← Previous: Part 04: Gaussian & Gauss-Jordan Elimination](./04_gaussian_and_gauss_jordan_elimination.md) | [🏠 Index](./README.md) | [Next: Part 06: Determinants →](./06_determinants.md)
