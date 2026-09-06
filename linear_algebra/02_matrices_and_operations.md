> 📖 **Navigation:** [← Previous: Part 01: Vectors & Vector Spaces](./01_vectors_and_vector_spaces.md) | [🏠 Index](./README.md) | [Next: Part 03: Matrix Inverses, Gauss-Jordan & Adjugate →](./03_matrix_inverses_and_gauss_jordan.md)

---

# PART 2 — MATRICES & MATRIX OPERATIONS

---

## 2.1 What is a Matrix? (Data Tables & Linear Transformations)

A matrix is a 2D rectangular array of numbers arranged into rows and columns. In Machine Learning, matrices serve two fundamental roles:

```
        DATA TABLE VIEW                          LINEAR OPERATOR VIEW
   (Rows = Samples, Cols = Features)             (Transforms Geometric Space)
   
       Age   Income  Tenure                                y
   ┌                        ┐                              │    A @ v
   │   25    50000     2    │  Sample 1                    │   ┌───►
   │   30    80000     5    │  Sample 2                    │  ╱
   │   45   120000    10    │  Sample 3                    │ ╱ v
   └                        ┘                              └───────► x
         Shape: (3, 3)
```

1. **The Data Matrix ($X$):** Storing $N$ observations with $p$ features: $X \in \mathbb{R}^{N \times p}$.
2. **The Linear Transformation ($A$):** A function that rotates, scales, or shears vectors in space: $\mathbf{y} = A\mathbf{x}$.

---

## 2.2 Matrix Shape & Dimensionality

* Matrix $A$ has shape $(M \times N)$ where:
  * $M$ = Number of **rows** (samples)
  * $N$ = Number of **columns** (features / dimensions)
* *Element notation:* $A_{ij}$ is the scalar in row $i$, column $j$.

---

## 2.3 Matrix Addition & Scalar Multiplication

### 1. Matrix Addition (Element-wise)
Matrices must have the **exact same shape**:

$$
\begin{bmatrix}
1 & 3 \\
2 & 4
\end{bmatrix}
+
\begin{bmatrix}
5 & 1 \\
0 & 2
\end{bmatrix} =
\begin{bmatrix}
1+5 & 3+1 \\
2+0 & 4+2
\end{bmatrix} =
\begin{bmatrix}
6 & 4 \\
2 & 6
\end{bmatrix}
$$

### 2. Scalar Multiplication

$$
2
\begin{bmatrix}
1 & 3 \\
2 & 4
\end{bmatrix} =
\begin{bmatrix}
2(1) & 2(3) \\
2(2) & 2(4)
\end{bmatrix} =
\begin{bmatrix}
2 & 6 \\
4 & 8
\end{bmatrix}
$$

---

## 2.4 Matrix Multiplication (Row-by-Column Deep Trace)

Matrix multiplication is **NOT** element-wise multiplication. It is the composition of linear transformations.

### Compatibility Rule
To multiply $A \times B$, the **number of columns in $A$ MUST equal the number of rows in $B$**:

$$
(M \times K) \times (K \times N) \implies (M \times N)
$$

```
               Matrix A (2x2)               Matrix B (2x2)               Matrix C (2x2)
             ┌                ┐           ┌                ┐           ┌                ┐
             │ [a11   a12]    │     ×     │  b11    [b12]  │     =     │   .      [C12] │
             │  a21   a22     │           │  b21    [b22]  │           │   .        .   │
             └                ┘           └                ┘           └                ┘
                                      C12 = (a11 * b12) + (a12 * b22)
```

### Complete Hand Calculation Example
Let:

$$
A =
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix},
\quad
B =
\begin{bmatrix}
5 & 6 \\
7 & 8
\end{bmatrix}
$$

1. **Calculate $C_{11}$ (Row 1 of $A \cdot$ Column 1 of $B$):**

$$
C_{11} = (1 \times 5) + (2 \times 7) = 5 + 14 = 19
$$

2. **Calculate $C_{12}$ (Row 1 of $A \cdot$ Column 2 of $B$):**

$$
C_{12} = (1 \times 6) + (2 \times 8) = 6 + 16 = 22
$$

3. **Calculate $C_{21}$ (Row 2 of $A \cdot$ Column 1 of $B$):**

$$
C_{21} = (3 \times 5) + (4 \times 7) = 15 + 28 = 43
$$

4. **Calculate $C_{22}$ (Row 2 of $A \cdot$ Column 2 of $B$):**

$$
C_{22} = (3 \times 6) + (4 \times 8) = 18 + 32 = 50
$$

**Final Result Matrix:**

$$
C = AB =
\begin{bmatrix}
19 & 22 \\
43 & 50
\end{bmatrix}
$$

### Machine Learning Connection: $X \mathbf{w} = \hat{\mathbf{y}}$
In linear regression, generating predictions for $N$ samples simultaneously is a single matrix-vector multiplication:

$$
\hat{\mathbf{y}} = X \mathbf{w} =
\begin{bmatrix}
1 & x_{11} & x_{12} \\
1 & x_{21} & x_{22} \\
1 & x_{31} & x_{32}
\end{bmatrix}
\begin{bmatrix}
w_0 \\
w_1 \\
w_2
\end{bmatrix} =
\begin{bmatrix}
w_0 + w_1 x_{11} + w_2 x_{12} \\
w_0 + w_1 x_{21} + w_2 x_{22} \\
w_0 + w_1 x_{31} + w_2 x_{32}
\end{bmatrix}
$$

* Shape check: $(3 \times 3) \times (3 \times 1) = (3 \times 1)$ predictions vector!

---

## 2.5 Matrix Transpose & Properties

The **transpose** $A^T$ flips a matrix over its diagonal, switching rows and columns:

$$
(A^T)_{ij} = A_{ji} \quad \quad \text{Shape: } (M \times N) \to (N \times M)
$$

### Hand Calculation Example

$$
A =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
\quad \implies \quad
A^T =
\begin{bmatrix}
1 & 4 \\
2 & 5 \\
3 & 6
\end{bmatrix}
$$

### Critical Transpose Properties in ML:
1. $(A^T)^T = A$
2. $(A + B)^T = A^T + B^T$
3. $(AB)^T = B^T A^T$ *(Order reverses! Essential for neural network backpropagation equations).*
4. $X^T X$ is **ALWAYS a square, symmetric matrix** ($p \times p$), representing unnormalized feature correlations.

---

## 2.6 Identity Matrix & Diagonal Matrices

The **Identity Matrix** $I$ is the matrix equivalent of the number $1$. It has $1$s along the main diagonal and $0$s elsewhere:

$$
I =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

### Property: $AI = IA = A$

$$
\begin{bmatrix}
3 & 7 \\
2 & 5
\end{bmatrix}
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix} =
\begin{bmatrix}
(3\times 1 + 7\times 0) & (3\times 0 + 7\times 1) \\
(2\times 1 + 5\times 0) & (2\times 0 + 5\times 1)
\end{bmatrix} =
\begin{bmatrix}
3 & 7 \\
2 & 5
\end{bmatrix}
$$

---

> 📖 **Navigation:** [← Previous: Part 01: Vectors & Vector Spaces](./01_vectors_and_vector_spaces.md) | [🏠 Index](./README.md) | [Next: Part 03: Matrix Inverses, Gauss-Jordan & Adjugate →](./03_matrix_inverses_and_gauss_jordan.md)
