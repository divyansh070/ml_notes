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

## 2.7 Matrix Trace ($\text{Tr}(A)$) & 5 Master Properties

The **Trace** of a square matrix $A \in \mathbb{R}^{n \times n}$ is the sum of its main diagonal elements:

$$
\text{Tr}(A) = \sum_{i=1}^{n} A_{ii} = A_{11} + A_{22} + \dots + A_{nn}
$$

### Hand Calculation Example
For matrix $A = \begin{bmatrix} 5 & 2 & 9 \\ 1 & 8 & 3 \\ 4 & 7 & 6 \end{bmatrix}$:

$$
\text{Tr}(A) = 5 + 8 + 6 = 19
$$

```
                         MATRIX TRACE PROPERTIES
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 1. Linearity:           Tr(A + B) = Tr(A) + Tr(B),  Tr(cA) = c Tr(A)   │
  │ 2. Transpose Invariance:Tr(A^T) = Tr(A)                                │
  │ 3. Cyclic Permutation:  Tr(ABC) = Tr(BCA) = Tr(CAB)                    │
  │ 4. Similarity Invariant:Tr(P^-1 A P) = Tr(A)                           │
  │ 5. Sum of Eigenvalues:  Tr(A) = λ_1 + λ_2 + ... + λ_n                  │
  │ 6. Frobenius Norm Link: ||A||_F^2 = Tr(A^T A) = Tr(A A^T)              │
  └────────────────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **The Cyclic Permutation Property:**
> For matrices of compatible dimensions:
> 
> $$
> \text{Tr}(ABCD) = \text{Tr}(BCDA) = \text{Tr}(CDAB) = \text{Tr}(DABC)
> $$
> 
> * **Exam Caution:** Cyclic order must be preserved! In general: $\text{Tr}(ABC) \neq \text{Tr}(BAC)$.
> * **Vector Dot Product as Trace:** For vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$:
> 
> $$
> \mathbf{u}^T \mathbf{v} = \text{Tr}(\mathbf{u}^T \mathbf{v}) = \text{Tr}(\mathbf{v} \mathbf{u}^T)
> $$

---

## 2.8 Special Matrix Families in Machine Learning

### 1. Symmetric & Skew-Symmetric Matrices
* **Symmetric Matrix:** $A^T = A \iff A_{ij} = A_{ji}$. (All eigenvalues are real; eigenvectors are orthogonal).
* **Skew-Symmetric (Anti-Symmetric) Matrix:** $A^T = -A \iff A_{ij} = -A_{ji}$. (Diagonal elements are all $0$; eigenvalues are purely imaginary).
* **Universal Decomposition Theorem:** Any square matrix $M$ can be uniquely decomposed into the sum of a symmetric and a skew-symmetric matrix:

$$
M = M_{\text{sym}} + M_{\text{skew}} = \frac{1}{2}(M + M^T) + \frac{1}{2}(M - M^T)
$$

### 2. Orthogonal Matrices ($Q$)
A square matrix $Q \in \mathbb{R}^{n \times n}$ is **Orthogonal** if its columns (and rows) form an orthonormal basis:

$$
Q^T Q = Q Q^T = I \quad \iff \quad Q^{-1} = Q^T
$$

* **Length Preservation (Isometry):** Multiplying any vector by an orthogonal matrix preserves its Euclidean length:

$$
\|Q\mathbf{x}\|_2^2 = (Q\mathbf{x})^T (Q\mathbf{x}) = \mathbf{x}^T Q^T Q \mathbf{x} = \mathbf{x}^T I \mathbf{x} = \|\mathbf{x}\|_2^2
$$

* **Angle Preservation:** $\langle Q\mathbf{u}, Q\mathbf{v} \rangle = \mathbf{u}^T Q^T Q \mathbf{v} = \mathbf{u}^T \mathbf{v}$.
* **Determinant:** $\det(Q) = \pm 1$ ($+1$ for pure rotations, $-1$ for reflections).

### 3. Idempotent Matrices (Projection Matrices)
A matrix $P$ is **Idempotent** if multiplying by itself leaves it unchanged:

$$
P^2 = P
$$

* **Eigenvalues:** If $P\mathbf{v} = \lambda\mathbf{v}$, then $P^2\mathbf{v} = \lambda^2\mathbf{v} = \lambda\mathbf{v} \implies \lambda^2 = \lambda \implies \lambda \in \{0, 1\}$.
* **ML Connection:** The OLS Hat matrix $H = X(X^T X)^{-1} X^T$ is symmetric and idempotent ($H^2 = H$), projecting data onto $\text{Col}(X)$.

### 4. Involutory Matrices
A matrix $A$ is **Involutory** if it is its own inverse:

$$
A^2 = I \quad \iff \quad A^{-1} = A
$$

* *Examples:* Reflection matrices (Householder reflections $H = I - 2\mathbf{u}\mathbf{u}^T$).

### 5. Nilpotent Matrices
A square matrix $N$ is **Nilpotent** if $N^k = 0$ for some positive integer $k$. All eigenvalues of a nilpotent matrix are identically $0$.

---

## 2.9 Matrix Norms (Frobenius, Spectral, Nuclear)

Just as vectors have lengths, matrices have norms measuring their overall magnitude or operator stretch.

| Matrix Norm | Mathematical Formula | Spectral Formulation | Machine Learning Role |
| :--- | :--- | :--- | :--- |
| **Frobenius Norm** ($\|A\|_F$) | $\sqrt{\sum_{i=1}^m \sum_{j=1}^n A_{ij}^2}$ | $\sqrt{\sum_{i=1}^{\min(m,n)} \sigma_i^2} = \sqrt{\text{Tr}(A^T A)}$ | Matrix generalization of Euclidean norm; used in matrix factorization loss ($\|X - WH\|_F^2$). |
| **Spectral Norm** ($\|A\|_2$) | $\max_{\mathbf{x} \neq \mathbf{0}} \frac{\|A\mathbf{x}\|_2}{\|\mathbf{x}\|_2}$ | $\sigma_{\max}(A) = \sqrt{\lambda_{\max}(A^T A)}$ | Maximum stretch factor; used in Spectral Normalization for GAN stability. |
| **Nuclear Norm** ($\|A\|_*$) | $\sum_{i=1}^{\min(m,n)} \sigma_i(A)$ | Sum of all singular values | Convex relaxation of matrix rank; used in **Matrix Completion** (Netflix Prize collaborative filtering). |

---

## 2.10 Specialized Matrix Products: Hadamard vs. Kronecker

### 1. Hadamard Product (Element-Wise Multiplication: $A \odot B$)
For matrices of the **exact same shape** $(M \times N)$:

$$
(A \odot B)_{ij} = A_{ij} \times B_{ij}
$$

* *ML Application:* Gating mechanisms in Recurrent Neural Networks (LSTM forget/input gates: $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$) and Dropout masks ($X \odot M$).

### 2. Kronecker Product (Block Outer Product: $A \otimes B$)
If $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{p \times q}$, then $A \otimes B \in \mathbb{R}^{mp \times nq}$:

$$
A \otimes B =
\begin{bmatrix}
a_{11} B & a_{12} B & \dots & a_{1n} B \\
a_{21} B & a_{22} B & \dots & a_{2n} B \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} B & a_{m2} B & \dots & a_{mn} B
\end{bmatrix}
$$

* *Key Properties:*
  * $\text{Tr}(A \otimes B) = \text{Tr}(A) \text{Tr}(B)$
  * $\det(A \otimes B) = (\det A)^p (\det B)^m$
  * $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$

---

> 📖 **Navigation:** [← Previous: Part 01: Vectors & Vector Spaces](./01_vectors_and_vector_spaces.md) | [🏠 Index](./README.md) | [Next: Part 03: Matrix Inverses, Gauss-Jordan & Adjugate →](./03_matrix_inverses_and_gauss_jordan.md)
