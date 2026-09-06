> 📖 **Navigation:** [← Previous: Part 01: Vectors, Span & Vector Spaces](./01_vectors_and_vector_spaces.md) | [🏠 Index](./README.md) | [Next: Part 03: Matrix Inverses, Gauss-Jordan & Systems →](./03_matrix_inverses_and_gauss_jordan.md)

---

# PART 2 — MATRICES, MULTIPLICATION & TRANSFORMATIONS

A **matrix** is a 2D rectangular grid of numbers. In Machine Learning, matrices serve two foundational roles: storing datasets and executing geometric transformations.

---

## 2.1 The Two Fundamental Views of a Matrix

```
         1. THE DATA TABLE VIEW (X)                      2. THE LINEAR OPERATOR VIEW (A)
      (Rows = Samples, Cols = Features)                   (Transforms Space: y = Ax)
    
        Age   Income  Tenure                                      y
    ┌                        ┐                                    │       A @ v
    │   25    50000     2    │  Sample 1                          │      ┌───►
    │   30    80000     5    │  Sample 2                          │     ╱
    │   45   120000    10    │  Sample 3                          │    ╱ v
    └                        ┘                                    └───┴────────► x
         Shape: (n x d) = (3 x 3)
```

1. **The Data Matrix ($X \in \mathbb{R}^{n \times d}$):** Stores $n$ observations across $d$ features.
2. **The Linear Transformation ($A \in \mathbb{R}^{m \times d}$):** A mapping that takes an input vector $\mathbf{x} \in \mathbb{R}^d$ and transforms it into an output vector $\mathbf{y} \in \mathbb{R}^m$ via $\mathbf{y} = A\mathbf{x}$.

---

## 2.2 Matrix Addition & Scalar Multiplication

* **Matrix Addition:** Matrices must have identical dimensions $(m \times n)$. Add entry-by-entry:
  $$
  \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix} + \begin{bmatrix} 5 & 1 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 1+5 & 3+1 \\ 2+0 & 4+2 \end{bmatrix} = \begin{bmatrix} 6 & 4 \\ 2 & 6 \end{bmatrix}
  $$
* **Scalar Multiplication:** Multiplies every element by scalar $c$:
  $$
  2 \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix} = \begin{bmatrix} 2 & 6 \\ 4 & 8 \end{bmatrix}
  $$

---

## 2.3 Matrix-Vector Multiplication: Linear Combination of Columns

When an $m \times d$ matrix $A$ multiplies a $d \times 1$ vector $\mathbf{x}$, the result is a **linear combination of the columns of $A$**:

$$
A\mathbf{x} =
\begin{bmatrix} \mid & \mid & & \mid \\ \mathbf{a}_1 & \mathbf{a}_2 & \dots & \mathbf{a}_d \\ \mid & \mid & & \mid \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_d \end{bmatrix}
= x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \dots + x_d \mathbf{a}_d
$$

* **ML Role:** In linear regression, predictions for all $n$ samples are calculated in a single matrix-vector multiplication $\hat{\mathbf{y}} = X\mathbf{w}$.

---

## 2.4 Matrix Multiplication: The Three Foundational Interpretations

Multiplying two matrices $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$ produces $C = AB \in \mathbb{R}^{m \times n}$.

### 1. The Row-Column (Entry-by-Entry) View
Each entry $C_{ij}$ is the dot product of Row $i$ of matrix $A$ and Column $j$ of matrix $B$:
$$
C_{ij} = \sum_{r=1}^{k} A_{ir} B_{rj} = A_{i1}B_{1j} + A_{i2}B_{2j} + \dots + A_{ik}B_{kj}
$$

### 2. The Composition of Transformations View ($B$ first, then $A$)
Matrix multiplication represents the sequential composition of linear transformations:
$$
(AB)\mathbf{x} = A(B\mathbf{x})
$$
* **Order of Operations:** $B$ acts on $\mathbf{x}$ first, then $A$ acts on the resulting vector $B\mathbf{x}$.
* **Non-Commutativity:** In general, $AB \neq BA$! Rotating then shearing produces a completely different result than shearing then rotating.

### 3. The Column-by-Column View
If $B = [\mathbf{b}_1, \mathbf{b}_2, \dots, \mathbf{b}_n]$, then:
$$
AB = \begin{bmatrix} A\mathbf{b}_1 & A\mathbf{b}_2 & \dots & A\mathbf{b}_n \end{bmatrix}
$$
* *Every column in $AB$ is the transformation $A$ applied to the corresponding column of $B$!*

---

## 2.5 Matrix as a Geometric Linear Transformation

A $2 \times 2$ matrix $A$ transforms the standard 2D basis vectors $\mathbf{i} = [1, 0]^T$ and $\mathbf{j} = [0, 1]^T$:

```
     1. SCALING (s_x=2, s_y=0.5)      2. ROTATION (by 90 deg)         3. SHEAR (k=1.0)
             y                                y                               y
             │       ● (2, 0.5)               │                               │      ● (2, 1)
             │      ╱                         │      ● (-1, 1)                │     ╱│
             └───●─┴────────► x               └───●──┴────────► x             └───●──┴────────► x
               [ 2   0   ]                      [ 0  -1  ]                      [ 1   1  ]
               [ 0  0.5  ]                      [ 1   0  ]                      [ 0   1  ]
```

* **Scaling:** Stretches or compresses axes:
  $$
  A_{\text{scale}} = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}
  $$
* **Rotation (by angle $\theta$):**
  $$
  R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
  $$
* **Reflection (across x-axis):**
  $$
  A_{\text{reflect}} = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}
  $$
* **Projection (onto x-axis):**
  $$
  P_x = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}
  $$
* **Shear:** Slides rows parallel to an axis:
  $$
  A_{\text{shear}} = \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}
  $$

---

## 2.6 Matrix Transpose & Properties

The **transpose** $A^T$ flips a matrix over its main diagonal, turning rows into columns ($A_{ij}^T = A_{ji}$):

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \implies A^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}
$$

### Core Transpose Properties in ML:
1. $(A^T)^T = A$
2. $(A + B)^T = A^T + B^T$
3. $(AB)^T = B^T A^T$ *(Order reverses! Fundamental for neural network backpropagation)*.
4. **$X^T X$ is ALWAYS a square, symmetric matrix ($d \times d$),** representing the unnormalized feature correlation matrix in machine learning.

---

## 2.7 Identity Matrix & Diagonal Matrices

* **Identity Matrix ($I$):** The matrix equivalent of the number 1. $AI = IA = A$.
  $$
  I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
  $$
* **Diagonal Matrix ($D$):** Non-zero entries exist only on the main diagonal. Diagonal matrix multiplication simply scales each dimension independently.

---

## 2.8 Matrix Trace ($\text{Tr}(A)$) & The Cyclic Permutation Property

The **Trace** of a square matrix $A \in \mathbb{R}^{n \times n}$ is the sum of its main diagonal entries:

$$
\text{Tr}(A) = \sum_{i=1}^{n} A_{ii} = \sum_{i=1}^{n} \lambda_i
$$

### The Cyclic Property:
For matrices of compatible shapes:
$$
\text{Tr}(ABC) = \text{Tr}(BCA) = \text{Tr}(CAB)
$$
* *Caution:* $\text{Tr}(ABC) \neq \text{Tr}(BAC)$ in general (cyclic order must be preserved!).

---

## 2.9 Essential ML Matrix Operations: Frobenius Norm & Hadamard Product

* **Frobenius Norm ($\|A\|_F$):** Measures the overall energy or magnitude of a matrix:
  $$
  \|A\|_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} A_{ij}^2} = \sqrt{\text{Tr}(A^T A)}
  $$
  *(Used in Matrix Factorization loss functions: $\|X - W H\|_F^2$).*
* **Hadamard Product ($A \odot B$):** Element-wise multiplication of matrices with the same shape:
  $$
  (A \odot B)_{ij} = A_{ij} B_{ij}
  $$
  *(Used in LSTM/GRU gate activation masking: $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$).*

---

## Advanced / Optional — Do not study until Core track is complete

### A.1 Special Matrix Families in Linear Algebra
* **Orthogonal Matrix ($Q^T Q = Q Q^T = I$):** Preserves vector lengths ($\|Q\mathbf{x}\|_2 = \|\mathbf{x}\|_2$) and angles; $\det(Q) = \pm 1$.
* **Idempotent Matrix ($P^2 = P$):** Projecting a second time changes nothing; all eigenvalues $\lambda \in \{0, 1\}$.
* **Involutory Matrix ($A^2 = I$):** A matrix that is its own inverse ($A^{-1} = A$), such as reflection matrices.
* **Nilpotent Matrix ($N^k = 0$):** All eigenvalues are identically 0.
* **Skew-Symmetric Matrix ($A^T = -A$):** Diagonal entries are 0; all eigenvalues are purely imaginary.

### A.2 Spectral Norm & Nuclear Norm
* **Spectral Norm ($\|A\|_2 = \sigma_{\max}(A)$):** Maximum stretch factor (operator norm); used in GAN Spectral Normalization.
* **Nuclear Norm ($\|A\|_* = \sum \sigma_i$):** Sum of singular values; convex relaxation of rank used in Matrix Completion.

### A.3 Kronecker Product ($A \otimes B$)
Block matrix outer product where each entry of $A$ scales the entire matrix $B$:
$$
A \otimes B = \begin{bmatrix} a_{11} B & \dots & a_{1n} B \\ \vdots & \ddots & \vdots \\ a_{m1} B & \dots & a_{mn} B \end{bmatrix}
$$

---

> 📖 **Navigation:** [← Previous: Part 01: Vectors, Span & Vector Spaces](./01_vectors_and_vector_spaces.md) | [🏠 Index](./README.md) | [Next: Part 03: Matrix Inverses, Gauss-Jordan & Systems →](./03_matrix_inverses_and_gauss_jordan.md)
