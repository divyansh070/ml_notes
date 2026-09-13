> 📖 **Navigation:** [← Previous: Part 01: Vectors & Vector Basics](./01_vectors_and_vector_spaces.md) | [🏠 Index](./README.md) | [Next: Part 03: Systems of Linear Equations (Ax = b) →](./03_systems_of_linear_equations.md)

---

# PART 2 — MATRICES & MATRIX OPERATIONS

In machine learning, a **matrix** is much more than a 2D table of numbers: it is simultaneously the storage structure for entire datasets and the computational engine for geometric transformations.

---

## 2.1 What is a Matrix? Dimensions, Elements, and Types

A matrix $A \in \mathbb{R}^{m \times n}$ is a rectangular array with $m$ rows and $n$ columns:

$$
A = \begin{bmatrix}
a_{11} & a_{12} & \dots & a_{1n} \\
a_{21} & a_{22} & \dots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \dots & a_{mn}
\end{bmatrix}
$$

* **Dimensions ($m \times n$):** $m$ rows (height) by $n$ columns (width).
* **Elements and Indexing ($a_{ij}$ or $A_{ij}$):** The entry at Row $i$, Column $j$.
* **Row Vector ($1 \times n$):** A matrix with a single row: $\mathbf{r} = \begin{bmatrix} r_1 & r_2 & \dots & r_n \end{bmatrix}$.
* **Column Vector ($m \times 1$):** A matrix with a single column: $\mathbf{c} = \begin{bmatrix} c_1 \\ c_2 \\ \vdots \\ c_m \end{bmatrix}$.

### The Two Complementary Roles of a Matrix in ML

```
       1. THE DATA MATRIX VIEW (X)                    2. THE LINEAR OPERATOR VIEW (A)
    (Rows = Samples, Cols = Features)                     (Transforms Vectors: y = Ax)
  
       Age   Income  Tenure                                    y
   ┌                        ┐                                  │       y = A @ x
   │   25    50000     2    │  Sample 1                        │      ┌───►
   │   30    80000     5    │  Sample 2                        │     ╱
   │   45   120000    10    │  Sample 3                        │    ╱ x
   └                        ┘                                  └───┴────────► x
        Shape: (n x d) = (3 x 3)
```

1. **The Data Table ($X \in \mathbb{R}^{n \times d}$):** Stores static data ($n$ observations across $d$ features).
2. **The Linear Operator ($A \in \mathbb{R}^{m \times n}$):** Acts as a function mapping an input vector $\mathbf{x} \in \mathbb{R}^n$ to an output vector $\mathbf{y} \in \mathbb{R}^m$ via $\mathbf{y} = A\mathbf{x}$.

---

## 2.2 Special Matrix Families

* **Square Matrix:** Number of rows equals number of columns ($m = n$).
* **Zero Matrix ($0$):** All entries are zero ($0_{ij} = 0$). Adding $0$ leaves any matrix unchanged: $A + 0 = A$.
* **Identity Matrix ($I$ or $I_n$):** A square matrix with $1$s on the main diagonal and $0$s elsewhere:
  $$
  I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad A I = I A = A
  $$
* **Diagonal Matrix ($D$):** Non-zero entries exist only along the main diagonal ($d_{ij} = 0$ for $i \neq j$):
  $$
  D = \operatorname{diag}(d_1, d_2, \dots, d_n) = \begin{bmatrix} d_1 & 0 & 0 \\ 0 & d_2 & 0 \\ 0 & 0 & d_3 \end{bmatrix}
  $$
  Multiplying by a diagonal matrix scales each coordinate independently.
* **Upper Triangular Matrix ($U$):** All entries strictly below the main diagonal are zero ($u_{ij} = 0$ for $i > j$).
* **Lower Triangular Matrix ($L$):** All entries strictly above the main diagonal are zero ($l_{ij} = 0$ for $i < j$).
* **Symmetric Matrix:** Equal to its own transpose ($A = A^T$, meaning $a_{ij} = a_{ji}$ for all $i, j$). Must be square.

---

## 2.3 Matrix Addition & Scalar Multiplication

* **Matrix Addition:** Matrices must have identical dimensions $(m \times n)$. Add corresponding entries:
  $$
  (A + B)_{ij} = A_{ij} + B_{ij}
  $$
* **Scalar Multiplication:** Multiplies every entry by scalar $c \in \mathbb{R}$:
  $$
  (c A)_{ij} = c A_{ij}
  $$

---

## 2.4 Matrix Transpose & Properties

The **transpose** $A^T$ swaps rows and columns ($A_{ij}^T = A_{ji}$), converting an $m \times n$ matrix into an $n \times m$ matrix:

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \implies A^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}
$$

### Master Transpose Properties in ML:
1. $(A^T)^T = A$
2. $(A + B)^T = A^T + B^T$
3. $(cA)^T = c A^T$
4. **Product Transpose Reversal:** $(AB)^T = B^T A^T$ *(Crucial for backpropagation in neural networks)*.
5. **The Covariance / Gram Matrix $X^T X$:** For any real data matrix $X \in \mathbb{R}^{n \times d}$, the matrix $X^T X$ is **always a square, symmetric matrix ($d \times d$)**:
   $$
   (X^T X)^T = X^T (X^T)^T = X^T X \quad \checkmark
   $$

---

## 2.5 Matrix-Vector Multiplication ($A\mathbf{x}$): Row vs. Column Views

Let $A \in \mathbb{R}^{m \times n}$ and $\mathbf{x} \in \mathbb{R}^n$. The product $\mathbf{y} = A\mathbf{x} \in \mathbb{R}^m$ can be evaluated from two fundamentally different perspectives:

### 1. The Row Interpretation (Dot Products of Rows with x)
Each entry $y_i$ of the output vector is the dot product of the $i$-th row of $A$ with $\mathbf{x}$:

$$
A\mathbf{x} = \begin{bmatrix} \text{---} & \mathbf{r}_1^T & \text{---} \\ \text{---} & \mathbf{r}_2^T & \text{---} \\ & \vdots & \\ \text{---} & \mathbf{r}_m^T & \text{---} \end{bmatrix} \mathbf{x}
= \begin{bmatrix} \mathbf{r}_1 \cdot \mathbf{x} \\ \mathbf{r}_2 \cdot \mathbf{x} \\ \vdots \\ \mathbf{r}_m \cdot \mathbf{x} \end{bmatrix}
$$

* *Computational Meaning:* How much does the input $\mathbf{x}$ align with each row constraint/sensor?

### 2. The Column Interpretation (Linear Combination of Columns of A)
Write $A$ as a sequence of column vectors $A = \begin{bmatrix} \mathbf{a}_1 & \mathbf{a}_2 & \dots & \mathbf{a}_n \end{bmatrix}$:

$$
A\mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \dots + x_n \mathbf{a}_n
$$

> [!IMPORTANT]
> **The Most Important Intuition in Machine Learning:**
> Multiplying a matrix by a vector produces a **linear combination of the columns of the matrix**, scaled by the components of the vector.
> * In linear regression $\hat{\mathbf{y}} = X\mathbf{w}$, the model prediction $\hat{\mathbf{y}}$ is **literally a linear combination of feature columns**, weighted by parameters $w_j$.
> * A prediction can ONLY exist if it lies within the **span of the feature columns**!

---

## 2.6 Matrix-Matrix Multiplication ($C = AB$)

To multiply $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$, the **inner dimensions must match** ($k = k$). The output is $C \in \mathbb{R}^{m \times n}$.

### 1. Row-Column (Entry-by-Entry) Formula:
$$
C_{ij} = \sum_{r=1}^{k} A_{ir} B_{rj} = (\text{Row } i \text{ of } A) \cdot (\text{Column } j \text{ of } B)
$$

### 2. Column-by-Column View:
$$
AB = A \begin{bmatrix} \mathbf{b}_1 & \mathbf{b}_2 & \dots & \mathbf{b}_n \end{bmatrix} = \begin{bmatrix} A\mathbf{b}_1 & A\mathbf{b}_2 & \dots & A\mathbf{b}_n \end{bmatrix}
$$
Every column of $C$ is the transformation $A$ applied to the corresponding column of $B$.

---

## 2.7 Matrix Multiplication is NOT Commutative ($AB \neq BA$)

Matrix multiplication represents the sequential composition of transformations. Applying transformation $B$ first, then $A$, does not produce the same result as applying $A$ first, then $B$.

### Concrete Numerical Proof:
Let:
$$
A = \begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix}, \quad B = \begin{bmatrix} 2 & 0 \\ 3 & 4 \end{bmatrix}
$$

### Compute $AB$:
$$
AB = \begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 3 & 4 \end{bmatrix} =
\begin{bmatrix} (1)(2) + (2)(3) & (1)(0) + (2)(4) \\ (0)(2) + (1)(3) & (0)(0) + (1)(4) \end{bmatrix} =
\begin{bmatrix} 8 & 8 \\ 3 & 4 \end{bmatrix}
$$

### Compute $BA$:
$$
BA = \begin{bmatrix} 2 & 0 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix} =
\begin{bmatrix} (2)(1) + (0)(0) & (2)(2) + (0)(1) \\ (3)(1) + (4)(0) & (3)(2) + (4)(1) \end{bmatrix} =
\begin{bmatrix} 2 & 4 \\ 3 & 10 \end{bmatrix}
$$

Clearly, $\begin{bmatrix} 8 & 8 \\ 3 & 4 \end{bmatrix} \neq \begin{bmatrix} 2 & 4 \\ 3 & 10 \end{bmatrix}$. **Therefore, $AB \neq BA$.**

---

## 2.8 Matrix Trace & Frobenius Norm

### 1. Matrix Trace ($\operatorname{Tr}(A)$)
The sum of the diagonal entries of a square matrix $A \in \mathbb{R}^{n \times n}$:

$$
\operatorname{Tr}(A) = \sum_{i=1}^{n} A_{ii}
$$

* **Cyclic Permutation Property:** For matrices of compatible dimensions:
  $$
  \operatorname{Tr}(ABC) = \operatorname{Tr}(BCA) = \operatorname{Tr}(CAB)
  $$
  *(Caution: Non-cyclic swaps such as $\operatorname{Tr}(BAC)$ are generally not equal!).*

### 2. Frobenius Norm ($\|A\|_F$)
The total energy / magnitude of a matrix (equivalent to the Euclidean length of the flattened matrix):

$$
\|A\|_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} A_{ij}^2} = \sqrt{\operatorname{Tr}(A^T A)}
$$

---

## 2.9 Complete Worked Numerical Example

Let:
$$
A = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} 5 \\ 6 \end{bmatrix}
$$

### 1. Matrix-Vector Product via Row View (Dot Products):
$$
A\mathbf{x} = \begin{bmatrix} (1)(5) + (3)(6) \\ (2)(5) + (4)(6) \end{bmatrix} = \begin{bmatrix} 5 + 18 \\ 10 + 24 \end{bmatrix} = \begin{bmatrix} 23 \\ 34 \end{bmatrix}
$$

### 2. Matrix-Vector Product via Column View (Linear Combination):
$$
A\mathbf{x} = 5 \begin{bmatrix} 1 \\ 2 \end{bmatrix} + 6 \begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 5 \\ 10 \end{bmatrix} + \begin{bmatrix} 18 \\ 24 \end{bmatrix} = \begin{bmatrix} 23 \\ 34 \end{bmatrix} \quad \checkmark
$$
*(Both interpretations yield the identical result, but the column view explains feature space geometry).*

### 3. Trace and Frobenius Norm of Matrix A:
* Trace: $\operatorname{Tr}(A) = 1 + 4 = 5$
* Frobenius Norm: $\|A\|_F = \sqrt{1^2 + 3^2 + 2^2 + 4^2} = \sqrt{1 + 9 + 4 + 16} = \sqrt{30} \approx 5.477$

---

## 2.10 Why this matters in ML

1. **Neural Network Layers:** A fully-connected layer is defined by $\mathbf{z} = W\mathbf{x} + \mathbf{b}$. Weight matrix $W$ linearly transforms the input representation $\mathbf{x}$.
2. **Batch Processing:** Passing a batch of $B$ inputs through a model bundles sample vectors into matrix $X \in \mathbb{R}^{B \times d}$. The layer computation $X W^T$ evaluates all $B$ predictions simultaneously via hardware-accelerated BLAS matrix multiplication.
3. **Loss Functions for Matrix Factorization:** In recommendation systems and Latent Semantic Analysis, we approximate a user-item rating matrix $R \approx U V^T$ by minimizing the Frobenius norm reconstruction loss:
   $$
   \mathcal{L} = \|R - U V^T\|_F^2
   $$

---

## 2.11 Common mistakes

* **Assuming Commutativity ($AB = BA$):** Never commute matrix products. In general $AB \neq BA$. Reversing the order changes the composition and often violates shape compatibility.
* **Mismatched Inner Dimensions:** To multiply $A(m \times k)$ by $B(p \times n)$, you must have $k = p$. Multiplying $(3 \times 2)$ by $(3 \times 2)$ is undefined for standard matrix multiplication.
* **Confusing Matrix Multiplication with Element-Wise (Hadamard) Multiplication:** Standard matrix multiplication $AB$ computes row-column dot products. Element-wise multiplication $A \odot B$ (or `A * B` in NumPy/PyTorch) multiplies identical coordinate positions and requires identical shapes.
* **Forgetting the Transpose Product Reversal:** Writing $(AB)^T = A^T B^T$. The correct identity is $(AB)^T = B^T A^T$.

---

> 📖 **Navigation:** [← Previous: Part 01: Vectors & Vector Basics](./01_vectors_and_vector_spaces.md) | [🏠 Index](./README.md) | [Next: Part 03: Systems of Linear Equations (Ax = b) →](./03_systems_of_linear_equations.md)
