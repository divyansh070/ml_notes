# Linear Algebra & Mathematics for Machine Learning & Data Science
### Visual Intuition • Hand Calculations • Step-by-Step Derivations • ML Applications

> **How to Use This Document:**
> This document is designed for **active paper-and-pencil revision** before technical interviews, online assessments (OAs), and machine learning exams.
> * Every core concept begins with **geometric intuition**.
> * Every formula is followed by a **complete step-by-step hand calculation** on small numbers ($2\times 2$ matrices and 2D vectors).
> * The focus is on **understanding derivations**, not blindly memorizing definitions or running Python libraries.

---

## Table of Contents
1. [PART 1 — VECTORS & VECTOR SPACES](#part-1-vectors-vector-spaces)
   - [1.1 Scalars vs. Vectors (What They Represent in ML)](#11-scalars-vs-vectors-what-they-represent-in-ml)
   - [1.2 Vector Addition (Geometric & Algebraic)](#12-vector-addition-geometric-algebraic)
   - [1.3 Scalar Multiplication (Scaling Space)](#13-scalar-multiplication-scaling-space)
   - [1.4 Dot Product & Angle Derivation](#14-dot-product-angle-derivation)
   - [1.5 Vector Norms (L1, L2, & Euclidean Distance)](#15-vector-norms-l1-l2-euclidean-distance)
   - [1.6 Cosine Similarity & Embeddings](#16-cosine-similarity-embeddings)
2. [PART 2 — MATRICES & MATRIX OPERATIONS](#part-2-matrices-matrix-operations)
   - [2.1 What is a Matrix? (Data Tables & Linear Transformations)](#21-what-is-a-matrix-data-tables-linear-transformations)
   - [2.2 Matrix Shape & Dimensionality](#22-matrix-shape-dimensionality)
   - [2.3 Matrix Addition & Scalar Multiplication](#23-matrix-addition-scalar-multiplication)
   - [2.4 Matrix Multiplication (Row-by-Column Deep Trace)](#24-matrix-multiplication-row-by-column-deep-trace)
   - [2.5 Matrix Transpose & Properties](#25-matrix-transpose-properties)
   - [2.6 Identity Matrix & Diagonal Matrices](#26-identity-matrix-diagonal-matrices)
   - [2.7 Matrix Inverse (2x2 Hand Derivation & Invertibility)](#27-matrix-inverse-2x2-hand-derivation-invertibility)
3. [PART 3 — DETERMINANTS](#part-3-determinants)
   - [3.1 The 2x2 Determinant Formula](#31-the-2x2-determinant-formula)
   - [3.2 Geometric Meaning: Area & Volume Scaling](#32-geometric-meaning-area-volume-scaling)
   - [3.3 Why Determinants Matter in ML (Invertibility & Singularities)](#33-why-determinants-matter-in-ml-invertibility-singularities)
4. [PART 4 — SYSTEMS OF LINEAR EQUATIONS ($Ax = b$)](#part-4-systems-of-linear-equations-ax-b)
   - [4.1 Simultaneous Equations to Matrix Form](#41-simultaneous-equations-to-matrix-form)
   - [4.2 Solving by Elimination & Matrix Inversion](#42-solving-by-elimination-matrix-inversion)
   - [4.3 The Three Solution Scenarios (Unique, None, Infinite)](#43-the-three-solution-scenarios-unique-none-infinite)
5. [PART 5 — LINEAR INDEPENDENCE & MATRIX RANK](#part-5-linear-independence-matrix-rank)
   - [5.1 Linear Combinations & Span](#51-linear-combinations-span)
   - [5.2 Linear Dependence vs. Independence (Hand Check)](#52-linear-dependence-vs-independence-hand-check)
   - [5.3 Matrix Rank (Column Rank & Row Rank)](#53-matrix-rank-column-rank-row-rank)
   - [5.4 Connection to Multicollinearity & Singular Models](#54-connection-to-multicollinearity-singular-models)
6. [PART 6 — EIGENVALUES & EIGENVECTORS (13-STEP MASTER DERIVATION)](#part-6-eigenvalues-eigenvectors-13-step-master-derivation)
   - [6.1 The Fundamental Equation: $Av = \lambda v$](#61-the-fundamental-equation-av-lambda-v)
   - [6.2 Complete 13-Step Hand Calculation on a 2x2 Matrix](#62-complete-13-step-hand-calculation-on-a-2x2-matrix)
   - [6.3 Geometric Meaning of Eigenvectors & Eigenvalues](#63-geometric-meaning-of-eigenvectors-eigenvalues)
7. [PART 7 — EIGENVALUES IN PRINCIPAL COMPONENT ANALYSIS (PCA)](#part-7-eigenvalues-in-principal-component-analysis-pca)
   - [7.1 The Chain of Logic: Data $\to$ Covariance $\to$ Eigenvalues $\to$ Components](#71-the-chain-of-logic-data-to-covariance-to-eigenvalues-to-components)
   - [7.2 Why Eigenvectors Maximize Variance](#72-why-eigenvectors-maximize-variance)
8. [PART 8 — COVARIANCE MATRIX: COMPLETE HAND CALCULATION](#part-8-covariance-matrix-complete-hand-calculation)
   - [8.1 Variance vs. Covariance Formulas](#81-variance-vs-covariance-formulas)
   - [8.2 Complete Step-by-Step Hand Calculation](#82-complete-step-by-step-hand-calculation)
   - [8.3 The Symmetry & Quadratic Structure of $\Sigma$](#83-the-symmetry-quadratic-structure-of-sigma)
   - [8.4 Sample ($n-1$) vs. Population ($N$) Bessel's Correction](#84-sample-n-1-vs-population-n-bessels-correction)
9. [PART 9 — COMPLETE PCA WALKTHROUGH FROM SCRATCH](#part-9-complete-pca-walkthrough-from-scratch)
   - [9.1 The 6-Step End-to-End PCA Algorithm](#91-the-6-step-end-to-end-pca-algorithm)
   - [9.2 Step-by-Step Numerical Hand Walkthrough (2D to 1D)](#92-step-by-step-numerical-hand-walkthrough-2d-to-1d)
   - [9.3 Calculating Explained Variance Ratio](#93-calculating-explained-variance-ratio)
10. [PART 10 — SINGULAR VALUE DECOMPOSITION (SVD)](#part-10-singular-value-decomposition-svd)
    - [10.1 The SVD Equation: $A = U \Sigma V^T$](#101-the-svd-equation-a-u-sigma-vt)
    - [10.2 Geometric Interpretation of $U, \Sigma, V^T$](#102-geometric-interpretation-of-u-sigma-vt)
    - [10.3 Relationship Between SVD and PCA](#103-relationship-between-svd-and-pca)
11. [PART 11 — ORTHOGONALITY & ORTHONORMAL BASES](#part-11-orthogonality-orthonormal-bases)
    - [11.1 Orthogonal Vectors ($u \cdot v = 0$)](#111-orthogonal-vectors-u-cdot-v-0)
    - [11.2 Orthonormal Bases & Unit Length](#112-orthonormal-bases-unit-length)
    - [11.3 Orthogonal Matrices ($Q^T Q = I$)](#113-orthogonal-matrices-qt-q-i)
12. [PART 12 — VECTOR PROJECTIONS](#part-12-vector-projections)
    - [12.1 Derivation of the Scalar & Vector Projection Formula](#121-derivation-of-the-scalar-vector-projection-formula)
    - [12.2 Step-by-Step Numerical Hand Calculation](#122-step-by-step-numerical-hand-calculation)
    - [12.3 Projection Matrices & Subspaces](#123-projection-matrices-subspaces)
13. [PART 13 — LINEAR REGRESSION: MATRIX MATHEMATICS](#part-13-linear-regression-matrix-mathematics)
    - [13.1 The Matrix System: $y = Xw + \epsilon$](#131-the-matrix-system-y-xw-epsilon)
    - [13.2 Minimizing the Sum of Squared Residuals (Least Squares)](#132-minimizing-the-sum-of-squared-residuals-least-squares)
    - [13.3 Derivation of the Normal Equation ($X^T X w = X^T y$)](#133-derivation-of-the-normal-equation-xt-x-w-xt-y)
    - [13.4 Why $(X^T X)^{-1}$ Fails in Practice (Condition Number & QR Solvers)](#134-why-xt-x-1-fails-in-practice-condition-number-qr-solvers)
14. [PART 14 — GRADIENTS & DERIVATIVES FOR OPTIMIZATION](#part-14-gradients-derivatives-for-optimization)
    - [14.1 1D Derivatives & Slope of Tangent Line](#141-1d-derivatives-slope-of-tangent-line)
    - [14.2 Partial Derivatives (Multivariable Functions)](#142-partial-derivatives-multivariable-functions)
    - [14.3 The Gradient Vector (Direction of Steepest Ascent)](#143-the-gradient-vector-direction-of-steepest-ascent)
    - [14.4 Gradient Descent: 3-Step Numerical Hand Trace](#144-gradient-descent-3-step-numerical-hand-trace)
15. [PART 15 — THE CHAIN RULE & BACKPROPAGATION](#part-15-the-chain-rule-backpropagation)
    - [15.1 Single-Variable & Multivariable Chain Rule](#151-single-variable-multivariable-chain-rule)
    - [15.2 Hand Trace on a Nested Function](#152-hand-trace-on-a-nested-function)
    - [15.3 Neural Network Computational Graphs](#153-neural-network-computational-graphs)
16. [PART 16 — DISTANCE & SIMILARITY METRICS](#part-16-distance-similarity-metrics)
    - [16.1 Euclidean Distance ($L_2$) vs. Manhattan Distance ($L_1$)](#161-euclidean-distance-l_2-vs-manhattan-distance-l_1)
    - [16.2 Cosine Distance vs. Cosine Similarity](#162-cosine-distance-vs-cosine-similarity)
    - [16.3 Comparison Table & Real-World Selection Matrix](#163-comparison-table-real-world-selection-matrix)
17. [PART 17 — REGULARIZATION MATHEMATICS (L1 vs. L2)](#part-17-regularization-mathematics-l1-vs-l2)
    - [17.1 The Constrained Optimization Formulation](#171-the-constrained-optimization-formulation)
    - [17.2 Mathematical Geometry: Why L1 Creates Exact Zeros (Sparsity)](#172-mathematical-geometry-why-l1-creates-exact-zeros-sparsity)
    - [17.3 Mathematical Geometry: Why L2 Shrinks Smoothly (Weight Decay)](#173-mathematical-geometry-why-l2-shrinks-smoothly-weight-decay)
18. [PART 18 — ENTROPY & INFORMATION GAIN MATHEMATICS](#part-18-entropy-information-gain-mathematics)
    - [18.1 Shannon Entropy Formula: $H(X) = -\sum p_i \log_2 p_i$](#181-shannon-entropy-formula-hx--sum-p_i-log_2-p_i)
    - [18.2 Hand Calculation: 50/50 Split vs. Pure Split](#182-hand-calculation-5050-split-vs-pure-split)
    - [18.3 Information Gain & Gini Impurity Hand Trace](#183-information-gain-gini-impurity-hand-trace)
19. [PART 19 — ML MATHEMATICS ROADMAP TABLE](#part-19-ml-mathematics-roadmap-table)
20. [PART 20 — "WHAT I SHOULD BE ABLE TO DO ON PAPER" CHECKLIST](#part-20-what-i-should-be-able-to-do-on-paper-checklist)
---

### 🚀 Advanced Topics Continued in Part 2:
* **[PART 21 — 40 ESSENTIAL TECHNICAL INTERVIEW QUESTIONS](LINEAR_ALGEBRA_AND_MATH_PART_2.md#part-21--40-essential-technical-interview-questions)**
* **[PART 22 — FOUR FUNDAMENTAL SUBSPACES](LINEAR_ALGEBRA_AND_MATH_PART_2.md#part-22--four-fundamental-subspaces)**
* **[PART 23 — GRAM-SCHMIDT ORTHOGONALIZATION & QR DECOMPOSITION](LINEAR_ALGEBRA_AND_MATH_PART_2.md#part-23--gram-schmidt-orthogonalization--qr-decomposition)**
* **[PART 24 — POSITIVE DEFINITE & POSITIVE SEMIDEFINITE MATRICES](LINEAR_ALGEBRA_AND_MATH_PART_2.md#part-24--positive-definite--positive-semidefinite-matrices)**
* **[PART 25 — MOORE-PENROSE PSEUDOINVERSE](LINEAR_ALGEBRA_AND_MATH_PART_2.md#part-25--moore-penrose-pseudoinverse)**
* **[PART 26 — LINEAR TRANSFORMATIONS](LINEAR_ALGEBRA_AND_MATH_PART_2.md#part-26--linear-transformations)**
* **[FINAL SECTION — ONE-PAGE MASTER FORMULA CHEAT SHEET](LINEAR_ALGEBRA_AND_MATH_PART_2.md#final-section--one-page-master-formula-cheat-sheet)**
27. [FINAL SECTION — ONE-PAGE MASTER FORMULA CHEAT SHEET](#final-section-one-page-master-formula-cheat-sheet)

---

# PART 1 — VECTORS & VECTOR SPACES

---

## 1.1 Scalars vs. Vectors (What They Represent in ML)

```
        SCALAR                           VECTOR
   (Single Magnitude)             (Magnitude + Direction)
          3.5                                 y
                                              │      x = [2, 3]^T
                                            3 ┼──────●
                                              │      │
                                              │      │
                                              └──────┴──── x
                                                     2
```

* **Scalar:** A single real number (e.g., $c \in \mathbb{R}$, like a learning rate $\alpha = 0.01$, a loss value $\mathcal{L} = 0.45$, or temperature $T = 25.0$).
* **Vector:** An ordered 1D array of numbers representing a point or direction in space.
  * In Machine Learning, a **feature vector** represents a single sample/data point across $p$ features:

$$
\mathbf{x} =
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_p
\end{bmatrix}
\in \mathbb{R}^p
$$

  * *Example:* A house with 2 bedrooms and 3 bathrooms is represented as the vector:

$$
\mathbf{x} =
\begin{bmatrix}
2 \\
3
\end{bmatrix}
$$

---

## 1.2 Vector Addition (Geometric & Algebraic)

### Algebraic Definition
To add two vectors of the same dimension, add their corresponding components:

$$
\mathbf{a} + \mathbf{b} =
\begin{bmatrix}
a_1 \\
a_2
\end{bmatrix}
+
\begin{bmatrix}
b_1 \\
b_2
\end{bmatrix} =
\begin{bmatrix}
a_1 + b_1 \\
a_2 + b_2
\end{bmatrix}
$$

### Hand Calculation Example
Let $\mathbf{a} = [2, 3]^T$ and $\mathbf{b} = [4, 1]^T$:

$$
\mathbf{a} + \mathbf{b} =
\begin{bmatrix}
2 \\
3
\end{bmatrix}
+
\begin{bmatrix}
4 \\
1
\end{bmatrix} =
\begin{bmatrix}
2 + 4 \\
3 + 1
\end{bmatrix} =
\begin{bmatrix}
6 \\
4
\end{bmatrix}
$$

### Geometric Intuition (Tip-to-Tail Rule)
Place the tail of vector $\mathbf{b}$ at the tip of vector $\mathbf{a}$. The resulting vector $\mathbf{a} + \mathbf{b}$ runs from the origin $(0,0)$ to the final tip $(6,4)$, forming the diagonal of a parallelogram.

* **ML Connection:** Adding a bias vector $\mathbf{b}$ to linear logits: $\mathbf{z} = W\mathbf{x} + \mathbf{b}$ shifts the activation hyperplane in space.

---

## 1.3 Scalar Multiplication (Scaling Space)

### Algebraic Definition
Multiplying a vector by a scalar scales every component by that scalar:

$$
c \mathbf{x} = c
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix} =
\begin{bmatrix}
c x_1 \\
c x_2
\end{bmatrix}
$$

### Hand Calculation Example
Let $c = 3$ and $\mathbf{x} = [2, 4]^T$:

$$
3 \mathbf{x} = 3
\begin{bmatrix}
2 \\
4
\end{bmatrix} =
\begin{bmatrix}
3 \times 2 \\
3 \times 4
\end{bmatrix} =
\begin{bmatrix}
6 \\
12
\end{bmatrix}
$$

### Geometric Intuition
* If $c \gt 1$: Stretches the vector in the same direction.
* If $0 \lt c \lt 1$: Shrinks the vector in the same direction.
* If $c \lt 0$: Reverses the vector's direction by $180^\circ$ and scales its length by $|c|$.
* **ML Connection:** Gradient descent step update: $\mathbf{w}_{\text{new}} = \mathbf{w} - \alpha \nabla \mathcal{L}$. The scalar learning rate $\alpha$ scales the magnitude of the gradient vector step.

---

## 1.4 Dot Product & Angle Derivation

The **Dot Product** (inner product) takes two vectors and produces a single **scalar**.

```
                           THE DOT PRODUCT GEOMETRY
                                     b
                                    ╱
                                   ╱ θ
                                  ╱───────► a
                                  |-------|
                             ||b|| cos(θ)  (Projection of b on a)
```

### 1. Algebraic Definition

$$
\mathbf{a} \cdot \mathbf{b} = \mathbf{a}^T \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \dots + a_n b_n
$$

### Hand Calculation Example
Let $\mathbf{a} = [2, 3]^T$ and $\mathbf{b} = [4, 1]^T$:

$$
\mathbf{a} \cdot \mathbf{b} = (2 \times 4) + (3 \times 1) = 8 + 3 = 11
$$

### 2. Geometric Definition & Angle Formula

$$
\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|_2 \|\mathbf{b}\|_2 \cos\theta
$$

Where:
* $\|\mathbf{a}\|_2$ and $\|\mathbf{b}\|_2$ are the geometric lengths (L2 norms) of vectors $\mathbf{a}$ and $\mathbf{b}$.
* $\theta$ is the angle between them ($0^\circ \le \theta \le 180^\circ$).

### Step-by-Step Angle Calculation:
1. Compute lengths:

$$
\|\mathbf{a}\|_2 = \sqrt{2^2 + 3^2} = \sqrt{4 + 9} = \sqrt{13} \approx 3.606
$$

$$
\|\mathbf{b}\|_2 = \sqrt{4^2 + 1^2} = \sqrt{16 + 1} = \sqrt{17} \approx 4.123
$$

2. Solve for $\cos\theta$:

$$
\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2} = \frac{11}{\sqrt{13} \times \sqrt{17}} = \frac{11}{\sqrt{221}} = \frac{11}{14.866} \approx 0.740
$$

3. Angle: $\theta = \arccos(0.740) \approx 42.3^\circ$.

### Orthogonality Rule (Perpendicular Vectors)
If $\theta = 90^\circ$, then $\cos(90^\circ) = 0$. Therefore:

$$
\mathbf{a} \perp \mathbf{b} \iff \mathbf{a} \cdot \mathbf{b} = 0
$$

* *Test:* Let $\mathbf{u} = [2, 3]^T$ and $\mathbf{v} = [-3, 2]^T$:

$$
\mathbf{u} \cdot \mathbf{v} = 2(-3) + 3(2) = -6 + 6 = 0 \quad \text{(Orthogonal!)}
$$

---

## 1.5 Vector Norms (L1, L2, & Euclidean Distance)

A **norm** is a mathematical function that measures the length or magnitude of a vector.

```
       L2 NORM (Euclidean)                    L1 NORM (Manhattan)
     Direct straight-line distance           Grid / City block distance
               y                                       y
               │      ● (x1, y1)                       │      ●
               │     ╱                                 │     │
               │    ╱ sqrt(dx^2 + dy^2)                │     │ dy
               │   ╱                                   │     │
               └──●───────── x                         └──●──┴────── x
                  (x0, y0)                                 dx
```

### 1. $L_2$ Norm (Euclidean Norm / Length)
Derived directly from the Pythagorean theorem ($c = \sqrt{a^2 + b^2}$):

$$
\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2} = \sqrt{x_1^2 + x_2^2 + \dots + x_n^2}
$$

* *Hand Calculation:* For $\mathbf{v} = [3, -4]^T$:

$$
\|\mathbf{v}\|_2 = \sqrt{3^2 + (-4)^2} = \sqrt{9 + 16} = \sqrt{25} = 5
$$

### 2. $L_1$ Norm (Manhattan / Taxicab Norm)
Sum of absolute values:

$$
\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i| = |x_1| + |x_2| + \dots + |x_n|
$$

* *Hand Calculation:* For $\mathbf{v} = [3, -4]^T$:

$$
\|\mathbf{v}\|_1 = |3| + |-4| = 3 + 4 = 7
$$

### 3. Euclidean Distance between Two Points
The straight-line distance between vectors $\mathbf{p}$ and $\mathbf{q}$:

$$
d(\mathbf{p}, \mathbf{q}) = \|\mathbf{p} - \mathbf{q}\|_2 = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}
$$

* *Hand Calculation:* Let $\mathbf{p} = [1, 2]^T$ and $\mathbf{q} = [4, 6]^T$:

$$
\mathbf{p} - \mathbf{q} =
\begin{bmatrix}
1 - 4 \\
2 - 6
\end{bmatrix} =
\begin{bmatrix}
-3 \\
-4
\end{bmatrix}
$$

$$
d(\mathbf{p}, \mathbf{q}) = \sqrt{(-3)^2 + (-4)^2} = \sqrt{9 + 16} = \sqrt{25} = 5
$$

---

## 1.6 Cosine Similarity & Embeddings

**Cosine Similarity** measures the alignment (angle) between two vectors regardless of their scale/magnitude:

$$
\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \cos\theta
$$

* **Range:** $[-1.0, +1.0]$
  * $+1.0 \implies$ Vectors point in the exact same direction ($\theta = 0^\circ$).
  * $0.0 \implies$ The vectors are orthogonal ($90^\circ$ apart), so they have **no directional alignment**.
  * $-1.0 \implies$ Vectors point in opposite directions ($\theta = 180^\circ$).

### Step-by-Step Hand Calculation
Compare two document embedding vectors: $\mathbf{u} = [1, 2]^T$ (Document A) and $\mathbf{v} = [2, 4]^T$ (Document B):
1. Dot product: $\mathbf{u} \cdot \mathbf{v} = (1 \times 2) + (2 \times 4) = 2 + 8 = 10$.
2. Norm of $\mathbf{u}$: $\|\mathbf{u}\|_2 = \sqrt{1^2 + 2^2} = \sqrt{5}$.
3. Norm of $\mathbf{v}$: $\|\mathbf{v}\|_2 = \sqrt{2^2 + 4^2} = \sqrt{4 + 16} = \sqrt{20} = 2\sqrt{5}$.
4. Compute similarity:

$$
\text{Cosine Similarity} = \frac{10}{\sqrt{5} \times 2\sqrt{5}} = \frac{10}{2 \times 5} = \frac{10}{10} = 1.0
$$

*Even though Document B is twice as long as Document A, their cosine similarity is 1.0 because their topic direction is identical!*

* **Why Cosine Similarity is Preferred in NLP & LLMs:** In high-dimensional text embeddings (e.g. Word2Vec, BERT, OpenAI embeddings), document length affects vector magnitude but not semantic meaning. Cosine similarity normalizes out length differences.

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

## 2.7 Matrix Inverse (2x2 Hand Derivation & Invertibility)

The **inverse** $A^{-1}$ of a square matrix $A$ is the matrix that undoes the transformation of $A$:

$$
A A^{-1} = A^{-1} A = I
$$

### 2x2 Inverse Formula:
For:

$$
A =
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
$$

$$
A^{-1} = \frac{1}{\det(A)}
\begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}
= \frac{1}{ad - bc}
\begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}
$$

### Step-by-Step Hand Calculation
Let:

$$
A =
\begin{bmatrix}
4 & 7 \\
2 & 6
\end{bmatrix}
$$

1. Calculate determinant:

$$
\det(A) = (4 \times 6) - (7 \times 2) = 24 - 14 = 10
$$

2. Swap diagonal elements ($4 \leftrightarrow 6$) and negate off-diagonal elements ($7 \to -7, 2 \to -2$):

$$
\text{Adj}(A) =
\begin{bmatrix}
6 & -7 \\
-2 & 4
\end{bmatrix}
$$

3. Multiply by $\frac{1}{\det(A)}$:

$$
A^{-1} = \frac{1}{10}
\begin{bmatrix}
6 & -7 \\
-2 & 4
\end{bmatrix} =
\begin{bmatrix}
0.6 & -0.7 \\
-0.2 & 0.4
\end{bmatrix}
$$

4. **Verify by multiplication ($A A^{-1} = I$):**

$$
A A^{-1} =
\begin{bmatrix}
4 & 7 \\
2 & 6
\end{bmatrix}
\begin{bmatrix}
0.6 & -0.7 \\
-0.2 & 0.4
\end{bmatrix} =
\begin{bmatrix}
4(0.6)+7(-0.2) & 4(-0.7)+7(0.4) \\
2(0.6)+6(-0.2) & 2(-0.7)+6(0.4)
\end{bmatrix} =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

### When Does an Inverse NOT Exist?
* A matrix is **singular (non-invertible)** if and only if **$\det(A) = 0$**.
* If $\det(A) = 0$, the transformation squashes space into a lower dimension (e.g. 2D plane into a 1D line), destroying information. You cannot divide by zero ($\frac{1}{0}$), so no inverse exists.

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

# PART 4 — SYSTEMS OF LINEAR EQUATIONS ($Ax = b$)

---

## 4.1 Simultaneous Equations to Matrix Form

Consider a system of 2 linear equations with 2 unknowns:

$$
\begin{aligned}
2x + y &= 5 \\
x - y &= 1
\end{aligned}
$$

### Converting to Matrix-Vector Form:

$$
\begin{bmatrix}
2 & 1 \\
1 & -1
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix} =
\begin{bmatrix}
5 \\
1
\end{bmatrix}
\quad \iff \quad A \mathbf{x} = \mathbf{b}
$$

Where:
* The **Coefficient Matrix** $A$:

$$
A =
\begin{bmatrix}
2 & 1 \\
1 & -1
\end{bmatrix}
$$

* $\mathbf{x} = [x, y]^T$ is the **Vector of Unknowns**.
* $\mathbf{b} = [5, 1]^T$ is the **Output / Target Vector**.

---

## 4.2 Solving by Elimination & Matrix Inversion

### Method 1: Algebraic Elimination (Fast on Paper)
1. Add equation (1) and equation (2):

$$
(2x + y) + (x - y) = 5 + 1 \implies 3x = 6 \implies x = 2
$$

2. Substitute $x = 2$ back into equation (2):

$$
2 - y = 1 \implies y = 1
$$

3. Solution: $\mathbf{x} = [2, 1]^T$.

### Method 2: Matrix Inversion ($\mathbf{x} = A^{-1} \mathbf{b}$)
1. Compute $\det(A)$:

$$
\det(A) = (2 \times -1) - (1 \times 1) = -2 - 1 = -3
$$

2. Compute $A^{-1}$:

$$
A^{-1} = \frac{1}{-3}
\begin{bmatrix}
-1 & -1 \\
-1 & 2
\end{bmatrix} =
\begin{bmatrix}
1/3 & 1/3 \\
1/3 & -2/3
\end{bmatrix}
$$

3. Multiply $A^{-1} \mathbf{b}$:

$$
\mathbf{x} =
\begin{bmatrix}
1/3 & 1/3 \\
1/3 & -2/3
\end{bmatrix}
\begin{bmatrix}
5 \\
1
\end{bmatrix} =
\begin{bmatrix}
(5/3 + 1/3) \\
(5/3 - 2/3)
\end{bmatrix} =
\begin{bmatrix}
6/3 \\
3/3
\end{bmatrix} =
\begin{bmatrix}
2 \\
1
\end{bmatrix}
$$

---

## 4.3 The Three Solution Scenarios (Unique, None, Infinite)

```
      UNIQUE SOLUTION (det != 0)           NO SOLUTION (Parallel)         INFINITE SOLUTIONS (Same Line)
               y                                     y                                    y
               │      ╲   ╱                          │    ╱   ╱                           │    ╱ (Lines
               │       ╲ ╱                           │   ╱   ╱                            │   ╱   overlap
               │        ● Intersection               │  ╱   ╱  No intersection            │  ╱    completely)
               └────────┼─────► x                    └─┼───┼────────► x                   └─┼────────► x
```

1. **Unique Solution ($\det(A) \neq 0$):** The two lines cross at exactly one coordinate point. Matrix $A$ has full rank.
2. **No Solution ($\det(A) = 0$, Parallel Lines):** E.g., $x + y = 2$ and $x + y = 5$. The lines never intersect; the equations contradict each other.
3. **Infinitely Many Solutions ($\det(A) = 0$, Dependent Lines):** E.g., $x + y = 2$ and $2x + 2y = 4$. The two equations describe the exact same line.

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

# PART 6 — EIGENVALUES & EIGENVECTORS (13-STEP MASTER DERIVATION)

---

## 6.1 The Fundamental Equation: $Av = \lambda v$

When a matrix $A$ multiplies most vectors $\mathbf{x}$, it both **rotates** and **stretches** them.

However, for square matrices, there can exist special directions $\mathbf{v}$ where the matrix **ONLY STRETCHES or SHRINKS the vector, without changing its direction at all!**

$$
A \mathbf{v} = \lambda \mathbf{v}
$$

```
        TYPICAL VECTOR (Rotated & Stretched)            EIGENVECTOR (Only Stretched by Lambda!)
                 y                                                y
                 │        A @ x                                   │            A @ v = lambda * v
                 │       ╱                                        │           ╱
                 │      ╱                                         │          ╱
                 │     ● ◄── Rotated                              │         ●
                 │    ╱                                           │        ╱
                 │   ● x                                          │       ● v (Same direction!)
                 └───┴────────► x                                 └───────┴────────► x
```

* $\mathbf{v} \neq \mathbf{0}$ is the **Eigenvector** (the invariant direction).
* $\lambda \in \mathbb{R}$ is the **Eigenvalue** (the scalar stretch factor).

> [!NOTE]
> **Mathematical Rigor Note:** For an arbitrary square matrix, eigenvalues and eigenvectors may be real or complex. For the **real symmetric matrices** commonly encountered in PCA and Machine Learning (such as covariance matrices $\Sigma = \Sigma^T$), the Spectral Theorem guarantees that all eigenvalues and eigenvectors are strictly **real**, and eigenvectors corresponding to distinct eigenvalues are mutually **orthogonal**.

---

## 6.2 Complete 13-Step Hand Calculation on a 2x2 Matrix

We will solve for all eigenvalues and eigenvectors of:

$$
A =
\begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}
$$

---

### Step 1: Write the Fundamental Definition

$$
A \mathbf{v} = \lambda \mathbf{v}
$$

### Step 2: Move Everything to One Side

$$
A \mathbf{v} - \lambda \mathbf{v} = \mathbf{0} \quad \implies \quad (A - \lambda I) \mathbf{v} = \mathbf{0}
$$

### Step 3: Understand Why $\det(A - \lambda I) = 0$ is Required
If $(A - \lambda I)$ had an inverse, we could multiply both sides by $(A - \lambda I)^{-1}$:

$$
\mathbf{v} = (A - \lambda I)^{-1} \mathbf{0} = \mathbf{0}
$$

This would only give the trivial solution $\mathbf{v} = \mathbf{0}$. To find a **non-zero** eigenvector $\mathbf{v} \neq \mathbf{0}$, the matrix $(A - \lambda I)$ **MUST be non-invertible (singular)**. Therefore, its determinant must be zero:

$$
\det(A - \lambda I) = 0
$$

### Step 4: Construct the Matrix $(A - \lambda I)$

$$
A - \lambda I =
\begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix} - \lambda
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix} =
\begin{bmatrix}
2 - \lambda & 1 \\
1 & 2 - \lambda
\end{bmatrix}
$$

### Step 5: Compute the Determinant

$$
\det(A - \lambda I) = (2 - \lambda)(2 - \lambda) - (1 \times 1) = (2 - \lambda)^2 - 1
$$

### Step 6: Form the Characteristic Polynomial

$$
(4 - 4\lambda + \lambda^2) - 1 = 0 \quad \implies \quad \lambda^2 - 4\lambda + 3 = 0
$$

### Step 7: Solve for the Eigenvalues ($\lambda$)
Factor the quadratic equation:

$$
(\lambda - 3)(\lambda - 1) = 0 \quad \implies \quad \lambda_1 = 3, \quad \lambda_2 = 1
$$

*We have found our two eigenvalues!*

---

### Step 8: Find Eigenvector 1 for $\lambda_1 = 3$
Substitute $\lambda_1 = 3$ into $(A - \lambda I)\mathbf{v} = \mathbf{0}$:

$$
\begin{bmatrix}
2 - 3 & 1 \\
1 & 2 - 3
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix} =
\begin{bmatrix}
0 \\
0
\end{bmatrix}
\quad \implies \quad
\begin{bmatrix}
-1 & 1 \\
1 & -1
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix} =
\begin{bmatrix}
0 \\
0
\end{bmatrix}
$$

### Step 9: Solve the Simultaneous Equations for $\mathbf{v}_1$

$$
-1v_1 + 1v_2 = 0 \implies v_2 = v_1
$$

Any non-zero vector where $v_1 = v_2$ is an eigenvector. Choose $v_1 = 1 \implies v_2 = 1$:

$$
\mathbf{v}_1 =
\begin{bmatrix}
1 \\
1
\end{bmatrix}
\quad \quad \text{Normalized (Unit Length): } \mathbf{u}_1 =
\begin{bmatrix}
1/\sqrt{2} \\
1/\sqrt{2}
\end{bmatrix}
$$

---

### Step 10: Find Eigenvector 2 for $\lambda_2 = 1$
Substitute $\lambda_2 = 1$ into $(A - \lambda I)\mathbf{v} = \mathbf{0}$:

$$
\begin{bmatrix}
2 - 1 & 1 \\
1 & 2 - 1
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix} =
\begin{bmatrix}
0 \\
0
\end{bmatrix}
\quad \implies \quad
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix} =
\begin{bmatrix}
0 \\
0
\end{bmatrix}
$$

$$
1v_1 + 1v_2 = 0 \implies v_2 = -v_1
$$

Choose $v_1 = 1 \implies v_2 = -1$:

$$
\mathbf{v}_2 =
\begin{bmatrix}
1 \\
-1
\end{bmatrix}
\quad \quad \text{Normalized (Unit Length): } \mathbf{u}_2 =
\begin{bmatrix}
1/\sqrt{2} \\
-1/\sqrt{2}
\end{bmatrix}
$$

---

### Step 11: Verify the Result ($A \mathbf{v} = \lambda \mathbf{v}$)
1. **Check $\mathbf{v}_1$ with $\lambda_1 = 3$:**

$$
A \mathbf{v}_1 =
\begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}
\begin{bmatrix}
1 \\
1
\end{bmatrix} =
\begin{bmatrix}
2(1) + 1(1) \\
1(1) + 2(1)
\end{bmatrix} =
\begin{bmatrix}
3 \\
3
\end{bmatrix}
= 3
\begin{bmatrix}
1 \\
1
\end{bmatrix}
= \lambda_1 \mathbf{v}_1 \quad \checkmark
$$

2. **Check $\mathbf{v}_2$ with $\lambda_2 = 1$:**

$$
A \mathbf{v}_2 =
\begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}
\begin{bmatrix}
1 \\
-1
\end{bmatrix} =
\begin{bmatrix}
2(1) + 1(-1) \\
1(1) + 2(-1)
\end{bmatrix} =
\begin{bmatrix}
1 \\
-1
\end{bmatrix}
= 1
\begin{bmatrix}
1 \\
-1
\end{bmatrix}
= \lambda_2 \mathbf{v}_2 \quad \checkmark
$$

---

### Step 12: Verify Orthogonality of Eigenvectors
Because matrix $A$ is symmetric ($A = A^T$), its eigenvectors for distinct eigenvalues **must be orthogonal**:

$$
\mathbf{v}_1 \cdot \mathbf{v}_2 = (1 \times 1) + (1 \times -1) = 1 - 1 = 0 \quad \checkmark
$$

## 6.3 Geometric Meaning of Eigenvectors & Eigenvalues

### Step 13: Geometric Interpretation
* Any vector along the line $y = x$ ($\mathbf{v}_1$) is stretched by a factor of $3$.
* Any vector along the perpendicular line $y = -x$ ($\mathbf{v}_2$) is stretched by a factor of $1$ (unchanged in length).
* All other vectors in 2D space get rotated toward the dominant eigenvector direction $\mathbf{v}_1$.

---

# PART 7 — EIGENVALUES IN PRINCIPAL COMPONENT ANALYSIS (PCA)

---

## 7.1 The Chain of Logic: Data $\to$ Covariance $\to$ Eigenvalues $\to$ Components

```
     RAW DATA (X)          CENTER DATA (X - mu)        COVARIANCE MATRIX (Sigma)
   [x1, y1]                     [x1', y1']                [ Var(x)   Cov(x,y) ]
   [x2, y2]             ──►     [x2', y2']          ──►   [ Cov(x,y) Var(y)   ]
   [x3, y3]                     [x3', y3']                           │
                                                                     ▼
   DIMENSIONALITY REDUCTION      PROJECT DATA              EIGENDECOMPOSITION
   Keep top k components   ◄──   Z = X_centered @ V_k  ◄── Sigma @ v = lambda * v
   (Max variance retained)                                 (Sort lambda1 >= lambda2)
```

1. **Why do we care about eigenvectors in PCA?**
   * PCA seeks the straight line in space along which the data varies the most (maximum variance).
   * The direction of maximum variance is mathematically the **principal eigenvector** of the data's covariance matrix $\Sigma$.
2. **Why do we care about eigenvalues in PCA?**
   * The eigenvalue $\lambda_i$ equals the exact numerical **variance** of the data when projected onto eigenvector $\mathbf{v}_i$.
   * A larger eigenvalue means more information/variance is preserved along that axis.

---

## 7.2 Why Eigenvectors Maximize Variance

Let $\mathbf{u}$ be a unit projection vector ($\|\mathbf{u}\|_2 = 1$). The variance of the projected data points is:

$$
\text{Variance}(\text{Projection}) = \mathbf{u}^T \Sigma \mathbf{u}
$$

To find the direction $\mathbf{u}$ that maximizes this variance, we set up the **Lagrangian optimization**:

$$
\mathcal{L}(\mathbf{u}, \lambda) = \mathbf{u}^T \Sigma \mathbf{u} - \lambda (\mathbf{u}^T \mathbf{u} - 1)
$$

Taking the partial derivative with respect to $\mathbf{u}$ and setting it to $\mathbf{0}$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{u}} = 2\Sigma \mathbf{u} - 2\lambda \mathbf{u} = \mathbf{0} \quad \implies \quad \Sigma \mathbf{u} = \lambda \mathbf{u}
$$

* **The Takeaway:** The condition for maximum variance is *identically* the eigenvalue equation $\Sigma \mathbf{u} = \lambda \mathbf{u}$.

---

# PART 8 — COVARIANCE MATRIX: COMPLETE HAND CALCULATION

---

## 8.1 Variance vs. Covariance Formulas

* **Variance (Spread of single variable $X$):**

$$
\text{Var}(X) = s_X^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2
$$

* **Covariance (Joint linear association between $X$ and $Y$):**

$$
\text{Cov}(X, Y) = s_{XY} = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})
$$

  * If $\text{Cov}(X,Y) \gt 0$: When $X$ increases, $Y$ tends to increase.
  * If $\text{Cov}(X,Y) \lt 0$: When $X$ increases, $Y$ tends to decrease.
  * If $\text{Cov}(X,Y) = 0$: **No linear association exists between $X$ and $Y$. (A non-linear relationship can still exist!)**

---

## 8.2 Complete Step-by-Step Hand Calculation

We have a tiny dataset of $n=3$ observations on two features ($X=$ Study Hours, $Y=$ Exam Score):
* $X = [1, 2, 3]$
* $Y = [2, 3, 7]$

---

### Step 1: Calculate Sample Means ($\bar{x}, \bar{y}$)

$$
\bar{x} = \frac{1 + 2 + 3}{3} = \frac{6}{3} = 2.0
$$

$$
\bar{y} = \frac{2 + 3 + 7}{3} = \frac{12}{3} = 4.0
$$

---

### Step 2: Build the Centered Deviation Table

| Sample $i$ | $x_i$ | $y_i$ | $(x_i - \bar{x})$ | $(y_i - \bar{y})$ | $(x_i - \bar{x})^2$ | $(y_i - \bar{y})^2$ | $(x_i - \bar{x})(y_i - \bar{y})$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1 | 2 | $1 - 2 = \mathbf{-1}$ | $2 - 4 = \mathbf{-2}$ | $(-1)^2 = \mathbf{1}$ | $(-2)^2 = \mathbf{4}$ | $(-1) \times (-2) = \mathbf{2}$ |
| **2** | 2 | 3 | $2 - 2 = \mathbf{0}$ | $3 - 4 = \mathbf{-1}$ | $(0)^2 = \mathbf{0}$ | $(-1)^2 = \mathbf{1}$ | $(0) \times (-1) = \mathbf{0}$ |
| **3** | 3 | 7 | $3 - 2 = \mathbf{1}$ | $7 - 4 = \mathbf{3}$ | $(1)^2 = \mathbf{1}$ | $(3)^2 = \mathbf{9}$ | $(1) \times (3) = \mathbf{3}$ |
| **SUM** | | | | | $\sum = \mathbf{2}$ | $\sum = \mathbf{14}$ | $\sum = \mathbf{5}$ |

---

### Step 3: Calculate Sample Variances & Covariance (Divide by $n-1 = 2$)

$$
\text{Var}(X) = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n - 1} = \frac{2}{2} = 1.0
$$

$$
\text{Var}(Y) = \frac{\sum_{i=1}^{n} (y_i - \bar{y})^2}{n - 1} = \frac{14}{2} = 7.0
$$

$$
\text{Cov}(X, Y) = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{n - 1} = \frac{5}{2} = 2.5
$$

---

### Step 4: Construct the 2x2 Covariance Matrix

$$
\Sigma =
\begin{bmatrix}
\text{Var}(X) & \text{Cov}(X,Y) \\
\text{Cov}(Y,X) & \text{Var}(Y)
\end{bmatrix} =
\begin{bmatrix}
1.0 & 2.5 \\
2.5 & 7.0
\end{bmatrix}
$$

---

## 8.3 The Symmetry & Quadratic Structure of $\Sigma$

* **Diagonal elements:** Always individual feature variances ($\ge 0$).
* **Off-diagonal elements:** Pairwise covariances. Because $\text{Cov}(X, Y) = \text{Cov}(Y, X)$, the covariance matrix is **always symmetric**:

$$
\Sigma = \Sigma^T
$$

* **Matrix Formulation:** If $X_c$ is the mean-centered data matrix ($(N \times p)$), then:

$$
\Sigma = \frac{1}{n-1} X_c^T X_c
$$

---

## 8.4 Sample ($n-1$) vs. Population ($N$) Bessel's Correction

* **Population Covariance (Divide by $N$):** Used when the dataset contains every single member of the population:

$$
\Sigma_{\text{pop}} = \frac{1}{N} X_c^T X_c \implies \Sigma_{\text{pop}} =
\begin{bmatrix}
2/3 & 5/3 \\
5/3 & 14/3
\end{bmatrix}
\approx
\begin{bmatrix}
0.67 & 1.67 \\
1.67 & 4.67
\end{bmatrix}
$$

* **Sample Covariance (Divide by $n-1$, Bessel's Correction):** Used when the dataset is a sample drawn from a larger population. Dividing by $n-1$ corrects for the fact that sample deviations around the sample mean $\bar{x}$ are systematically smaller than deviations around the true unknown population mean $\mu$, providing an **unbiased estimator**.

---

# PART 9 — COMPLETE PCA WALKTHROUGH FROM SCRATCH

---

## 9.1 The 6-Step End-to-End PCA Algorithm

```
  1. Mean Center Data  ──►  2. Compute Covariance  ──►  3. Solve Eigenvalues
                                                                  │
  6. Reduced Dataset   ◄──  5. Project Data Points ◄──  4. Select Top Eigenvectors
```

---

## 9.2 Step-by-Step Numerical Hand Walkthrough (2D to 1D)

We have a 2D dataset of 3 points lying along the line $y = x$:

$$
X =
\begin{bmatrix}
1 & 1 \\
2 & 2 \\
3 & 3
\end{bmatrix}
$$

---

### Step 1: Center the Data ($\bar{x}=2, \bar{y}=2$)

$$
X_c = X - \mu =
\begin{bmatrix}
1 - 2 & 1 - 2 \\
2 - 2 & 2 - 2 \\
3 - 2 & 3 - 2
\end{bmatrix} =
\begin{bmatrix}
-1 & -1 \\
0 & 0 \\
1 & 1
\end{bmatrix}
$$

---

### Step 2: Compute Sample Covariance Matrix $\Sigma = \frac{1}{n-1} X_c^T X_c$

$$
X_c^T X_c =
\begin{bmatrix}
-1 & 0 & 1 \\
-1 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
-1 & -1 \\
0 & 0 \\
1 & 1
\end{bmatrix} =
\begin{bmatrix}
(-1)^2 + 0 + 1^2 & (-1)(-1) + 0 + (1)(1) \\
(-1)(-1) + 0 + (1)(1) & (-1)^2 + 0 + 1^2
\end{bmatrix} =
\begin{bmatrix}
2 & 2 \\
2 & 2
\end{bmatrix}
$$

$$
\Sigma = \frac{1}{3 - 1}
\begin{bmatrix}
2 & 2 \\
2 & 2
\end{bmatrix} =
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}
$$

---

### Step 3: Compute Eigenvalues of $\Sigma$

$$
\det(\Sigma - \lambda I) = \det
\begin{bmatrix}
1 - \lambda & 1 \\
1 & 1 - \lambda
\end{bmatrix}
= (1 - \lambda)^2 - 1 = \lambda^2 - 2\lambda = \lambda(\lambda - 2) = 0
$$

$$
\lambda_1 = 2, \quad \lambda_2 = 0
$$

---

### Step 4: Compute Top Eigenvector ($\mathbf{u}_1$ for $\lambda_1 = 2$)

$$
\begin{bmatrix}
1 - 2 & 1 \\
1 & 1 - 2
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix} =
\begin{bmatrix}
0 \\
0
\end{bmatrix}
\implies -v_1 + v_2 = 0 \implies v_1 = v_2
$$

Unit Eigenvector:

$$
\mathbf{u}_1 =
\begin{bmatrix}
1/\sqrt{2} \\
1/\sqrt{2}
\end{bmatrix}
$$

---

### Step 5: Project Centered Data onto Principal Component $\mathbf{u}_1$

$$
Z = X_c \mathbf{u}_1 =
\begin{bmatrix}
-1 & -1 \\
0 & 0 \\
1 & 1
\end{bmatrix}
\begin{bmatrix}
1/\sqrt{2} \\
1/\sqrt{2}
\end{bmatrix} =
\begin{bmatrix}
-1(1/\sqrt{2}) - 1(1/\sqrt{2}) \\
0(1/\sqrt{2}) + 0(1/\sqrt{2}) \\
1(1/\sqrt{2}) + 1(1/\sqrt{2})
\end{bmatrix} =
\begin{bmatrix}
-2/\sqrt{2} \\
0 \\
2/\sqrt{2}
\end{bmatrix} =
\begin{bmatrix}
-\sqrt{2} \\
0 \\
\sqrt{2}
\end{bmatrix}
\approx
\begin{bmatrix}
-1.414 \\
0.000 \\
1.414
\end{bmatrix}
$$

---

## 9.3 Calculating Explained Variance Ratio

$$
\text{Explained Variance Ratio} = \frac{\lambda_k}{\sum \lambda_i}
$$

* For Component 1: $\frac{\lambda_1}{\lambda_1 + \lambda_2} = \frac{2}{2 + 0} = \frac{2}{2} = 1.0 \implies$ **100%**.
* **The Takeaway:** By compressing from 2D coordinates $[x, y]$ down to a 1D scalar $z$, we **retained 100% of the variance in this particular dataset** while eliminating 1 redundant feature axis.

---

# PART 10 — SINGULAR VALUE DECOMPOSITION (SVD)

---

## 10.1 The SVD Equation: $A = U \Sigma V^T$

While eigendecomposition only works on **square** matrices ($N \times N$), **Singular Value Decomposition (SVD)** factorizes **ANY matrix** of shape $(M \times N)$ into three constituent geometric matrices:

$$
A = U \Sigma V^T
$$

```
      A (M x N)              U (M x M)               Sigma (M x N)             V^T (N x N)
   ┌             ┐       ┌                ┐       ┌                 ┐       ┌                ┐
   │             │   =   │ Left Singular  │   ×   │ Singular Values │   ×   │ Right Singular │
   │   DATA      │       │ Vectors        │       │ (Stretch scale) │       │ Vectors        │
   │             │       │ (Orthonormal)  │       │ (Diagonal)      │       │ (Orthonormal)  │
   └             ┘       └                ┘       └                 ┘       └                ┘
```

1. **$U$ (Left Singular Vectors, $M \times M$):** Orthonormal eigenvectors of $A A^T$ ($U^T U = I$). Represents column space basis directions.
2. **$\Sigma$ (Singular Values, $M \times N$):** Diagonal matrix of non-negative singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$ in descending order. Represents scaling/stretch factors along each principal axis.
3. **$V^T$ (Right Singular Vectors Transpose, $N \times N$):** Transpose of the orthonormal eigenvectors of $A^T A$ ($V^T V = I$). Represents row space / principal feature directions.

---

## 10.2 Geometric Interpretation of $U, \Sigma, V^T$

Any linear transformation can be decomposed into three fundamental geometric steps:

$$
\mathbf{x} \quad \xrightarrow{\quad V^T \text{ (Rotate/Reflect)} \quad} \quad \xrightarrow{\quad \Sigma \text{ (Scale axes)} \quad} \quad \xrightarrow{\quad U \text{ (Second Rotation)} \quad} \quad A\mathbf{x}
$$

---

## 10.3 Relationship Between SVD and PCA

For a mean-centered data matrix $X_c$ ($N \times p$):
1. **Covariance Matrix:** $\Sigma_{\text{cov}} = \frac{1}{n-1} X_c^T X_c$.
2. Substitute SVD factorization $X_c = U \Sigma V^T$:

$$
X_c^T X_c = (U \Sigma V^T)^T (U \Sigma V^T) = (V \Sigma^T U^T)(U \Sigma V^T) = V \Sigma^T (U^T U) \Sigma V^T = V \Sigma^2 V^T
$$

*(Since $U$ is orthonormal, $U^T U = I$).*

3. **Core Relationships to Memorize:**
   * The **Right Singular Vectors ($V$)** of $X_c$ are **identical to the Principal Component Directions (Eigenvectors of $\Sigma_{\text{cov}}$)**.
   * The **Singular Values ($\sigma_i$)** relate directly to the Eigenvalues ($\lambda_i$):

$$
\lambda_i = \frac{\sigma_i^2}{n - 1} \quad \iff \quad \sigma_i = \sqrt{(n - 1)\lambda_i}
$$

   * For the unnormalized matrix $A^T A$, singular values satisfy $\sigma_i = \sqrt{\lambda_i(A^T A)}$.
   * *Why modern libraries use SVD instead of Eigendecomposition:* Scikit-Learn's `PCA` uses SVD internally because computing $X_c = U \Sigma V^T$ directly avoids explicitly forming the $X_c^T X_c$ matrix, offering higher numerical precision and avoiding the squaring of condition numbers.

---

# PART 11 — ORTHOGONALITY & ORTHONORMAL BASES

---

## 11.1 Orthogonal Vectors ($u \cdot v = 0$)

Two vectors are orthogonal if they meet at a $90^\circ$ right angle:

$$
\mathbf{u} \perp \mathbf{v} \iff \mathbf{u} \cdot \mathbf{v} = 0
$$

* *Hand Example:* Let $\mathbf{u} = [3, 1]^T$ and $\mathbf{v} = [-2, 6]^T$:

$$
\mathbf{u} \cdot \mathbf{v} = 3(-2) + 1(6) = -6 + 6 = 0
$$

---

## 11.2 Orthonormal Bases & Unit Length

A set of vectors $\lbrace\mathbf{q}_1, \mathbf{q}_2, \dots, \mathbf{q}_k\rbrace$ is **Orthonormal** if:
1. Every vector is orthogonal to every other vector: $\mathbf{q}_i \cdot \mathbf{q}_j = 0$ for $i \neq j$.
2. Every vector has unit length ($L_2$ norm = 1): $\|\mathbf{q}_i\|_2 = 1 \implies \mathbf{q}_i \cdot \mathbf{q}_i = 1$.

* *Standard Basis in 2D:*

$$
\mathbf{e}_1 =
\begin{bmatrix}
1 \\
0
\end{bmatrix},
\quad \mathbf{e}_2 =
\begin{bmatrix}
0 \\
1
\end{bmatrix}
$$

* *Rotated Orthonormal Basis:*

$$
\mathbf{q}_1 =
\begin{bmatrix}
1/\sqrt{2} \\
1/\sqrt{2}
\end{bmatrix},
\quad \mathbf{q}_2 =
\begin{bmatrix}
-1/\sqrt{2} \\
1/\sqrt{2}
\end{bmatrix}
$$

---

## 11.3 Orthogonal Matrices ($Q^T Q = I$)

A square matrix $Q$ whose columns are mutually orthonormal is called an **Orthogonal Matrix**:

$$
Q^T Q = Q Q^T = I \quad \iff \quad Q^{-1} = Q^T
$$

* **The Superpower of Orthogonal Matrices:** To invert an orthogonal matrix, **you never need to compute matrix inversion—you simply transpose it!**
* **Distance & Angle Preserving:** Multiplying any vector by an orthogonal matrix preserves length and angles: $\|Q\mathbf{x}\|_2 = \|\mathbf{x}\|_2$. It performs pure rigid rotation/reflection with zero distortion.

---

# PART 12 — VECTOR PROJECTIONS

---

## 12.1 Derivation of the Scalar & Vector Projection Formula

```
                                      a
                                     ╱│
                                    ╱ │ Error (a - p) is orthogonal to b!
                                   ╱  │
                                  ●───┴──────────► b
                                  0   p = proj_b(a)
```

We wish to drop a perpendicular shadow from vector $\mathbf{a}$ onto vector $\mathbf{b}$.
1. The projection $\mathbf{p} = \text{proj}_{\mathbf{b}}(\mathbf{a})$ lies along the direction of $\mathbf{b}$, so $\mathbf{p} = c \mathbf{b}$ for some scalar $c$.
2. The error vector $(\mathbf{a} - c\mathbf{b})$ must be **orthogonal** to $\mathbf{b}$:

$$
\mathbf{b} \cdot (\mathbf{a} - c\mathbf{b}) = 0
$$

3. Expand the dot product:

$$
\mathbf{b} \cdot \mathbf{a} - c(\mathbf{b} \cdot \mathbf{b}) = 0 \implies c = \frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{b} \cdot \mathbf{b}} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{b}\|_2^2}
$$

4. Multiply scalar $c$ by vector $\mathbf{b}$:

$$
\text{proj}_{\mathbf{b}}(\mathbf{a}) = \left(\frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{b} \cdot \mathbf{b}}\right) \mathbf{b}
$$

---

## 12.2 Step-by-Step Numerical Hand Calculation

Project vector $\mathbf{a} = [3, 4]^T$ onto vector $\mathbf{b} = [4, 0]^T$:
1. Compute $\mathbf{a} \cdot \mathbf{b} = (3 \times 4) + (4 \times 0) = 12 + 0 = 12$.
2. Compute $\mathbf{b} \cdot \mathbf{b} = (4 \times 4) + (0 \times 0) = 16 + 0 = 16$.
3. Compute projection vector:

$$
\mathbf{p} = \frac{12}{16}
\begin{bmatrix}
4 \\
0
\end{bmatrix}
= \frac{3}{4}
\begin{bmatrix}
4 \\
0
\end{bmatrix} =
\begin{bmatrix}
3 \\
0
\end{bmatrix}
$$

4. Verification of Orthogonal Error: $\mathbf{e} = \mathbf{a} - \mathbf{p} = [3, 4]^T - [3, 0]^T = [0, 4]^T$.
   * $\mathbf{e} \cdot \mathbf{b} = (0 \times 4) + (4 \times 0) = 0 \quad \checkmark$

---

## 12.3 Projection Matrices & Subspaces

The **Projection Matrix** $P$ that projects any arbitrary vector onto the column space of a matrix $X$:

$$
P = X (X^T X)^{-1} X^T
$$

* **Property:** $P^2 = P$ (Projecting a second time changes nothing).
* **ML Connection:** Linear regression predictions $\hat{\mathbf{y}} = X \mathbf{w} = X(X^T X)^{-1} X^T \mathbf{y} = P \mathbf{y}$ is the orthogonal projection of target vector $\mathbf{y}$ onto the column space of feature matrix $X$.

---

# PART 13 — LINEAR REGRESSION: MATRIX MATHEMATICS

---

## 13.1 The Matrix System: $y = Xw + \epsilon$

$$
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_N
\end{bmatrix} =
\begin{bmatrix}
1 & x_{11} & \dots & x_{1p} \\
1 & x_{21} & \dots & x_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
1 & x_{N1} & \dots & x_{Np}
\end{bmatrix}
\begin{bmatrix}
w_0 \\
w_1 \\
\vdots \\
w_p
\end{bmatrix}
+
\begin{bmatrix}
\epsilon_1 \\
\epsilon_2 \\
\vdots \\
\epsilon_N
\end{bmatrix}
$$

* Target vector: $\mathbf{y} \in \mathbb{R}^N$
* Design matrix: $X \in \mathbb{R}^{N \times (p+1)}$
* Weights vector: $\mathbf{w} \in \mathbb{R}^{p+1}$

---

## 13.2 Minimizing the Sum of Squared Residuals (Least Squares)

Residual error vector: $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} = \mathbf{y} - X\mathbf{w}$.

The Ordinary Least Squares (OLS) Loss function is the squared $L_2$ norm of the error vector:

$$
\mathcal{L}(\mathbf{w}) = \|\mathbf{y} - X\mathbf{w}\|_2^2 = (\mathbf{y} - X\mathbf{w})^T (\mathbf{y} - X\mathbf{w})
$$

Expand the matrix transpose product:

$$
\mathcal{L}(\mathbf{w}) = (\mathbf{y}^T - \mathbf{w}^T X^T)(\mathbf{y} - X\mathbf{w}) = \mathbf{y}^T \mathbf{y} - \mathbf{y}^T X \mathbf{w} - \mathbf{w}^T X^T \mathbf{y} + \mathbf{w}^T X^T X \mathbf{w}
$$

Since $\mathbf{y}^T X \mathbf{w}$ is a scalar, $(\mathbf{y}^T X \mathbf{w})^T = \mathbf{w}^T X^T \mathbf{y}$. Combining terms:

$$
\mathcal{L}(\mathbf{w}) = \mathbf{y}^T \mathbf{y} - 2\mathbf{w}^T X^T \mathbf{y} + \mathbf{w}^T X^T X \mathbf{w}
$$

---

## 13.3 Derivation of the Normal Equation ($X^T X w = X^T y$)

To find the minimum loss, take the matrix derivative with respect to $\mathbf{w}$ and set it to $\mathbf{0}$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = -2 X^T \mathbf{y} + 2 X^T X \mathbf{w} = \mathbf{0}
$$

Divide by 2 and rearrange:

$$
X^T X \mathbf{w} = X^T \mathbf{y}
$$

Assuming $X^T X$ is invertible, multiply both sides by $(X^T X)^{-1}$:

$$
\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}
$$

> [!IMPORTANT]
> **Key Conditions & Practical Reality:**
> 1. **Invertibility Assumption:** The closed-form expression $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$ strictly assumes that $X^T X$ is full rank and invertible ($\det(X^T X) \neq 0$).
> 2. **Numerical Solvers vs. Matrix Inversion:** The formula is essential for mathematical derivations and understanding OLS. In production numerical software (e.g. Scikit-Learn, LAPACK), explicitly computing $(X^T X)^{-1}$ is avoided due to $\mathcal{O}(p^3)$ cost and numerical instability. Instead, systems use **QR Decomposition** ($X = QR$), **SVD**, or **iterative Gradient Descent**.

---

## 13.4 Why $(X^T X)^{-1}$ Fails in Practice (Condition Number & QR Solvers)

1. **Computational Cost:** Matrix inversion of $(p \times p)$ scales as $\mathcal{O}(p^3)$. If $p = 50,000$ features, computing $(X^T X)^{-1}$ requires $\approx 1.25 \times 10^{14}$ operations.
2. **Ill-Conditioned / Multicollinearity:** If features are highly correlated, $(X^T X)$ has a condition number near $\infty$. Tiny floating-point rounding errors cause learned weights to blow up to massive unstable numbers ($w_1 = +10^6, w_2 = -10^6$).
3. **Industry Standard Alternatives:**
   * **QR Decomposition:** $X = QR \implies R \mathbf{w} = Q^T \mathbf{y}$ (Solves via fast back-substitution without explicit inversion).
   * **Gradient Descent:** Iteratively steps down the loss gradient in $\mathcal{O}(Np)$ per iteration.

---

# PART 14 — GRADIENTS & DERIVATIVES FOR OPTIMIZATION

---

## 14.1 1D Derivatives & Slope of Tangent Line

The derivative $\frac{df}{dx}$ measures the instantaneous rate of change of $f(x)$ with respect to $x$:

$$
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

* *Hand Example:* For $f(x) = x^2$:

$$
f'(x) = \lim_{h \to 0} \frac{(x+h)^2 - x^2}{h} = \lim_{h \to 0} \frac{x^2 + 2xh + h^2 - x^2}{h} = \lim_{h \to 0} (2x + h) = 2x
$$

  * At $x = 3$: Slope $= 2(3) = +6$ (Function is increasing steeply).
  * At $x = 0$: Slope $= 2(0) = 0$ (Minimum point / flat slope).
  * At $x = -2$: Slope $= 2(-2) = -4$ (Function is decreasing).

---

## 14.2 Partial Derivatives (Multivariable Functions)

When a function depends on multiple variables $f(x_1, x_2, \dots, x_n)$, the **partial derivative** $\frac{\partial f}{\partial x_i}$ measures how $f$ changes when varying $x_i$ while treating **all other variables as constants**.

### Hand Calculation Example
Let $f(x, y) = x^2 + 3xy + y^2$:
1. **Compute $\frac{\partial f}{\partial x}$ (treat $y$ as a constant number):**

$$
\frac{\partial f}{\partial x} = \frac{d}{dx}[x^2] + \frac{d}{dx}[3y \cdot x] + \frac{d}{dx}[y^2] = 2x + 3y + 0 = 2x + 3y
$$

2. **Compute $\frac{\partial f}{\partial y}$ (treat $x$ as a constant number):**

$$
\frac{\partial f}{\partial y} = \frac{d}{dy}[x^2] + \frac{d}{dy}[3x \cdot y] + \frac{d}{dy}[y^2] = 0 + 3x + 2y = 3x + 2y
$$

---

## 14.3 The Gradient Vector (Direction of Steepest Ascent)

The **Gradient** $\nabla f$ bundles all partial derivatives into a single vector:

$$
\nabla f(x, y) =
\begin{bmatrix}
\frac{\partial f}{\partial x} \\
\frac{\partial f}{\partial y}
\end{bmatrix}
$$

* **Fundamental Theorem:** The gradient vector $\nabla f$ points in the **direction of greatest rate of increase (steepest uphill slope)**.
* **Negative Gradient ($-\nabla f$):** Points in the **direction of steepest descent (fastest downhill path to the minimum)**.

---

## 14.4 Gradient Descent: 3-Step Numerical Hand Trace

We wish to find the minimum of $f(x) = x^2$ using Gradient Descent.
* Update Rule: $x_{t+1} = x_t - \alpha \nabla f(x_t)$
* Gradient: $\nabla f(x) = 2x$
* Settings: Start at initial guess $x_0 = 4.0$, Learning Rate $\alpha = 0.1$.

```
     Iteration 0: x = 4.0  ──►  f(x) = 16.0   (Gradient = 8.0)
     Iteration 1: x = 3.2  ──►  f(x) = 10.24  (Gradient = 6.4)
     Iteration 2: x = 2.56 ──►  f(x) = 6.55   (Gradient = 5.12)
     Iteration 3: x = 2.05 ──►  f(x) = 4.19   (Converging smoothly toward x=0!)
```

### Iteration 1:
1. Compute gradient at $x_0 = 4.0$:

$$
\nabla f(4.0) = 2(4.0) = 8.0
$$

2. Update parameter:

$$
x_1 = 4.0 - 0.1(8.0) = 4.0 - 0.8 = \mathbf{3.2}
$$

### Iteration 2:
1. Compute gradient at $x_1 = 3.2$:

$$
\nabla f(3.2) = 2(3.2) = 6.4
$$

2. Update parameter:

$$
x_2 = 3.2 - 0.1(6.4) = 3.2 - 0.64 = \mathbf{2.56}
$$

### Iteration 3:
1. Compute gradient at $x_2 = 2.56$:

$$
\nabla f(2.56) = 2(2.56) = 5.12
$$

2. Update parameter:

$$
x_3 = 2.56 - 0.1(5.12) = 2.56 - 0.512 = \mathbf{2.048}
$$

---

# PART 15 — THE CHAIN RULE & BACKPROPAGATION

---

## 15.1 Single-Variable & Multivariable Chain Rule

For a composite nested function $y = f(u)$ where $u = g(x)$:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

---

## 15.2 Hand Trace on a Nested Function

Let $y = (3x + 2)^2$. Find $\frac{dy}{dx}$ at $x = 1$.
1. Break down into sub-nodes:
   * Inner function: $u = 3x + 2$
   * Outer function: $y = u^2$
2. Compute individual derivatives:
   * $\frac{du}{dx} = 3$
   * $\frac{dy}{du} = 2u$
3. Apply chain rule:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = (2u) \times 3 = 2(3x + 2) \times 3 = 6(3x + 2) = 18x + 12
$$

4. Evaluate at $x = 1$:

$$
\frac{dy}{dx}\Big|_{x=1} = 18(1) + 12 = 30
$$

---

## 15.3 Neural Network Computational Graphs

In deep learning, every neural network layer is a composite function:

$$
\text{Input } x \quad \xrightarrow{\quad \text{Linear} \quad} \quad z = wx + b \quad \xrightarrow{\quad \text{Activation} \quad} \quad a = \sigma(z) \quad \xrightarrow{\quad \text{Loss} \quad} \quad \mathcal{L}(a, y)
$$

```
     FORWARD PASS:   x ──► [ z = w*x + b ] ──► [ a = sigma(z) ] ──► [ Loss L ]
                                                                        │
     BACKWARD PASS:  dL/dw = (dL/da) * (da/dz) * (dz/dw)  ◄─────────────┘
```

By the Chain Rule:

$$
\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

* $\frac{\partial z}{\partial w} = x$
* $\frac{\partial a}{\partial z} = \sigma'(z)$
* $\frac{\partial \mathcal{L}}{\partial a} = \text{Loss gradient}$
* **The Insight:** **Backpropagation is nothing more than the repeated application of the Chain Rule from output to input!**

---

# PART 16 — DISTANCE & SIMILARITY METRICS

---

## 16.1 Euclidean Distance ($L_2$) vs. Manhattan Distance ($L_1$)

Let Point $A = (1, 2)$ and Point $B = (4, 6)$:

1. **Euclidean Distance ($L_2$):**

$$
d_{L2}(A, B) = \sqrt{(4 - 1)^2 + (6 - 2)^2} = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = \mathbf{5.0}
$$

2. **Manhattan Distance ($L_1$):**

$$
d_{L1}(A, B) = |4 - 1| + |6 - 2| = |3| + |4| = 3 + 4 = \mathbf{7.0}
$$

3. **Minkowski Distance ($L_p$ Generalization):**

$$
d_{Lp}(A, B) = \left(\sum_{i=1}^{n} |a_i - b_i|^p\right)^{1/p}
$$

   * $p=1 \implies$ Manhattan
   * $p=2 \implies$ Euclidean

---

## 16.2 Cosine Distance vs. Cosine Similarity

$$
\text{Cosine Distance} = 1 - \text{Cosine Similarity} = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}
$$

* If vectors are identical ($\theta = 0^\circ$): Similarity $= 1.0 \implies$ Distance $= 0.0$.
* If vectors are orthogonal ($\theta = 90^\circ$): Similarity $= 0.0 \implies$ Distance $= 1.0$.

---

## 16.3 Comparison Table & Real-World Selection Matrix

| Metric | Formula | Sensitive to Scale? | When to Use (ML Applications) |
| :--- | :--- | :--- | :--- |
| **Euclidean ($L_2$)** | $\sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$ | **High** | Physical spatial coordinates, image pixel grids, KNN on standardized features. |
| **Manhattan ($L_1$)** | $\sum_{i=1}^{n} \lvert p_i - q_i \rvert$ | **High** | High-dimensional data (less vulnerable to Curse of Dimensionality than $L_2$), grid layouts. |
| **Cosine Similarity** | $\frac{\mathbf{u} \cdot \mathbf{v}}{\lVert \mathbf{u} \rVert_2 \lVert \mathbf{v} \rVert_2}$ | **Zero (Invariant)** | Text embeddings, recommendation systems, semantic search (focuses on angle, not magnitude). |

---

# PART 17 — REGULARIZATION MATHEMATICS (L1 vs. L2)

---

## 17.1 The Constrained Optimization Formulation

$$
\min_{\mathbf{w}} \text{MSE}(\mathbf{w}) \quad \text{subject to} \quad \|\mathbf{w}\|_p \le C
$$

* **Ridge Regression ($L_2$):** Subject to $w_1^2 + w_2^2 \le C$ (A smooth **circular/spherical ball**).
* **Lasso Regression ($L_1$):** Subject to $|w_1| + |w_2| \le C$ (A **spiky diamond/polytope**).

---

## 17.2 Mathematical Geometry: Why L1 Creates Exact Zeros (Sparsity)

```
        L2 REGULARIZATION (Circle)                  L1 REGULARIZATION (Diamond)
                   w2                                          w2
                   │   * OLS Minimum                           │   * OLS Minimum
                ╭──┼──╮                                       ╱│╲
               │   │   │                                     ╱ │ ╲
           ────┼───●───┼──── w1                          ────●──┼──●──── w1  (Hits sharp corner
               │   │   │                                     ╲ │ ╲    on w1 axis -> w2 = 0!)
                ╰──┼──╯                                       ╲│╱
          (Contact occurs at smooth edge              (Contact occurs at sharp corner
           -> w1 and w2 are small floats)              -> one weight is set to EXACT 0.0)
```

1. **The Elliptical Contours:** The MSE loss contours expand outward from the unconstrained OLS minimum.
2. **The Geometric Contact Point:**
   * **$L_1$ Diamond:** The diamond has sharp, pointy vertices lying directly on the coordinate axes ($w_1=0$ or $w_2=0$). When the growing MSE ellipses hit the constraint, they almost always touch a sharp corner first. This sets the other weight to **exact 0.0**, performing **automatic feature selection**.
   * **$L_2$ Ball:** The circle is uniformly smooth with no corners. The elliptical contours touch the circle along smooth boundaries where neither weight is exactly zero, shrinking weights asymptotically toward zero.

---

## 17.3 Mathematical Geometry: Why L2 Shrinks Smoothly (Weight Decay)

Loss function:

$$
\mathcal{L}_{\text{Ridge}} = \text{MSE} + \frac{\lambda}{2} \sum_{j=1}^{p} w_j^2
$$

Gradient update step:

$$
w_{j}^{(t+1)} = w_j^{(t)} - \alpha \left(\frac{\partial \text{MSE}}{\partial w_j} + \lambda w_j\right) = (1 - \alpha \lambda) w_j^{(t)} - \alpha \frac{\partial \text{MSE}}{\partial w_j}
$$

* Since $(1 - \alpha \lambda) \lt 1$, the weights are multiplied by a decay factor less than 1 at every single step before taking the gradient step! This is why $L_2$ is called **Weight Decay**.

---

# PART 18 — ENTROPY & INFORMATION GAIN MATHEMATICS

---

## 18.1 Shannon Entropy Formula: $H(X) = -\sum p_i \log_2 p_i$

**Entropy** measures the degree of uncertainty, disorder, or impurity in a probability distribution:

$$
H(S) = - \sum_{i=1}^{C} p_i \log_2(p_i)
$$

* Units: **Bits** (when using $\log_2$).

---

## 18.2 Hand Calculation: 50/50 Split vs. Pure Split

### Case 1: Maximum Disorder (50% Cat, 50% Dog)
* $p(\text{cat}) = 0.5$, $p(\text{dog}) = 0.5$

$$
H = - [0.5 \log_2(0.5) + 0.5 \log_2(0.5)] = - [0.5(-1) + 0.5(-1)] = - [-0.5 - 0.5] = \mathbf{1.0 \text{ Bit}}
$$

*(Maximum possible entropy for binary classification; completely unpredictable).*

### Case 2: Complete Certainty / Pure Node (100% Cat, 0% Dog)
* $p(\text{cat}) = 1.0$, $p(\text{dog}) = 0.0$ ($0 \log_2 0 \equiv 0$ by limit):

$$
H = - [1.0 \log_2(1.0) + 0] = - [1.0(0) + 0] = \mathbf{0.0 \text{ Bits}}
$$

*(Zero uncertainty; completely pure).*

---

## 18.3 Information Gain & Gini Impurity Hand Trace

### 1. Information Gain (Decision Tree Split Criterion)

$$
\text{Information Gain} = H(\text{Parent}) - \sum_{v \in \text{Children}} \frac{N_v}{N} H(v)
$$

* **Goal:** A Decision Tree chooses the feature split that **maximizes Information Gain** (creates children with the lowest combined entropy).

### 2. Gini Impurity (Faster Alternative)

$$
\text{Gini}(S) = 1 - \sum_{i=1}^{C} p_i^2
$$

* Pure Node: $\text{Gini} = 1 - (1.0)^2 = \mathbf{0.0}$.
* 50/50 Split: $\text{Gini} = 1 - (0.5^2 + 0.5^2) = 1 - (0.25 + 0.25) = \mathbf{0.50}$.
* *Why Scikit-Learn defaults to Gini:* Computing squared sums ($p_i^2$) is much faster for CPUs than computing logarithms ($\log_2 p_i$).

---

# PART 19 — ML MATHEMATICS ROADMAP TABLE

| Machine Learning Algorithm | Primary Mathematical Foundations | Key Equations / Operations |
| :--- | :--- | :--- |
| **Linear Regression** | Matrix Inverses, Least Squares, QR Decomposition, Pseudoinverse | $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y} = X^+ \mathbf{y}$, $R\mathbf{w} = Q^T \mathbf{y}$ |
| **Ridge Regression ($L_2$)** | Positive Definite Inversion, Quadratic Forms, Regularization | $(X^T X + \alpha I)^{-1} X^T \mathbf{y}$, $(1 - \alpha \lambda) \mathbf{w}$ |
| **Lasso Regression ($L_1$)** | L1 Geometry, Non-differentiable Optimization, Subgradients | $\text{MSE} + \alpha \sum_{j=1}^{p} \lvert w_j \rvert \implies \text{Sparsity}$ |
| **Logistic Regression** | Sigmoid Activation, Log-Loss (BCE), Gradients | $\sigma(z) = \frac{1}{1 + e^{-z}}$, $\nabla_{\mathbf{w}} = \frac{1}{N} X^T (\hat{\mathbf{p}} - \mathbf{y})$ |
| **K-Nearest Neighbors (KNN)** | Vector Norms, Metric Spaces, Curse of Dimensionality | $d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$ |
| **K-Means Clustering** | Euclidean Distance, Centroids, Optimization (Lloyd's) | $\arg\min_{\mu} \sum_{i=1}^{n} \lVert \mathbf{x}_i - \mu_k \rVert_2^2$ |
| **Support Vector Machines (SVM)** | Hyperplane Geometry, Projections, Quadratic Programming | $\text{Margin} = \frac{2}{\lVert \mathbf{w} \rVert_2}$, Kernel Trick $K(\mathbf{x}, \mathbf{z})$ |
| **PCA** | Covariance Matrix (PSD), Eigendecomposition, Orthogonal Projections | $\Sigma \mathbf{v} = \lambda \mathbf{v}$, $\mathbf{x}^T \Sigma \mathbf{x} \ge 0$, $Z = X_c V_k$ |
| **Singular Value Decomposition (SVD)** | Matrix Factorization, Geometric Transformations, Pseudoinverse | $A = U \Sigma V^T$, $A^+ = V \Sigma^+ U^T$ |
| **Decision Trees** | Probability, Shannon Entropy, Information Gain, Gini Impurity | $H(S) = -\sum_{i=1}^{C} p_i \log_2 p_i$, $\text{Gini} = 1 - \sum_{i=1}^{C} p_i^2$ |
| **Neural Networks / Deep Learning** | Linear Transformations, Hessians (Positive Definite), Chain Rule | $\mathbf{z} = W\mathbf{x} + \mathbf{b}$, $H \succ 0 \implies \text{Local Minimum}$ |

---

# PART 20 — "WHAT I SHOULD BE ABLE TO DO ON PAPER" CHECKLIST

Before technical interviews and coding assessments, test yourself on a blank sheet of paper:

- [ ] **1. Vector Dot Product:** Given $\mathbf{a} = [2, 3]^T, \mathbf{b} = [4, 1]^T$, compute $\mathbf{a} \cdot \mathbf{b} = 11$.
- [ ] **2. Vector Norms:** Given $\mathbf{v} = [3, -4]^T$, compute $\|\mathbf{v}\|_2 = 5$ and $\|\mathbf{v}\|_1 = 7$.
- [ ] **3. Euclidean Distance:** Given $\mathbf{p} = [1, 2]^T, \mathbf{q} = [4, 6]^T$, compute $d(\mathbf{p}, \mathbf{q}) = 5$.
- [ ] **4. Cosine Similarity:** Given $\mathbf{u} = [1, 2]^T, \mathbf{v} = [2, 4]^T$, prove similarity $= 1.0$.
- [ ] **5. Matrix Multiplication:** Given $A$ and $B$, compute $AB$:

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
\implies
AB =
\begin{bmatrix}
19 & 22 \\
43 & 50
\end{bmatrix}
$$

- [ ] **6. Matrix Transpose:** Transpose a non-square matrix and prove $(AB)^T = B^T A^T$.
- [ ] **7. 2x2 Determinant:** Given $A$, compute $\det(A) = 14$:

$$
A =
\begin{bmatrix}
5 & 3 \\
2 & 4
\end{bmatrix}
$$

- [ ] **8. 2x2 Matrix Inverse:** Given $A$, compute $A^{-1}$ and verify $A A^{-1} = I$:

$$
A =
\begin{bmatrix}
4 & 7 \\
2 & 6
\end{bmatrix}
\implies
A^{-1} =
\begin{bmatrix}
0.6 & -0.7 \\
-0.2 & 0.4
\end{bmatrix}
$$

- [ ] **9. Solve Linear System:** Convert $2x+y=5, x-y=1$ into $A\mathbf{x}=\mathbf{b}$ and solve $\mathbf{x} = [2, 1]^T$.
- [ ] **10. Check Linear Independence:** Prove that $[1, 2]^T$ and $[2, 4]^T$ are dependent ($\det = 0$).
- [ ] **11. Sample Covariance Table:** Given $X = [1, 2, 3], Y = [2, 3, 7]$, compute $\text{Var}(X)=1, \text{Var}(Y)=7, \text{Cov}(X,Y)=2.5$.
- [ ] **12. Build Covariance Matrix:** Construct $\Sigma$ from scratch:

$$
\Sigma =
\begin{bmatrix}
1.0 & 2.5 \\
2.5 & 7.0
\end{bmatrix}
$$

- [ ] **13. Characteristic Equation:** Set up $\det(A - \lambda I) = 0$ for $A$ and get $\lambda^2 - 4\lambda + 3 = 0$:

$$
A =
\begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}
$$

- [ ] **14. Derive Eigenvalues:** Solve $\lambda_1 = 3, \lambda_2 = 1$.
- [ ] **15. Derive Eigenvectors:** Substitute $\lambda=3$ to find $\mathbf{v}_1 = [1, 1]^T$ and $\lambda=1$ to find $\mathbf{v}_2 = [1, -1]^T$.
- [ ] **16. Verify Eigenvectors:** Multiply $A \mathbf{v}_1$ and show it equals $3 \mathbf{v}_1$.
- [ ] **17. Project Vector:** Project $\mathbf{a} = [3, 4]^T$ onto $\mathbf{b} = [4, 0]^T$ to get $\mathbf{p} = [3, 0]^T$.
- [ ] **18. Normal Equation:** Write $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$ and explain every term.
- [ ] **19. Partial Derivatives:** Compute $\frac{\partial}{\partial x}[x^2 + 3xy + y^2] = 2x + 3y$.
- [ ] **20. Gradient Descent Step:** Given $f(x)=x^2, x_0=4, \alpha=0.1$, compute step $x_1 = 3.2, x_2 = 2.56$.
- [ ] **21. Chain Rule:** Differentiate $y = (3x + 2)^2 \implies \frac{dy}{dx} = 18x + 12$.
- [ ] **22. Entropy Calculation:** Compute $H(0.5, 0.5) = 1.0 \text{ Bit}$ and $H(1.0, 0.0) = 0.0 \text{ Bits}$.
- [ ] **23. Four Subspaces & Null Space:** For the $2 \times 2$ matrix with rows $[1, 2]$ and $[2, 4]$, find $N(A) = \text{span}([-2, 1]^T)$ and verify $\text{rank}(A) + \text{nullity}(A) = 2$.
- [ ] **24. Gram-Schmidt Orthogonalization:** Orthonormalize $\mathbf{a}_1 = [1, 1]^T, \mathbf{a}_2 = [1, 0]^T$ to find $\mathbf{q}_1 = [1/\sqrt{2}, 1/\sqrt{2}]^T, \mathbf{q}_2 = [1/\sqrt{2}, -1/\sqrt{2}]^T$.
- [ ] **25. Construct QR Decomposition:** Factor $A = QR$ and construct the upper-triangular matrix $R$ with diagonal entries $[\sqrt{2}, 1/\sqrt{2}]$.
- [ ] **26. Positive Definite Test:** Compute quadratic form $\mathbf{x}^T A \mathbf{x} = 2x_1^2 + 2x_1 x_2 + 2x_2^2$ for matrix $A$ (eigenvalues $\lambda_1=3, \lambda_2=1 \gt 0$) and prove it is Positive Definite.
- [ ] **27. Why Covariance is PSD:** Prove $\mathbf{x}^T \Sigma \mathbf{x} = \frac{1}{n-1} \|X_c \mathbf{x}\|_2^2 \ge 0$ for any vector $\mathbf{x}$.
- [ ] **28. Moore-Penrose Pseudoinverse:** For column vector $A = [1, 2]^T$, compute $A^+ = (A^T A)^{-1} A^T = [0.2, 0.4]$ and verify $A^+ A = [1.0]$.
- [ ] **29. Linear Transformation Geometry:** Apply 2D Rotation $R_{90^\circ}$ (mapping $[1, 0]^T \to [0, 1]^T$) and Scaling to vectors on paper.
- [ ] **30. SVD Geometric Decomposition:** Interpret $A = U \Sigma V^T$ as Rotate $\to$ Stretch $\to$ Rotate.

---

---

# 🚀 ADVANCED TOPICS CONTINUED IN PART 2

Parts 21 through 26 and the One-Page Master Formula Cheat Sheet are continued in Part 2:

👉 **[Click here to open LINEAR_ALGEBRA_AND_MATH_PART_2.md](LINEAR_ALGEBRA_AND_MATH_PART_2.md)**

### Contents Covered in Part 2:
* **Part 21:** 40 Essential Technical Interview Questions (with in-depth explanations & formulas)
* **Part 22:** Four Fundamental Subspaces ($C(A), C(A^T), N(A), N(A^T)$, Orthogonal Complements, Rank-Nullity Theorem)
* **Part 23:** Gram-Schmidt Orthogonalization & QR Decomposition (Gram-Schmidt derivation, $A=QR$, & Least Squares stability)
* **Part 24:** Positive Definite & Positive Semidefinite Matrices (Quadratic Forms $\mathbf{x}^T A \mathbf{x}$, Hessian, & Convexity)
* **Part 25:** Moore-Penrose Pseudoinverse ($A^+$, Minimum-Norm OLS, & SVD formulation)
* **Part 26:** Linear Transformations (Geometry of Rotation/Reflection/Scaling/Shear, Eigenvector Invariance, & SVD Geometry)
* **Final Section:** One-Page Master Formula Cheat Sheet
