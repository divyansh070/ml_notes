> 📖 **Navigation:** [← Previous: Part 05: Systems of Linear Equations](./05_systems_of_linear_equations.md) | [🏠 Index](./README.md) | [Next: Part 07: Eigenvalues & Eigenvectors →](./07_eigenvalues_and_eigenvectors.md)

---

# PART 6 — LINEAR INDEPENDENCE, BASIS, DIMENSION & RANK

This module forms the conceptual bridge from vectors and systems of equations to vector spaces, dimensionality reduction, and matrix rank.

```
                    THE CORE LINEAR ALGEBRA CONCEPT CHAIN
  Vectors  ──►  Linear Combinations  ──►  Span  ──►  Linear Independence
                                                            │
  Rank-Nullity  ◄──  Matrix Rank  ◄──  Dimension  ◄──  Vector Basis
```

---

## 6.1 Linear Combinations & Span

### 1. Linear Combination
Given vectors $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\}$, a **linear combination** scales each vector by a real number $c_i$ and sums them:
$$
\mathbf{y} = c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_k \mathbf{v}_k
$$

### 2. Span
The **Span** is the set of **ALL vectors** that can be created through every possible combination of coefficients $c_i$:
$$
\text{span}(\mathbf{v}_1, \dots, \mathbf{v}_k) = \left\lbrace \sum_{i=1}^{k} c_i \mathbf{v}_i \;\middle|\; c_i \in \mathbb{R} \right\rbrace
$$
* If 2 vectors point in different directions in 2D, their span is the entire 2D plane ($\mathbb{R}^2$).
* If 2 vectors lie on the exact same line, their span collapses into a 1D line.

---

## 6.2 Linear Independence vs. Linear Dependence

```
          LINEARLY INDEPENDENT VECTORS                  LINEARLY DEPENDENT VECTORS
             (New Dimension Added)                      (Redundant: Lies on Same Line)
                      y                                               y
                      │      / v2                                     │          v2 = 2 * v1
                      │     /                                         │         ●
                      │    /                                          │        ╱
                      │   ● v1                                        │   ● v1╱
                      └───┴────────► x                                └───┴───┴────► x
                     Span = 2D Plane                                  Span = 1D Line
```

### The Formal Definition:
A set of vectors $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\}$ is **Linearly Independent** if the equation:
$$
c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_k \mathbf{v}_k = \mathbf{0}
$$
has **ONLY the trivial solution**:
$$
c_1 = c_2 = \dots = c_k = 0
$$

* If there exists **any** non-zero choice of scalars ($c_i \neq 0$) such that the sum equals $\mathbf{0}$, the vectors are **Linearly Dependent**.
* **Intuition:** At least one vector is **redundant**—it can be written as a combination of the others and adds no new geometric dimensions to the span.

### Hand Check for Linear Independence:
Test vectors $\mathbf{v}_1 = [1, 2]^T$ and $\mathbf{v}_2 = [2, 4]^T$:
$$
c_1 \begin{bmatrix} 1 \\ 2 \end{bmatrix} + c_2 \begin{bmatrix} 2 \\ 4 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$
Choose $c_1 = 2, c_2 = -1$:
$$
2 \begin{bmatrix} 1 \\ 2 \end{bmatrix} - 1 \begin{bmatrix} 2 \\ 4 \end{bmatrix} = \begin{bmatrix} 2 - 2 \\ 4 - 4 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$
Because non-zero scalars exist, $\mathbf{v}_1$ and $\mathbf{v}_2$ are **Linearly Dependent!**

---

## 6.3 The Concept of a Basis

A **Basis** is the "minimal coordinate system" for a vector space.

> [!IMPORTANT]
> **Definition of a Basis:**
> A set of vectors $\mathcal{B} = \{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\}$ is a **Basis** for vector space $V$ if:
> 1. **Linearly Independent:** No vector in $\mathcal{B}$ is redundant.
> 2. **Spans the Space:** $\text{span}(\mathbf{v}_1, \dots, \mathbf{v}_k) = V$ (Every vector in $V$ can be built).

```
          THE STANDARD BASIS FOR R^2                  A NON-STANDARD BASIS FOR R^2
                      y                                               y
                      │                                               │      / b2 = [1, 2]^T
                    1 ┼───● e2 = [0, 1]^T                             │     / 
                      │   │                                           │    ●   ● b1 = [2, 1]^T
                      └───●──────► x                                  └───┴────┴────► x
                          1 e1 = [1, 0]^T
```

* **The Standard Basis for $\mathbb{R}^2$:** $\mathbf{e}_1 = [1, 0]^T, \mathbf{e}_2 = [0, 1]^T$.
* **Bases are NOT Unique:** Any two non-parallel vectors in $\mathbb{R}^2$ form a valid basis for $\mathbb{R}^2$.
* **Coordinates relative to a basis:** The unique weights $c_1, c_2$ required to write $\mathbf{x} = c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2$ are the **coordinates of $\mathbf{x}$ in basis $\mathcal{B}$**.

---

## 6.4 Dimension of a Vector Space

The **Dimension** ($\dim(V)$) of a vector space $V$ is the **number of vectors in any basis for $V$**.

* $\dim(\mathbb{R}^2) = 2$ (requires 2 basis vectors).
* $\dim(\mathbb{R}^3) = 3$ (requires 3 basis vectors).
* A line passing through the origin has $\text{dimension} = 1$, even if embedded inside 100-dimensional space $\mathbb{R}^{100}$!

---

## 6.5 Matrix Rank: The Number of True Dimensions

The **Rank** of a matrix $A$, denoted $\text{rank}(A)$, is the **number of linearly independent column vectors (or row vectors) in $A$**.

$$
\text{rank}(A) = \dim(\text{Col}(A)) = \dim(\text{Row}(A)) \le \min(m, n)
$$

```
                         GEOMETRIC MEANING OF MATRIX RANK
   A matrix transformation A @ x maps input space into output space.
   Matrix Rank = The NUMBER OF GEOMETRIC DIMENSIONS that survive the transformation!
```

### 1. Full Rank Matrix
* A matrix is **Full Column Rank** if $\text{rank}(A) = n$ (all columns independent; no information squashed).
* A square matrix is **Full Rank** if $\text{rank}(A) = n \iff \det(A) \neq 0 \iff A^{-1}$ exists.

### 2. Rank-Deficient Matrix
* If $\text{rank}(A) < \min(m, n)$, the columns are dependent. Space is flattened, and information is lost.

---

## 6.6 Nullity & The Null Space

The **Null Space** $\text{Null}(A)$ (or Kernel) is the set of all vectors $\mathbf{x}$ mapped to zero:

$$
\text{Null}(A) = \left\lbrace \mathbf{x} \in \mathbb{R}^n \mid A\mathbf{x} = \mathbf{0} \right\rbrace
$$

* **Nullity:** The **dimension of the Null Space**:
  $$
  \text{nullity}(A) = \dim(\text{Null}(A))
  $$
* **Geometric Meaning:** $\text{nullity}(A)$ is the exact **number of dimensions completely flattened/destroyed by matrix $A$**.

---

## 6.7 The Rank-Nullity Theorem (The Fundamental Conservation Law)

For any matrix $A \in \mathbb{R}^{m \times n}$ with $n$ input columns:

$$
\text{rank}(A) + \text{nullity}(A) = n
$$

```
     ┌────────────────────────────────────────────────────────────────────────┐
     │  (Surviving Dimensions)  +  (Destroyed Dimensions)  =  (Total Columns) │
     │         rank(A)          +       nullity(A)         =        n         │
     └────────────────────────────────────────────────────────────────────────┘
```

---

## 6.8 Recurring Hand Calculation Examples

### Example 1: Full Rank Matrix
$$
A = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$
* Columns $[1, 0]^T$ and $[0, 1]^T$ are independent $\implies \text{rank}(A) = 2$.
* Only $\mathbf{x} = [0, 0]^T$ satisfies $A\mathbf{x} = \mathbf{0} \implies \text{nullity}(A) = 0$.
* Check: $\text{rank}(A) + \text{nullity}(A) = 2 + 0 = 2 = n \quad \checkmark$.

### Example 2: Rank-Deficient Matrix
$$
A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}
$$
* Column 2 is $2 \times$ Column 1 $\implies \text{rank}(A) = 1$ (Surviving dimension is a 1D line).
* Solve $A\mathbf{x} = \mathbf{0}$: $x_1 + 2x_2 = 0 \implies \mathbf{x} = t [-2, 1]^T \implies \text{nullity}(A) = 1$.
* Check: $\text{rank}(A) + \text{nullity}(A) = 1 + 1 = 2 = n \quad \checkmark$.

---

## 6.9 Machine Learning Impact: Multicollinearity & The Dummy Variable Trap

1. **Multicollinearity:** If two feature columns are linearly dependent (e.g. `weight_kg = 2.204 * weight_lbs`), the design matrix $X$ loses full column rank ($\text{rank}(X) < d$).
   * Consequently, $(X^T X)$ has $\det(X^T X) = 0$ and is non-invertible! OLS regression fails.
2. **The Dummy Variable Trap:** When one-hot encoding a categorical variable with $K$ categories without dropping the reference category, the sum of the $K$ columns equals the intercept column $\mathbf{1}$, creating an exact linear dependence ($\text{rank} < d$).

---

> 📖 **Navigation:** [← Previous: Part 05: Systems of Linear Equations](./05_systems_of_linear_equations.md) | [🏠 Index](./README.md) | [Next: Part 07: Eigenvalues & Eigenvectors →](./07_eigenvalues_and_eigenvectors.md)
