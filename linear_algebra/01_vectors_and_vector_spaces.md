> 📖 **Navigation:** [← Return to Index](./README.md) | [🏠 Index](./README.md) | [Next: Part 02: Matrices & Matrix Operations →](./02_matrices_and_operations.md)

---

# PART 1 — VECTORS, SPAN & VECTOR SPACES

Linear algebra begins with the **vector**. In machine learning, almost all data—from tabular features and image pixels to LLM embeddings—is represented as vectors in multi-dimensional space.

---

## 1.1 The Three Interpretations of a Vector

A vector is not just a list of numbers; it has three distinct, equally important interpretations:

```
        1. GEOMETRIC ARROW                2. POINT IN SPACE               3. FEATURE VECTOR (ML)
     (Direction + Magnitude)              (Coordinates in R^d)              (Sample Attributes)
               y                                  y
               │      / x = [2, 3]^T              │        ● (2, 3)             House Sample 1:
             3 ┼─────►                          3 ┼───────┘                     [ Bedrooms: 2 ]
               │    /                             │                             [ Bathrooms: 3]
               │   /                              │                             [ Area (k): 1.5]
               └──┴────────► x                    └───┴────────► x              Shape: (d x 1)
                  2                                   2
```

1. **Computer Science / ML View (Feature Vector):** An ordered 1D array of $d$ numerical features representing a single sample:
   $$
   \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_d \end{bmatrix} \in \mathbb{R}^d
   $$
2. **Physics View (Arrow):** An object with a specific **magnitude (length)** and **direction** in space, starting at the origin $\mathbf{0}$.
3. **Geometry View (Point):** A static coordinate address $(x_1, x_2, \dots, x_d)$ in $d$-dimensional space.

* **Scalar:** A single real number ($c \in \mathbb{R}$) representing pure magnitude without direction (e.g., learning rate $\alpha = 0.01$, loss $\mathcal{L} = 0.42$).

---

## 1.2 Vector Addition & Scalar Multiplication

Vector spaces are built on two fundamental operations:

### 1. Vector Addition
Add corresponding components element-by-element:
$$
\mathbf{a} + \mathbf{b} =
\begin{bmatrix} a_1 \\ a_2 \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \end{bmatrix} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \end{bmatrix}
$$

* **Geometric Tip-to-Tail Rule:** Placing the tail of $\mathbf{b}$ at the tip of $\mathbf{a}$ produces the diagonal of a parallelogram.
* **ML Role:** Adding a bias vector $\mathbf{b}$ to linear logits ($\mathbf{z} = W\mathbf{x} + \mathbf{b}$) shifts the decision boundary in space.

### 2. Scalar Multiplication
Multiply every component by scalar $c \in \mathbb{R}$:
$$
c \mathbf{x} = \begin{bmatrix} c x_1 \\ c x_2 \end{bmatrix}
$$

* **Geometric Effect:** $c > 1$ stretches the arrow; $0 < c < 1$ shrinks it; $c < 0$ reverses its direction by $180^\circ$.
* **ML Role:** The gradient descent step $\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla \mathcal{L}$ scales the direction of steepest descent by learning rate $\alpha$.

---

## 1.3 Linear Combinations

A **Linear Combination** of a set of vectors $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\}$ is the vector obtained by scaling each vector by a scalar coefficient $c_i$ and adding them together:

$$
\mathbf{y} = c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_k \mathbf{v}_k = \sum_{i=1}^{k} c_i \mathbf{v}_i
$$

```
                         LINEAR COMBINATION GEOMETRY
                                  y
                                  │           c1*v1 + c2*v2
                                  │              ●
                                  │             / ╱
                                  │     c1*v1  / ╱ c2*v2
                                  │           ● ╱
                                  │          /
                                  └─────────┴────────► x
```

* **What varying $c_i$ does geometrically:** Changing the coefficients $c_1, c_2$ allows you to slide along each vector's direction, reaching different points across the coordinate grid.
* **Matrix Connection:** As we will see in [Part 02](./02_matrices_and_operations.md), matrix-vector multiplication $A\mathbf{x}$ is literally a **linear combination of the columns of matrix $A$**.

---

## 1.4 The Concept of Span

The **Span** of a set of vectors $S = \{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\}$ is the set of **ALL possible linear combinations** that can be created using those vectors:

$$
\text{span}(\mathbf{v}_1, \dots, \mathbf{v}_k) = \left\lbrace \sum_{i=1}^{k} c_i \mathbf{v}_i \;\middle|\; c_i \in \mathbb{R} \right\rbrace
$$

```
                           SPAN VISUALIZATION IN 2D & 3D
   1 Vector in R^2:              2 Non-Collinear Vectors in R^2:      2 Vectors in R^3:
   Span is a 1D Line             Span is the entire 2D Plane          Span is a 2D Plane in 3D
          y                                   y                                 z
          │     / Span                        │      / Span = R^2               │    / (2D Sheet)
          │    / (Line)                       │     / (Whole Plane)             │   / 
          │   ● v1                            │    /                            │  ●───────●
          └───┴─────► x                       └───┴───────► x                   └──┴───────┴──► y
                                                                                  / x
```

* **Span of 1 non-zero vector:** A 1-dimensional line passing through the origin.
* **Span of 2 non-parallel vectors in $\mathbb{R}^2$:** The entire 2-dimensional plane $\mathbb{R}^2$.
* **Span of 2 vectors in $\mathbb{R}^3$:** A 2-dimensional flat sheet (plane) slicing through 3D space.
* **ML Connection:** When fitting a linear regression model $\hat{\mathbf{y}} = X\mathbf{w}$, our prediction $\hat{\mathbf{y}}$ is strictly constrained to lie within the **Span of the feature columns** ($\text{Col}(X)$).

---

## 1.5 Affine Combinations vs. Linear Combinations

It is crucial in machine learning to distinguish between **Linear Spaces** and **Affine Spaces**:

```
           LINEAR SUBSPACE (Passes through Origin)         AFFINE HYPERPLANE (Translated by Bias b)
                      y                                               y
                      │      / y = Ax                                 │      / y = Ax + b
                      │     /                                         │     / (Shifted by +b)
                      │    /                                        b ┼────●
                      └───●──────► x                                  │   /
                       Origin (0, 0)                                  └──┴────────► x
```

1. **Linear Transformation ($A\mathbf{x}$):** Always passes through the origin $\mathbf{0}$ ($A\mathbf{0} = \mathbf{0}$). A linear subspace must contain $\mathbf{0}$.
2. **Affine Transformation ($A\mathbf{x} + \mathbf{b}$):** A linear transformation shifted by a fixed translation vector $\mathbf{b}$. 
   * *Why bias matters in ML:* Without bias $\mathbf{b}$, a neural network layer or logistic regression hyperplane is forced to pass through $(0, 0, \dots, 0)$, severely limiting its ability to separate data.

---

## 1.6 Vector Dot Product & Geometric Angle

The **Dot Product** (inner product) takes two vectors and produces a single scalar:

$$
\mathbf{a} \cdot \mathbf{b} = \mathbf{a}^T \mathbf{b} = \sum_{i=1}^{d} a_i b_i = a_1 b_1 + a_2 b_2 + \dots + a_d b_d
$$

### Geometric Formula & Angle $\theta$:
$$
\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|_2 \|\mathbf{b}\|_2 \cos\theta
$$

$$
\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}
$$

* **Orthogonality Condition:** Two non-zero vectors are perpendicular ($\theta = 90^\circ$) if and only if their dot product is zero:
  $$
  \mathbf{a} \perp \mathbf{b} \iff \mathbf{a} \cdot \mathbf{b} = 0
  $$
* **The Cauchy-Schwarz Bound:** The absolute value of the dot product is always bounded by the product of their lengths:
  $$
  |\mathbf{a} \cdot \mathbf{b}| \le \|\mathbf{a}\|_2 \|\mathbf{b}\|_2
  $$

---

## 1.7 Vector Norms & Distances

A **norm** $\|\mathbf{x}\|$ measures the geometric length or magnitude of a vector.

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

1. **$L_2$ Norm (Euclidean Length):**
   $$
   \|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{d} x_i^2} = \sqrt{\mathbf{x}^T \mathbf{x}}
   $$
2. **$L_1$ Norm (Manhattan / Taxicab Length):**
   $$
   \|\mathbf{x}\|_1 = \sum_{i=1}^{d} |x_i|
   $$
3. **Euclidean Distance between Points $\mathbf{p}$ and $\mathbf{q}$:**
   $$
   d(\mathbf{p}, \mathbf{q}) = \|\mathbf{p} - \mathbf{q}\|_2 = \sqrt{\sum_{i=1}^{d} (p_i - q_i)^2}
   $$

---

## 1.8 Cosine Similarity & High-Dimensional Embeddings

**Cosine Similarity** measures the angular alignment between two vectors regardless of their magnitude:

$$
\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \cos\theta \in [-1.0, +1.0]
$$

* $+1.0 \implies$ Pointing in identical direction ($\theta = 0^\circ$).
* $0.0 \implies$ Orthogonal / completely unrelated ($\theta = 90^\circ$).
* $-1.0 \implies$ Directly opposite directions ($\theta = 180^\circ$).

* **Why Cosine Similarity Dominates NLP & Vector Databases:** In high-dimensional text embeddings (Word2Vec, OpenAI, BERT), document length inflates vector magnitude without changing semantic topic. Cosine similarity normalizes out length, comparing pure semantic direction.

---

## 1.9 Vector Subspaces & The 3-Step Verification Test

A subset $S \subseteq \mathbb{R}^d$ is a **Vector Subspace** if it is closed under linear combinations.

### The 3 Subspace Axioms:
1. **Contains Zero:** $\mathbf{0} \in S$.
2. **Closed under Addition:** If $\mathbf{u}, \mathbf{v} \in S$, then $\mathbf{u} + \mathbf{v} \in S$.
3. **Closed under Scaling:** If $\mathbf{u} \in S$ and $c \in \mathbb{R}$, then $c\mathbf{u} \in S$.

> [!IMPORTANT]
> **Subspace Rule:** Any line, plane, or hyperplane that **does NOT pass through the origin $(0, 0, \dots, 0)$ is NOT a subspace** (it fails axiom 1).

---

## Advanced / Optional — Do not study until Core track is complete

### A.1 Full Proof of the Cauchy-Schwarz Inequality
For any real scalar $t \in \mathbb{R}$, the squared norm $\|\mathbf{u} - t\mathbf{v}\|_2^2 \ge 0$:
$$
\|\mathbf{u} - t\mathbf{v}\|_2^2 = \|\mathbf{u}\|_2^2 - 2t(\mathbf{u} \cdot \mathbf{v}) + t^2 \|\mathbf{v}\|_2^2 \ge 0
$$
This is a quadratic $A t^2 + B t + C \ge 0$ where $A = \|\mathbf{v}\|_2^2, B = -2(\mathbf{u} \cdot \mathbf{v}), C = \|\mathbf{u}\|_2^2$.
Because this quadratic is non-negative for all $t$, its discriminant $\Delta = B^2 - 4AC \le 0$:
$$
[-2(\mathbf{u} \cdot \mathbf{v})]^2 - 4(\|\mathbf{v}\|_2^2)(\|\mathbf{u}\|_2^2) \le 0 \implies |\mathbf{u} \cdot \mathbf{v}| \le \|\mathbf{u}\|_2 \|\mathbf{v}\|_2 \quad \blacksquare
$$

### A.2 General $L_p$ Norms, Minkowski & Hölder Inequalities
* **General $L_p$ Norm:** $\|\mathbf{x}\|_p = \left(\sum |x_i|^p\right)^{1/p}$ for $p \ge 1$.
* **$L_\infty$ Max Norm:** $\|\mathbf{x}\|_\infty = \max_i |x_i|$.
* **Minkowski Inequality:** $\|\mathbf{u} + \mathbf{v}\|_p \le \|\mathbf{u}\|_p + \|\mathbf{v}\|_p$.
* **Hölder's Inequality:** $|\mathbf{x}^T \mathbf{y}| \le \|\mathbf{x}\|_p \|\mathbf{y}\|_q$ where $\frac{1}{p} + \frac{1}{q} = 1$.

### A.3 Why the "$L_0$ Norm" is a Pseudo-Norm
$\|\mathbf{x}\|_0 = \sum \mathbb{I}(x_i \ne 0)$ counts non-zero entries. It violates absolute scalability: $\|c\mathbf{x}\|_0 = \|\mathbf{x}\|_0 \ne |c|\|\mathbf{x}\|_0$ for $c \ne 0$. Thus $L_0$ is non-convex; $L_1$ (Lasso) is its tightest convex surrogate.

### A.4 Formal Inner Product Space Axioms
An inner product $\langle \mathbf{u}, \mathbf{v} \rangle$ satisfies:
1. Symmetry: $\langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle$
2. Linearity in 1st argument: $\langle a\mathbf{u} + b\mathbf{v}, \mathbf{w} \rangle = a\langle \mathbf{u}, \mathbf{w} \rangle + b\langle \mathbf{v}, \mathbf{w} \rangle$
3. Positive-definiteness: $\langle \mathbf{u}, \mathbf{u} \rangle \ge 0$, and $\langle \mathbf{u}, \mathbf{u} \rangle = 0 \iff \mathbf{u} = \mathbf{0}$.

---

> 📖 **Navigation:** [← Return to Index](./README.md) | [🏠 Index](./README.md) | [Next: Part 02: Matrices & Matrix Operations →](./02_matrices_and_operations.md)
