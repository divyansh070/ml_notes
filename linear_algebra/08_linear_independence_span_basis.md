> 📖 **Navigation:** [← Previous: Part 07: Matrix Rank](./07_matrix_rank.md) | [🏠 Index](./README.md) | [Next: Part 09: Vector Spaces & Subspaces →](./09_vector_spaces_and_subspaces.md)

---

# PART 8 — LINEAR INDEPENDENCE, SPAN, BASIS & DIMENSION

To understand how datasets occupy space and why dimensionality reduction works, we need four closely linked concepts:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     THE FOUR-CONCEPT FOUNDATION CHAIN                       │
│                                                                             │
│  Linear Combination ──► Span (All Reachable Points)                         │
│                               │                                             │
│                               ▼                                             │
│  Linear Independence ──► BASIS (Minimal Coordinate System) ──► Dimension   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8.1 Linear Combinations

A **linear combination** scales a set of vectors $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\}$ by scalars $c_i \in \mathbb{R}$ and sums them:

$$
\mathbf{y} = c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_k \mathbf{v}_k = \sum_{i=1}^{k} c_i \mathbf{v}_i
$$

Varying the scalar coefficients $c_1, \dots, c_k$ slides along each vector direction, navigating to different locations across space.

---

## 8.2 The Concept of Span

The **Span** of a set of vectors is the set of **ALL possible linear combinations** that can be formed using them:

$$
\text{span}(\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k) = \left\lbrace \sum_{i=1}^{k} c_i \mathbf{v}_i \;\middle|\; c_i \in \mathbb{R} \right\rbrace
$$

### Geometric Examples in $\mathbb{R}^2$ and $\mathbb{R}^3$:

```
   1 Vector in R^2:              2 Non-Collinear Vectors in R^2:      2 Vectors in R^3:
   Span is a 1D Line             Span is the entire 2D Plane          Span is a 2D Plane in 3D
          y                                   y                                 z
          │     / Span                        │      / Span = R^2               │    / (2D Sheet)
          │    / (Line)                       │     / (Whole Plane)             │   / 
          │   ● v1                            │    /                            │  ●───────●
          └───┴─────► x                       └───┴───────► x                   └──┴───────┴──► y
                                                                                  / x
```

* **1 non-zero vector:** Span is a **1D line** passing through the origin.
* **2 non-parallel vectors in $\mathbb{R}^2$:** Span is the **entire 2D plane** $\mathbb{R}^2$.
* **2 vectors in $\mathbb{R}^3$:** Span is a **2D flat sheet (plane)** slicing through 3D space.
* **3 non-coplanar vectors in $\mathbb{R}^3$:** Span is the **entire 3D volume** $\mathbb{R}^3$.

---

## 8.3 Linear Independence vs. Linear Dependence

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

### 1. Formal Definition
A set of vectors $\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$ is **Linearly Independent** if the equation:

$$
c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_k \mathbf{v}_k = \mathbf{0}
$$

has **ONLY the trivial solution**:

$$
c_1 = c_2 = \dots = c_k = 0
$$

* If there exists **any** set of scalars not all zero ($c_i \neq 0$) such that the sum equals $\mathbf{0}$, the vectors are **Linearly Dependent**.
* **Intuition:** Linear dependence means at least one vector is **redundant**—it can be expressed as a linear combination of the others and adds zero new geometric dimensions to the span.

### 2. Computational Test via Matrix Ranks:
Pack the vectors into columns of a matrix $A = \begin{bmatrix} \mathbf{v}_1 & \dots & \mathbf{v}_k \end{bmatrix}$.
* Independent $\iff A\mathbf{c} = \mathbf{0}$ has only $\mathbf{c} = \mathbf{0} \iff \text{rank}(A) = k$ (Full Column Rank).

---

## 8.4 The Concept of a Basis

A **basis** is the minimal, non-redundant coordinate scaffolding for a vector space.

> [!IMPORTANT]
> **Definition of a Basis:**
> A set of vectors $\mathcal{B} = \{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\}$ is a **Basis** for space $V$ if and only if:
> 1. **Linearly Independent:** No vector in $\mathcal{B}$ is redundant.
> 2. **Spans the Space:** $\text{span}(\mathbf{v}_1, \dots, \mathbf{v}_k) = V$.

* **Standard Basis for $\mathbb{R}^2$:** $\mathbf{e}_1 = [1, 0]^T, \mathbf{e}_2 = [0, 1]^T$.
* **Bases are NOT Unique:** Any two non-collinear vectors in $\mathbb{R}^2$ form a completely valid basis for $\mathbb{R}^2$.
* **Coordinates Relative to a Basis:** If $\mathcal{B}$ is a basis, every vector $\mathbf{x} \in V$ can be written uniquely as $\mathbf{x} = c_1 \mathbf{v}_1 + \dots + c_k \mathbf{v}_k$. The scalar weights $[c_1, \dots, c_k]^T$ are the **coordinates of $\mathbf{x}$ in basis $\mathcal{B}$**.

---

## 8.5 Dimension of a Vector Space

The **Dimension** of a vector space $V$, denoted $\dim(V)$, is the **number of vectors in any basis for $V$**.

* Every basis for the same vector space has the **exact same number of vectors**.
* $\dim(\mathbb{R}^2) = 2$, $\dim(\mathbb{R}^3) = 3$, $\dim(\mathbb{R}^d) = d$.
* A 2D plane slicing through 100-dimensional space $\mathbb{R}^{100}$ has $\text{dimension} = 2$.

---

## 8.6 "These Vectors Span a Space" vs. "These Vectors Form a Basis"

This distinction is essential:

```
┌───────────────────────────────────────┬───────────────────────────────────────┐
│ "These Vectors Span Space V"          │ "These Vectors Form a Basis for V"    │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ • Can reach every point in V          │ • Can reach every point in V          │
│ • Redundancy is ALLOWED               │ • Redundancy is STRICTLY FORBIDDEN    │
│ • May be linearly dependent           │ • MUST be linearly independent        │
│ • Size can be ≥ dim(V)                │ • Size MUST be EXACTLY equal to dim(V)│
│ • Example: 3 non-parallel 2D vectors  │ • Example: Exactly 2 non-parallel 2D  │
│   span R^2 (one is redundant)         │   vectors form a basis for R^2        │
└───────────────────────────────────────┴───────────────────────────────────────┘
```

---

## 8.7 Complete Worked Numerical Example

Test whether the following vectors form a basis for $\mathbb{R}^3$:

$$
\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \\ 2 \end{bmatrix}, \quad
\mathbf{v}_2 = \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}, \quad
\mathbf{v}_3 = \begin{bmatrix} 1 \\ 2 \\ 4 \end{bmatrix}
$$

### Step 1: Form Matrix $A$ and Check Independence

$$
A = \begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 2 \\
2 & 1 & 4
\end{bmatrix}
$$

### Step 2: Row Reduce to Check Pivots
Eliminate Column 1 below pivot 1: $R_3 \leftarrow R_3 - 2R_1$:

$$
[2, 1, 4] - 2[1, 0, 1] = [0, 1, 2]
$$

Matrix becomes:

$$
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 2 \\
0 & 1 & 2
\end{bmatrix}
$$

Eliminate Column 2 below pivot 2: $R_3 \leftarrow R_3 - R_2$:

$$
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 2 \\
0 & 0 & 0
\end{bmatrix}
$$

### Step 3: Analyze the Result
* There are only **2 pivots** $\implies \text{rank}(A) = 2 < 3$.
* The vectors are **Linearly Dependent**:

$$
\mathbf{v}_3 = 1 \mathbf{v}_1 + 2 \mathbf{v}_2 = \begin{bmatrix} 1 \\ 0 \\ 2 \end{bmatrix} + \begin{bmatrix} 0 \\ 2 \\ 2 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \\ 4 \end{bmatrix}
$$

* **Span:** $\text{span}(\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3)$ is only a **2D plane** embedded in 3D space ($\text{dimension} = 2$).
* **Conclusion:** Because the vectors are linearly dependent, they **DO NOT form a basis for $\mathbb{R}^3$**.

---

## 8.8 Why this matters in ML

1. **Dimensionality Reduction (PCA & Autoencoders):** High-dimensional feature spaces ($d = 10,000$) often contain vast numbers of redundant, dependent features. PCA finds a new, orthogonal basis of $k \ll d$ dimensions that spans $99\%$ of data variance.
2. **Latent Representation Spaces:** In variational autoencoders (VAEs) and large language model token embeddings, the model projects raw data into an intrinsic, lower-dimensional latent space basis.
3. **Weight Initialization & Matrix Rank:** In neural networks, if weight rows in a layer become linearly dependent (representation collapse), capacity is wasted because the output space collapses to a lower dimension.

---

## 8.9 Common mistakes

* **Confusing Dimension with Number of Vectors:** A set of 10 vectors in $\mathbb{R}^3$ does not have dimension 10. The dimension of their span cannot exceed $\min(10, 3) = 3$.
* **Believing a Basis is Unique:** A space has infinitely many different bases. For example, any rotated coordinate system forms a completely valid new basis.
* **Assuming Linearly Independent Vectors Must Be Orthogonal:** Independence only requires that no vector is a linear combination of the others. Orthogonality is a much stricter condition ($\mathbf{u} \cdot \mathbf{v} = 0$).

---

> 📖 **Navigation:** [← Previous: Part 07: Matrix Rank](./07_matrix_rank.md) | [🏠 Index](./README.md) | [Next: Part 09: Vector Spaces & Subspaces →](./09_vector_spaces_and_subspaces.md)
