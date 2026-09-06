> 📖 **Navigation:** [← Previous: Part 11: Singular Value Decomposition (SVD)](./11_singular_value_decomposition.md) | [🏠 Index](./README.md) | [Next: Part 13: Vector Projections →](./13_vector_projections.md)

---

# PART 12 — ORTHOGONALITY & ORTHONORMAL BASES

**Orthogonality** (perpendicularity) is the foundation for projections, least squares linear regression, QR decomposition, and Fourier/wavelet analysis.

---

## 12.1 Orthogonal & Orthonormal Vectors

1. **Orthogonal Vectors:** Two vectors $\mathbf{u}, \mathbf{v}$ are orthogonal if their dot product is zero:
   $$
   \mathbf{u} \perp \mathbf{v} \iff \mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T \mathbf{v} = 0
   $$
2. **Orthonormal Vectors:** Vectors that are both mutually orthogonal AND normalized to unit length ($\|\mathbf{u}\|_2 = 1$):
   $$
   \mathbf{q}_i^T \mathbf{q}_j =
   \begin{cases}
   1 & \text{if } i = j \\
   0 & \text{if } i \neq j
   \end{cases}
   $$

---

## 12.2 Proof: Why Orthogonal Non-Zero Vectors are ALWAYS Linearly Independent

Let $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\}$ be mutually orthogonal non-zero vectors.
Set their linear combination to zero:
$$
c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_k \mathbf{v}_k = \mathbf{0}
$$
Take the dot product of both sides with vector $\mathbf{v}_1$:
$$
\mathbf{v}_1 \cdot (c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_k \mathbf{v}_k) = \mathbf{v}_1 \cdot \mathbf{0} = 0
$$
Since $\mathbf{v}_1 \cdot \mathbf{v}_j = 0$ for all $j \neq 1$:
$$
c_1 (\mathbf{v}_1 \cdot \mathbf{v}_1) + 0 + \dots + 0 = 0 \implies c_1 \|\mathbf{v}_1\|_2^2 = 0
$$
Because $\mathbf{v}_1 \neq \mathbf{0}$, $\|\mathbf{v}_1\|_2^2 > 0 \implies c_1 = 0$.
Repeating for all $i$ proves $c_1 = c_2 = \dots = c_k = 0$. **Therefore, orthogonal non-zero vectors are strictly linearly independent!** $\blacksquare$

---

## 12.3 Coordinates in an Orthonormal Basis (No Matrix Inversion Needed!)

If $\{\mathbf{q}_1, \mathbf{q}_2, \dots, \mathbf{q}_d\}$ forms an orthonormal basis for $\mathbb{R}^d$, expressing any vector $\mathbf{x}$ as a linear combination requires **zero matrix inversion**:

$$
\mathbf{x} = c_1 \mathbf{q}_1 + c_2 \mathbf{q}_2 + \dots + c_d \mathbf{q}_d
$$

Take the dot product with $\mathbf{q}_i$:
$$
\mathbf{q}_i^T \mathbf{x} = c_i (\mathbf{q}_i^T \mathbf{q}_i) = c_i(1) \implies c_i = \mathbf{q}_i^T \mathbf{x}
$$

$$
\mathbf{x} = \sum_{i=1}^{d} (\mathbf{x}^T \mathbf{q}_i) \mathbf{q}_i
$$

* *Each coordinate is simply the scalar projection of $\mathbf{x}$ onto that basis vector!*

---

## 12.4 Orthogonal Complement ($S^\perp$) & Orthogonal Decomposition

For any subspace $S \subseteq \mathbb{R}^d$, its **Orthogonal Complement** $S^\perp$ is the set of all vectors perpendicular to every vector in $S$:
$$
S^\perp = \left\lbrace \mathbf{x} \in \mathbb{R}^d \mid \mathbf{x} \cdot \mathbf{s} = 0 \quad \forall \mathbf{s} \in S \right\rbrace
$$

* **Unique Orthogonal Decomposition Theorem:** Any vector $\mathbf{x} \in \mathbb{R}^d$ can be uniquely split into a component in $S$ and a component in $S^\perp$:
  $$
  \mathbf{x} = \mathbf{x}_S + \mathbf{x}_{S^\perp} = \text{proj}_S(\mathbf{x}) + (\mathbf{x} - \text{proj}_S(\mathbf{x}))
  $$

---

## 12.5 Orthogonal Matrices ($Q$)

A square matrix $Q \in \mathbb{R}^{n \times n}$ whose columns are orthonormal is an **Orthogonal Matrix**:
$$
Q^T Q = Q Q^T = I \iff Q^{-1} = Q^T
$$

* **Isometry (Length Preservation):** $\|Q\mathbf{x}\|_2 = \|\mathbf{x}\|_2$.
* **Angle Preservation:** $\langle Q\mathbf{u}, Q\mathbf{v} \rangle = \mathbf{u}^T \mathbf{v}$.

---

## 12.6 The Conceptual Bridge: Orthogonality $\to$ Projection $\to$ Least Squares

```
  Orthogonality (u ⟂ v = 0)  ──►  Projection (Closest point p in S)  ──►  Least Squares (min ||y - Xw||^2)
```

In [Part 13](./13_vector_projections.md) and [Part 14](./14_linear_regression_matrix_math.md), we use this exact orthogonal decomposition to project target data onto feature spaces to solve linear regression!

---

> 📖 **Navigation:** [← Previous: Part 11: Singular Value Decomposition (SVD)](./11_singular_value_decomposition.md) | [🏠 Index](./README.md) | [Next: Part 13: Vector Projections →](./13_vector_projections.md)
