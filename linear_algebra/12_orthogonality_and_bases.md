> 📖 **Navigation:** [← Previous: Part 11: Singular Value Decomposition (SVD)](./11_singular_value_decomposition.md) | [🏠 Index](./README.md) | [Next: Part 13: Vector Projections →](./13_vector_projections.md)

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

> 📖 **Navigation:** [← Previous: Part 11: Singular Value Decomposition (SVD)](./11_singular_value_decomposition.md) | [🏠 Index](./README.md) | [Next: Part 13: Vector Projections →](./13_vector_projections.md)
