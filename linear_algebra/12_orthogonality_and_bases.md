> 📖 **Navigation:** [← Previous: Part 11: Linear Transformations](./11_linear_transformations.md) | [🏠 Index](./README.md) | [Next: Part 13: Vector Projections & Projection Matrices →](./13_vector_projections.md)

---

# PART 12 — ORTHOGONALITY & ORTHONORMAL BASES

**Orthogonality** (perpendicularity) is the bedrock of stable numerical computing and machine learning. When basis vectors are orthogonal, multidimensional problems decouple into independent 1D problems, matrix inversion reduces to a simple transpose, and rounding errors remain strictly bounded.

---

## 12.1 Orthogonal vs. Orthonormal Vectors

1. **Orthogonal Vectors:** Two vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$ are **orthogonal** ($\mathbf{u} \perp \mathbf{v}$) if and only if their dot product is zero:

$$
\mathbf{u} \perp \mathbf{v} \iff \mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T \mathbf{v} = 0
$$

2. **Orthonormal Vectors:** Vectors that are both mutually perpendicular AND normalized to unit length ($\|\mathbf{q}_i\|_2 = 1$):

$$
\mathbf{q}_i^T \mathbf{q}_j =
\begin{cases}
1 & \text{if } i = j \\
0 & \text{if } i \neq j
\end{cases}
$$

---

## 12.2 Proof: Why Orthogonal Non-Zero Vectors are ALWAYS Linearly Independent

Let $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_k\}$ be mutually orthogonal non-zero vectors ($\mathbf{v}_i \cdot \mathbf{v}_j = 0$ for $i \neq j$, and $\mathbf{v}_i \neq \mathbf{0}$).

Set their linear combination to zero:

$$
c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_k \mathbf{v}_k = \mathbf{0}
$$

Take the dot product of both sides with $\mathbf{v}_1$:

$$
\mathbf{v}_1 \cdot (c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_k \mathbf{v}_k) = \mathbf{v}_1 \cdot \mathbf{0} = 0
$$

By orthogonality, every term $\mathbf{v}_1 \cdot \mathbf{v}_j = 0$ for $j \neq 1$:

$$
c_1 (\mathbf{v}_1 \cdot \mathbf{v}_1) + 0 + \dots + 0 = 0 \implies c_1 \|\mathbf{v}_1\|_2^2 = 0
$$

Because $\mathbf{v}_1 \neq \mathbf{0}$, its squared length $\|\mathbf{v}_1\|_2^2 > 0$, forcing $c_1 = 0$.

Repeating for every vector $\mathbf{v}_i$ proves:

$$
c_1 = c_2 = \dots = c_k = 0
$$

**Therefore, mutually orthogonal non-zero vectors are strictly linearly independent.** ■

---

## 12.3 Expanding Vectors in an Orthonormal Basis (No Inversion Needed!)

If $\{\mathbf{q}_1, \mathbf{q}_2, \dots, \mathbf{q}_d\}$ forms an orthonormal basis for $\mathbb{R}^d$, expressing any arbitrary vector $\mathbf{x}$ as a linear combination requires **zero matrix inversion**:

$$
\mathbf{x} = c_1 \mathbf{q}_1 + c_2 \mathbf{q}_2 + \dots + c_d \mathbf{q}_d
$$

Take the dot product of both sides with basis vector $\mathbf{q}_i$:

$$
\mathbf{q}_i^T \mathbf{x} = c_i (\mathbf{q}_i^T \mathbf{q}_i) = c_i(1) \implies c_i = \mathbf{q}_i^T \mathbf{x}
$$

$$
\mathbf{x} = \sum_{i=1}^{d} (\mathbf{x} \cdot \mathbf{q}_i) \mathbf{q}_i
$$

* **The Takeaway:** In an orthonormal basis, each coordinate is simply the **scalar dot product** of $\mathbf{x}$ with that basis vector!

---

## 12.4 Orthogonal Complement ($S^\perp$) & Vector Decomposition

For any vector subspace $S \subseteq \mathbb{R}^d$, its **Orthogonal Complement** $S^\perp$ is the set of all vectors perpendicular to every vector in $S$:

$$
S^\perp = \left\lbrace \mathbf{x} \in \mathbb{R}^d \;\middle|\; \mathbf{x} \cdot \mathbf{s} = 0 \quad \forall \mathbf{s} \in S \right\rbrace
$$

### The Unique Orthogonal Decomposition Theorem:
Every vector $\mathbf{x} \in \mathbb{R}^d$ can be uniquely decomposed into two perpendicular components:

$$
\mathbf{x} = \mathbf{x}_S + \mathbf{x}_{S^\perp} = \text{proj}_S(\mathbf{x}) + (\mathbf{x} - \text{proj}_S(\mathbf{x}))
$$

where $\mathbf{x}_S \in S$ and $\mathbf{x}_{S^\perp} \in S^\perp$.

---

## 12.5 Orthogonal Matrices ($Q$)

A square matrix $Q \in \mathbb{R}^{n \times n}$ whose columns are orthonormal vectors is called an **Orthogonal Matrix**:

$$
Q = \begin{bmatrix} \mid & \mid & & \mid \\ \mathbf{q}_1 & \mathbf{q}_2 & \dots & \mathbf{q}_n \\ \mid & \mid & & \mid \end{bmatrix}
$$

$$
Q^T Q = I_n \iff Q^{-1} = Q^T
$$

### Master Properties of Orthogonal Matrices:
1. **Inversion is Trivial:** $Q^{-1} = Q^T$ (Inversion requires only a transpose!).
2. **Preserves Vector Lengths (Isometry):**

$$
\|Q\mathbf{x}\|_2^2 = (Q\mathbf{x})^T (Q\mathbf{x}) = \mathbf{x}^T (Q^T Q) \mathbf{x} = \mathbf{x}^T I \mathbf{x} = \|\mathbf{x}\|_2^2 \implies \|Q\mathbf{x}\|_2 = \|\mathbf{x}\|_2
$$

3. **Preserves Angles and Dot Products:**

$$
(Q\mathbf{u}) \cdot (Q\mathbf{v}) = \mathbf{u}^T (Q^T Q) \mathbf{v} = \mathbf{u}^T \mathbf{v}
$$

4. **Determinant:** $\det(Q) = \pm 1$ (Pure rotation if $+1$, rotation with reflection if $-1$).

---

## 12.6 Complete Worked Numerical Example

Let:

$$
\mathbf{q}_1 = \begin{bmatrix} 3/5 \\ 4/5 \end{bmatrix}, \quad \mathbf{q}_2 = \begin{bmatrix} -4/5 \\ 3/5 \end{bmatrix}, \quad \mathbf{x} = \begin{bmatrix} 10 \\ 5 \end{bmatrix}
$$

### 1. Check Orthonormality:
* Lengths:
  * $\|\mathbf{q}_1\|_2 = \sqrt{(3/5)^2 + (4/5)^2} = \sqrt{9/25 + 16/25} = \sqrt{25/25} = 1$ ✓
  * $\|\mathbf{q}_2\|_2 = \sqrt{(-4/5)^2 + (3/5)^2} = \sqrt{16/25 + 9/25} = 1$ ✓
* Dot product:
  * $\mathbf{q}_1 \cdot \mathbf{q}_2 = (3/5)(-4/5) + (4/5)(3/5) = -12/25 + 12/25 = 0$ ✓

### 2. Express x in Basis $\{\mathbf{q}_1, \mathbf{q}_2\}$ via Direct Dot Products:
* $c_1 = \mathbf{x} \cdot \mathbf{q}_1 = (10)(3/5) + (5)(4/5) = 6 + 4 = 10$
* $c_2 = \mathbf{x} \cdot \mathbf{q}_2 = (10)(-4/5) + (5)(3/5) = -8 + 3 = -5$

#### Verification:

$$
c_1 \mathbf{q}_1 + c_2 \mathbf{q}_2 = 10 \begin{bmatrix} 3/5 \\ 4/5 \end{bmatrix} - 5 \begin{bmatrix} -4/5 \\ 3/5 \end{bmatrix} =
\begin{bmatrix} 6 \\ 8 \end{bmatrix} + \begin{bmatrix} 4 \\ -3 \end{bmatrix} = \begin{bmatrix} 10 \\ 5 \end{bmatrix} = \mathbf{x}
$$

✓ **Verified**.

### 3. Verify Length Preservation:
* Input length: $\|\mathbf{x}\|_2 = \sqrt{10^2 + 5^2} = \sqrt{100 + 25} = \sqrt{125}$
* Transformed coordinates: $\sqrt{c_1^2 + c_2^2} = \sqrt{10^2 + (-5)^2} = \sqrt{100 + 25} = \sqrt{125}$ ✓

---

## 12.7 Why this matters in ML

1. **Numerical Stability:** In scientific computing, non-orthogonal bases cause matrix condition numbers to explode. Algorithms like **QR Decomposition** and **SVD** orthogonalize matrices to eliminate roundoff error accumulation.
2. **Zero Covariance (Decorrelated Features):** If feature vectors are orthogonal, their sample covariance is zero:

$$
\text{Cov}(X_i, X_j) \propto \mathbf{x}_i \cdot \mathbf{x}_j = 0
$$

   PCA achieves dimensionality reduction precisely by rotating data onto an orthonormal basis of eigenvectors where all cross-feature covariances are zero.

---

## 12.8 Common mistakes

* **Confusing "Orthogonal Vectors" with an "Orthogonal Matrix":** An *orthogonal matrix* requires that its columns be both mutually orthogonal **AND unit length** (i.e. orthonormal). A matrix whose columns are perpendicular but have length 2 is NOT an orthogonal matrix.
* **Calling Non-Square Matrices "Orthogonal":** If a tall matrix $Q \in \mathbb{R}^{m \times n}$ ($m > n$) has orthonormal columns, $Q^T Q = I_n$, but $Q Q^T \neq I_m$. It is called a *matrix with orthonormal columns*, not an orthogonal matrix (orthogonal matrices must be square).

---

> 📖 **Navigation:** [← Previous: Part 11: Linear Transformations](./11_linear_transformations.md) | [🏠 Index](./README.md) | [Next: Part 13: Vector Projections & Projection Matrices →](./13_vector_projections.md)
