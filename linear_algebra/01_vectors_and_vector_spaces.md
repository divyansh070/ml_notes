> 📖 **Navigation:** [← Return to Index](./README.md) | [🏠 Index](./README.md) | [Next: Part 02: Matrices & Matrix Operations →](./02_matrices_and_operations.md)

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

> 📖 **Navigation:** [← Return to Index](./README.md) | [🏠 Index](./README.md) | [Next: Part 02: Matrices & Matrix Operations →](./02_matrices_and_operations.md)
