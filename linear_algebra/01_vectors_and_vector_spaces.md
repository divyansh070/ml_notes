> 📖 **Navigation:** [🏠 Index](./README.md) | [Next: Part 02: Matrices & Matrix Operations →](./02_matrices_and_operations.md)

---

# PART 1 — VECTORS & VECTOR BASICS

In Machine Learning and Data Science, data begins as vectors. Tabular sample attributes, word and token embeddings, image pixel arrays, and model weights are all vectors living in multi-dimensional space.

---

## 1.1 What is a Vector? Notation, Dimensions, and Components

### 1. Mathematical Definitions
* **Scalar:** A single real number ($c \in \mathbb{R}$) representing magnitude without direction (e.g., learning rate $\alpha = 0.01$, temperature $T = 0.7$, loss $\mathcal{L} = 0.35$).
* **Vector:** An ordered sequence of numbers representing both magnitude and direction in space.
* **Dimensions:** The number of components (features) in the vector, denoted $d$. A vector with $d$ real entries belongs to Euclidean space $\mathbb{R}^d$.
* **Components / Elements:** The individual numerical values $x_i$ inside the vector:

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_d \end{bmatrix} \in \mathbb{R}^d
$$

By standard mathematical and machine learning convention, **vectors are column vectors** ($d \times 1$) unless explicitly transposed ($\mathbf{x}^T = [x_1, x_2, \dots, x_d]$ is a $1 \times d$ row vector).

### 2. The Three Complementary Views of a Vector

```
     1. GEOMETRIC ARROW               2. POINT IN SPACE              3. FEATURE VECTOR (ML)
   (Direction + Magnitude)           (Coordinates in R^d)              (Sample Attributes)
             y                                y
             │      / x = [3, 2]^T            │        ● (3, 2)           House Sample:
           2 ┼─────►                        2 ┼───────┘                   [ Bedrooms: 3 ]
             │    /                           │                           [ Bathrooms: 2]
             │   /                            │                           [ Area (k): 1.8]
             └──┴────────► x                  └───┴────────► x            Shape: (d x 1)
                3                                 3
```

1. **Feature Vector (ML):** A numerical representation of an observation where each coordinate corresponds to a measurable feature.
2. **Geometric Arrow (Physics):** A directed line segment starting at the origin $\mathbf{0} = [0, 0, \dots, 0]^T$ and terminating at the coordinates $(x_1, \dots, x_d)$.
3. **Point in Space (Geometry):** A static coordinate address in $\mathbb{R}^d$.

---

## 1.2 Vector Addition and Scalar Multiplication

Vector spaces are defined by two primary operations:

### 1. Vector Addition
Add corresponding components element-by-element. Both vectors must share the same dimension $d$:

$$
\mathbf{a} + \mathbf{b} =
\begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_d \end{bmatrix} +
\begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_d \end{bmatrix} =
\begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_d + b_d \end{bmatrix}
$$

* **Geometric Tip-to-Tail Rule:** Place the tail of $\mathbf{b}$ at the tip of $\mathbf{a}$. The resulting vector from the origin to the tip of $\mathbf{b}$ forms the diagonal of the parallelogram defined by $\mathbf{a}$ and $\mathbf{b}$.
* **Vector Subtraction:** $\mathbf{a} - \mathbf{b} = \mathbf{a} + (-1)\mathbf{b}$. Geometrically, $\mathbf{a} - \mathbf{b}$ is the displacement vector directed **from the tip of $\mathbf{b}$ to the tip of $\mathbf{a}$**.

### 2. Scalar Multiplication
Multiply every component by a scalar $c \in \mathbb{R}$:

$$
c\mathbf{x} = \begin{bmatrix} c x_1 \\ c x_2 \\ \vdots \\ c x_d \end{bmatrix}
$$

* **Geometric Meaning:** 
  * $c > 1$: Stretches the vector along its existing line of action.
  * $0 < c < 1$: Compresses (shrinks) the vector.
  * $c < 0$: Reverses the vector's direction by $180^\circ$ and scales its length by $|c|$.

---

## 1.3 The Dot Product (Inner Product)

The **dot product** takes two equal-length vectors and produces a single scalar.

### 1. Computational Definition (Algebraic Form)
Sum of the products of corresponding elements:

$$
\mathbf{a} \cdot \mathbf{b} = \mathbf{a}^T \mathbf{b} = \sum_{i=1}^{d} a_i b_i = a_1 b_1 + a_2 b_2 + \dots + a_d b_d
$$

### 2. Geometric Definition & Derivation
For vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^d$ separated by angle $\theta \in [0, \pi]$:

$$
\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|_2 \|\mathbf{b}\|_2 \cos\theta
$$

> [!NOTE]
> **Where does this formula come from?**
> Consider the triangle formed by $\mathbf{a}$, $\mathbf{b}$, and $\mathbf{c} = \mathbf{a} - \mathbf{b}$. By the geometric Law of Cosines:
>
> $$
> \|\mathbf{a} - \mathbf{b}\|_2^2 = \|\mathbf{a}\|_2^2 + \|\mathbf{b}\|_2^2 - 2 \|\mathbf{a}\|_2 \|\mathbf{b}\|_2 \cos\theta
> $$
>
> Expanding the left side algebraically using dot products:
>
> $$
> \|\mathbf{a} - \mathbf{b}\|_2^2 = (\mathbf{a} - \mathbf{b})^T (\mathbf{a} - \mathbf{b}) = \mathbf{a}^T \mathbf{a} - 2\mathbf{a}^T \mathbf{b} + \mathbf{b}^T \mathbf{b} = \|\mathbf{a}\|_2^2 - 2(\mathbf{a} \cdot \mathbf{b}) + \|\mathbf{b}\|_2^2
> $$
>
> Equating both expansions directly yields:
>
> $$
> -2(\mathbf{a} \cdot \mathbf{b}) = -2 \|\mathbf{a}\|_2 \|\mathbf{b}\|_2 \cos\theta \implies \mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|_2 \|\mathbf{b}\|_2 \cos\theta
> $$
>

### 3. What the Dot Product Tells Us
The sign of $\mathbf{a} \cdot \mathbf{b}$ directly reveals the directional alignment of two vectors:

```
     ACUTE ANGLE (θ < 90°)          ORTHOGONAL (θ = 90°)         OBTUSE ANGLE (θ > 90°)
          a · b > 0                      a · b = 0                     a · b < 0
       (Pointing Together)            (Perpendicular)              (Pointing Away)
              y                              y                             y
              │      / b                     │      │ b                    │   b \
              │     /                        │      │                      │      \
              │    /                         │      │                      │       \
              │   ●────► a                   │   ●──┴──► a                 │   ●────┴─► a
              └───┴────────► x               └───┴────────► x              └───┴────────► x
```

* **$\mathbf{a} \cdot \mathbf{b} > 0$ ($\theta < 90^\circ$):** Vectors point in generally the same direction.
* **$\mathbf{a} \cdot \mathbf{b} = 0$ ($\theta = 90^\circ$):** Vectors are **orthogonal** (perpendicular).
* **$\mathbf{a} \cdot \mathbf{b} < 0$ ($\theta > 90^\circ$):** Vectors point in opposing directions.

---

## 1.4 Vector Norms (Magnitude & Length)

A **norm** $\|\mathbf{x}\|$ is a function that assigns a strictly non-negative length to a vector.

### 1. $L_2$ Norm (Euclidean Norm)
The straight-line geometric distance from origin to point:

$$
\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{d} x_i^2} = \sqrt{\mathbf{x}^T \mathbf{x}}
$$

### 2. $L_1$ Norm (Manhattan / Taxicab Norm)
The grid-based distance traversed along coordinate axes:

$$
\|\mathbf{x}\|_1 = \sum_{i=1}^{d} |x_i|
$$

### 3. $L_\infty$ Norm (Chebyshev / Maximum Norm)
The single largest absolute component:

$$
\|\mathbf{x}\|_\infty = \max_{1 \le i \le d} |x_i|
$$

### 4. Distance Between Vectors
The Euclidean distance between two points $\mathbf{a}$ and $\mathbf{b}$ is the $L_2$ norm of their difference vector:

$$
d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2 = \sqrt{\sum_{i=1}^{d} (a_i - b_i)^2}
$$

---

## 1.5 Unit Vectors & Normalization

A **unit vector** $\mathbf{u}$ is a vector with length exactly equal to 1: $\|\mathbf{u}\|_2 = 1$.

* **Vector Normalization:** Any non-zero vector $\mathbf{v} \neq \mathbf{0}$ can be normalized into a unit vector pointing in the identical direction by dividing by its Euclidean norm:

$$
\mathbf{u} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}
$$

* **Verification:** $\left\|\frac{\mathbf{v}}{\|\mathbf{v}\|_2}\right\|_2 = \frac{1}{\|\mathbf{v}\|_2} \|\mathbf{v}\|_2 = 1$.

---

## 1.6 Angle Between Vectors & Cosine Similarity

Rearranging the geometric dot product gives the formula for the angle $\theta$ between any two vectors:

$$
\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2} = \left(\frac{\mathbf{a}}{\|\mathbf{a}\|_2}\right) \cdot \left(\frac{\mathbf{b}}{\|\mathbf{b}\|_2}\right)
$$

### Cosine Similarity:
In machine learning, **Cosine Similarity** measures pure angular alignment regardless of vector magnitudes:

$$
\text{Cosine Similarity}(\mathbf{a}, \mathbf{b}) = \cos\theta \in [-1.0, +1.0]
$$

* $+1.0 \implies$ Perfectly aligned ($\theta = 0^\circ$).
* $0.0 \implies$ Orthogonal / completely unrelated ($\theta = 90^\circ$).
* $-1.0 \implies$ Diametrically opposed ($\theta = 180^\circ$).

---

## 1.7 Complete Worked Numerical Example

Let:

$$
\mathbf{a} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 4 \\ -3 \end{bmatrix}, \quad \mathbf{c} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}
$$

### Step 1: Vector Addition and Subtraction

$$
\mathbf{a} + \mathbf{c} = \begin{bmatrix} 3 + 1 \\ 4 + 2 \end{bmatrix} = \begin{bmatrix} 4 \\ 6 \end{bmatrix}, \quad \mathbf{a} - \mathbf{c} = \begin{bmatrix} 3 - 1 \\ 4 - 2 \end{bmatrix} = \begin{bmatrix} 2 \\ 2 \end{bmatrix}
$$

### Step 2: Norm Calculations for Vector a
* $L_2$ Norm: $\|\mathbf{a}\|_2 = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$
* $L_1$ Norm: $\|\mathbf{a}\|_1 = |3| + |4| = 7$
* $L_\infty$ Norm: $\|\mathbf{a}\|_\infty = \max(|3|, |4|) = 4$

### Step 3: Normalizing Vector a into a Unit Vector

$$
\mathbf{u}_{\mathbf{a}} = \frac{\mathbf{a}}{\|\mathbf{a}\|_2} = \frac{1}{5} \begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 0.6 \\ 0.8 \end{bmatrix}
$$

* Check unit length: $\|\mathbf{u}_{\mathbf{a}}\|_2 = \sqrt{0.6^2 + 0.8^2} = \sqrt{0.36 + 0.64} = \sqrt{1.0} = 1$ ✓

### Step 4: Dot Product and Orthogonality Check for a and b

$$
\mathbf{a} \cdot \mathbf{b} = (3)(4) + (4)(-3) = 12 - 12 = 0
$$

* Because $\mathbf{a} \cdot \mathbf{b} = 0$, vectors $\mathbf{a}$ and $\mathbf{b}$ are **strictly orthogonal** ($\theta = 90^\circ$).

### Step 5: Angle and Distance Between a and c
* Dot product: $\mathbf{a} \cdot \mathbf{c} = (3)(1) + (4)(2) = 3 + 8 = 11$
* Norm of $\mathbf{c}$: $\|\mathbf{c}\|_2 = \sqrt{1^2 + 2^2} = \sqrt{5} \approx 2.236$
* Cosine similarity:

$$
\cos\theta = \frac{11}{(5)(\sqrt{5})} = \frac{11}{5\sqrt{5}} = \frac{11}{11.18} \approx 0.9838 \implies \theta \approx 10.3^\circ
$$

* Euclidean distance:

$$
d(\mathbf{a}, \mathbf{c}) = \|\mathbf{a} - \mathbf{c}\|_2 = \sqrt{(3 - 1)^2 + (4 - 2)^2} = \sqrt{2^2 + 2^2} = \sqrt{8} \approx 2.828
$$

---

## 1.8 Why this matters in ML

1. **Feature Vectors & Embeddings:** Every document, image, or user profile in machine learning is converted into a vector in $\mathbb{R}^d$.
2. **Attention in Transformers:** The core self-attention query-key score is a scaled dot product:

$$
\text{Attention Logit} = \frac{\mathbf{q}^T \mathbf{k}}{\sqrt{d_k}}
$$

   High dot products indicate strong semantic affinity between tokens.
3. **Cosine Similarity in Vector Databases:** When searching for nearest neighbors in vector databases (Pinecone, Milvus, Chroma), cosine similarity compares the direction of text embeddings without being distorted by document length.
4. **Regularization ($L_1$ vs $L_2$):** Ridge regression penalizes the squared $L_2$ norm of weights ($\lambda \|\mathbf{w}\|_2^2$), preventing weights from growing too large. Lasso penalizes the $L_1$ norm ($\lambda \|\mathbf{w}\|_1$), forcing irrelevant weights to exact 0.0.

---

## 1.9 Common mistakes

* **Confusing Dot Product with Scalar Multiplication:** Scalar multiplication ($c\mathbf{v}$) multiplies a vector by a scalar to produce a *vector*. The dot product ($\mathbf{u} \cdot \mathbf{v}$) multiplies two vectors together to produce a *scalar*.
* **Omitting Normalization in Cosine Similarity:** Writing $\mathbf{a}^T \mathbf{b}$ instead of $\frac{\mathbf{a}^T \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$. The raw dot product combines both angle and magnitude; cosine similarity requires dividing by lengths.
* **Assuming Orthogonal Vectors Must Be Unit Length:** Two non-zero vectors are orthogonal as long as $\mathbf{u} \cdot \mathbf{v} = 0$, regardless of whether their individual lengths are 1, 10, or 1000. (Vectors that are both orthogonal and unit length are called *orthonormal*).

---

> 📖 **Navigation:** [🏠 Index](./README.md) | [Next: Part 02: Matrices & Matrix Operations →](./02_matrices_and_operations.md)
