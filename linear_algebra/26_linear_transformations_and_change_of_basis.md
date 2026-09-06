> 📖 **Navigation:** [← Previous: Part 25: Moore-Penrose Pseudoinverse (A^+)](./25_moore_penrose_pseudoinverse.md) | [🏠 Index](./README.md) | [Next: Part 27: 40 Essential Technical Interview Questions & Answers →](./27_interview_questions_and_answers.md)

---

# PART 26 — LINEAR TRANSFORMATIONS

---

## 26.1 Definition

A **Transformation** $T: \mathbb{R}^n \to \mathbb{R}^m$ is a rule that takes an input vector $\mathbf{x} \in \mathbb{R}^n$ and produces an output vector $T(\mathbf{x}) \in \mathbb{R}^m$.

A transformation is **Linear** if it can be represented entirely as a matrix multiplication:

$$
T(\mathbf{x}) = A\mathbf{x}
$$

```
                   LINEAR TRANSFORMATION AS MAPPING
        Input Space R^2                             Output Space R^2
               y                                           y
               │     x = [1, 2]^T                          │           A @ x = [2, 6]^T
             2 ┼────●                                    6 ┼──────────●
               │    │                                      │          │
               └───┴────► x                                └──────────┴────► x
                   1                                                  2
```

---

## 26.2 The Two Defining Properties

A transformation $T$ is linear if and only if it preserves vector addition and scalar multiplication:
1. **Additivity:** $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
2. **Homogeneity (Scalar Scaling):** $T(c\mathbf{u}) = cT(\mathbf{u})$

### Numerical Verification Example
Let $\mathbf{u} = [1, 2]^T$, $\mathbf{v} = [3, 1]^T$, $c = 4$, and:

$$
A =
\begin{bmatrix}
2 & 0 \\
0 & 3
\end{bmatrix}
$$

* $T(\mathbf{u}) = A\mathbf{u} = [2, 6]^T$, $T(\mathbf{v}) = A\mathbf{v} = [6, 3]^T$.
* $\mathbf{u} + \mathbf{v} = [4, 3]^T \implies T(\mathbf{u} + \mathbf{v}) = A[4, 3]^T = [8, 9]^T$.
* Check Additivity: $T(\mathbf{u}) + T(\mathbf{v}) = [2+6, 6+3]^T = [8, 9]^T \quad \checkmark$.
* Check Homogeneity: $T(4\mathbf{u}) = A[4, 8]^T = [8, 24]^T = 4[2, 6]^T = 4T(\mathbf{u}) \quad \checkmark$.

---

## 26.3 Matrices as Geometric Transformations

Every 2D linear transformation is completely determined by where it sends the standard unit basis vectors $\mathbf{i} = [1, 0]^T$ and $\mathbf{j} = [0, 1]^T$:

1. **Non-Uniform Scaling (Stretch/Shrink):**

$$
A_{\text{scale}} =
\begin{bmatrix}
s_x & 0 \\
0 & s_y
\end{bmatrix}
\implies
\begin{bmatrix}
2 & 0 \\
0 & 0.5
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix} =
\begin{bmatrix}
2x \\
0.5y
\end{bmatrix}
$$

2. **Rotation by Angle** $\theta$ **(Counter-Clockwise):**

$$
R_\theta =
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
\implies R_{90^\circ} =
\begin{bmatrix}
0 & -1 \\
1 & 0
\end{bmatrix}
$$

   * Sends $\mathbf{i} = [1, 0]^T \to [0, 1]^T$ and $\mathbf{j} = [0, 1]^T \to [-1, 0]^T$.

3. **Reflection over the x-axis:**

$$
A_{\text{reflect}} =
\begin{bmatrix}
1 & 0 \\
0 & -1
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix} =
\begin{bmatrix}
x \\
-y
\end{bmatrix}
$$

4. **Horizontal Shear (Sliding Layers):**

$$
A_{\text{shear}} =
\begin{bmatrix}
1 & k \\
0 & 1
\end{bmatrix}
\implies
\begin{bmatrix}
1 & 1.5 \\
0 & 1
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix} =
\begin{bmatrix}
x + 1.5y \\
y
\end{bmatrix}
$$

---

## 26.4 Connection to Eigenvectors

* **Geometric Insight:** Under most linear transformations, vectors change both length and direction.
* **The Eigenvector Exception:** An **eigenvector** is a direction where the linear transformation $A$ acts as a pure scalar stretch without rotating the vector at all:

$$
T(\mathbf{v}) = A\mathbf{v} = \lambda \mathbf{v}
$$

---

## 26.5 Connection to SVD

Singular Value Decomposition factorizes ANY linear transformation $A = U \Sigma V^T$ into three pure geometric actions:

$$
\mathbf{x} \quad \xrightarrow{\quad V^T \quad} \quad \mathbf{x}' \quad \xrightarrow{\quad \Sigma \quad} \quad \mathbf{x}'' \quad \xrightarrow{\quad U \quad} \quad A\mathbf{x}
$$

1. **Rotate / Reflect** ($V^T$): Aligns the input coordinate system with the principal axes.
2. **Scale** ($\Sigma$): Stretches or compresses along each principal axis by singular value factors $\sigma_i$.
3. **Second Rotation** ($U$): Rotates the resulting ellipse into the target coordinate space.

---

## 26.6 ML Connection

* **Neural Network Dense Layers:** A fully connected neural network layer $\mathbf{z} = W\mathbf{x} + \mathbf{b}$ is an **affine transformation**: the weight matrix $W$ performs a linear transformation on the input features, and the bias vector $\mathbf{b}$ translates the origin. Non-linear activation functions (ReLU, GELU) then warp the space to learn non-linear decision boundaries.
* **Embeddings & Latent Representations:** Learned weight matrices project raw inputs into lower-dimensional latent spaces where geometric relationships (like cosine angle and Euclidean distance) encode semantic meaning.

> [!TIP]
> **Common Interview Question:** *"How does Singular Value Decomposition decompose any arbitrary linear transformation geometrically?"*
> **Answer:** SVD factorizes $A = U \Sigma V^T$ into three successive geometric operations: (1) an initial rotation/reflection in the domain by orthogonal matrix $V^T$, (2) an axis-aligned stretching/compression along coordinate axes by diagonal matrix $\Sigma$, and (3) a second rotation/reflection in the codomain by orthogonal matrix $U$.

> [!WARNING]
> **Common Mistake:** Confusing a **linear transformation** ($T(\mathbf{x}) = W\mathbf{x}$) with an **affine transformation** ($T(\mathbf{x}) = W\mathbf{x} + \mathbf{b}$). A linear transformation must map the origin to the origin ($T(\mathbf{0}) = \mathbf{0}$). A neural network dense layer is an affine transformation because the bias vector $\mathbf{b}$ shifts the origin.

---

> 📖 **Navigation:** [← Previous: Part 25: Moore-Penrose Pseudoinverse (A^+)](./25_moore_penrose_pseudoinverse.md) | [🏠 Index](./README.md) | [Next: Part 27: 40 Essential Technical Interview Questions & Answers →](./27_interview_questions_and_answers.md)
