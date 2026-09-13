> 📖 **Navigation:** [← Previous: Part 10: Four Fundamental Subspaces](./10_four_fundamental_subspaces.md) | [🏠 Index](./README.md) | [Next: Part 12: Orthogonality & Orthonormal Bases →](./12_orthogonality_and_bases.md)

---

# PART 11 — LINEAR TRANSFORMATIONS

Matrices are more than static grids of numbers: they are geometric engines that rotate, scale, shear, reflect, and project space. In Machine Learning, every dense layer in a neural network is an affine transformation mapping representations from one feature space to another.

---

## 11.1 Definition of a Linear Transformation

A mapping $T: \mathbb{R}^n \to \mathbb{R}^m$ is a **Linear Transformation** if and only if it preserves vector addition and scalar multiplication:

1. **Additivity:** $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$ for all $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$.
2. **Homogeneity (Scalar Scaling):** $T(c\mathbf{u}) = c T(\mathbf{u})$ for all $c \in \mathbb{R}, \mathbf{u} \in \mathbb{R}^n$.

Combining both gives the single linearity condition:

$$
T(c\mathbf{u} + d\mathbf{v}) = c T(\mathbf{u}) + d T(\mathbf{v})
$$

### The Origin Invariant:
A strictly linear transformation must **always map the origin to the origin**:

$$
T(\mathbf{0}) = T(0 \cdot \mathbf{v}) = 0 \cdot T(\mathbf{v}) = \mathbf{0}
$$

---

## 11.2 The Fundamental Theorem of Linear Transformations

Every linear transformation $T: \mathbb{R}^n \to \mathbb{R}^m$ is uniquely represented by **matrix multiplication**:

$$
T(\mathbf{x}) = A\mathbf{x}
$$

The columns of matrix $A$ are simply the transformed outputs of the standard unit basis vectors $\mathbf{e}_1, \dots, \mathbf{e}_n$:

$$
A = \begin{bmatrix}
\mid & \mid & & \mid \\
T(\mathbf{e}_1) & T(\mathbf{e}_2) & \dots & T(\mathbf{e}_n) \\
\mid & \mid & & \mid
\end{bmatrix} \in \mathbb{R}^{m \times n}
$$

---

## 11.3 Linear vs. Affine Maps in Neural Networks

```
       LINEAR: T(x) = W x (Origin Fixed)            AFFINE: T(x) = W x + b (Origin Shifted)
                y                                            y
                │     ● T(v)                                 │          ● W v + b
                │    ╱                                       │         ╱
                │   ╱                                        │   b ───● (Shifted Origin)
                └──●────► x                                  └───┴────► x
                 Origin Fixed                                  Origin
```

* **Linear Map:** $T(\mathbf{x}) = W\mathbf{x}$. The origin remains pinned at $(0, 0, \dots, 0)$.
* **Affine Map:** $T(\mathbf{x}) = W\mathbf{x} + \mathbf{b}$. A linear transformation followed by a vector translation $\mathbf{b}$.
* **The Dense Layer:** $\mathbf{z} = W\mathbf{x} + \mathbf{b}$ in deep learning is an **affine transformation**. The bias vector $\mathbf{b}$ allows the network to shift decision boundaries away from the origin.

---

## 11.4 The 5 Elementary 2D Geometric Transformations

Every $2 \times 2$ matrix transforms 2D space by repositioning the basis vectors $\mathbf{i} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $\mathbf{j} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$:

### 1. Scaling (Dilation / Contraction):
Stretches or compresses axes by factors $s_x, s_y$:

$$
A_{\text{scale}} = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix} \implies \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 2x \\ 3y \end{bmatrix}
$$

### 2. Rotation (Counter-Clockwise by Angle $\theta$):
Rotates all vectors by angle $\theta$ around the origin without changing their lengths:

$$
R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \implies R_{90^\circ} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}
$$

### 3. Reflection (Across Coordinate Axes):
Flips vectors across an axis:

$$
A_{\text{reflect (x-axis)}} = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}, \quad A_{\text{reflect (y-axis)}} = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}
$$

### 4. Shear (Sliding Parallel to an Axis):
Slides points parallel to an axis proportionally to their perpendicular coordinate:

$$
A_{\text{shear (horizontal)}} = \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix} \implies \begin{bmatrix} 1 & 1.5 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} x + 1.5y \\ y \end{bmatrix}
$$

### 5. Projection (Onto an Axis):
Flattens space by dropping perpendicular shadows onto a coordinate line:

$$
P_x = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix} \implies \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} x \\ 0 \end{bmatrix}
$$

---

## 11.5 Connection to Determinant & Invertibility

* **Area/Volume Multiplier:** The absolute determinant $|\det(A)|$ is the volume scaling factor of transformation $A$.
* **Reversibility:** 
  * If $\det(A) \neq 0$, the transformation preserves dimension. Every output point has a unique ancestor point: $T^{-1}(\mathbf{y}) = A^{-1}\mathbf{y}$.
  * If $\det(A) = 0$ (e.g. projection $P_x$), space is flattened into a lower dimension. Multiple points collapse onto the same output point ($N(A) \neq \{\mathbf{0}\}$), so the transformation **cannot be inverted**.

---

## 11.6 Change of Basis & Matrix Similarity

A linear transformation $T$ exists independently of any coordinate system. If we change from the standard basis to a new basis $\mathcal{B} = \{\mathbf{v}_1, \dots, \mathbf{v}_n\}$ with change-of-basis matrix $P = [\mathbf{v}_1 \dots \mathbf{v}_n]$, the matrix representation of $T$ transforms as:

$$
B = P^{-1} A P
$$

* Matrices $A$ and $B$ are **similar matrices**: they represent the exact same underlying linear transformation viewed from two different coordinate perspectives.
* Similar matrices share the **exact same eigenvalues, trace, and determinant**.
* **Diagonalization Goal:** Finding a basis $P$ such that $B = P^{-1} A P = D$ is diagonal (covered in [Part 16](./16_diagonalization.md)).

---

## 11.7 Complete Worked Numerical Example

Construct the transformation matrix $A$ that:
1. Rotates 2D vectors counter-clockwise by $90^\circ$.
2. Then scales the x-axis by $2$ and the y-axis by $3$.

### Step 1: Write Individual Transformation Matrices

$$
R_{90^\circ} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}, \quad S = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}
$$

### Step 2: Compose Transformations ($S$ after $R$)

$$
A = S R_{90^\circ} = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix} \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} =
\begin{bmatrix}
(2)(0) + (0)(1) & (2)(-1) + (0)(0) \\
(0)(0) + (3)(1) & (0)(-1) + (3)(0)
\end{bmatrix}
= \begin{bmatrix} 0 & -2 \\ 3 & 0 \end{bmatrix}
$$

### Step 3: Test on Input Vector $\mathbf{v} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$

$$
A\mathbf{v} = \begin{bmatrix} 0 & -2 \\ 3 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} =
\begin{bmatrix} 0 - 2 \\ 3 + 0 \end{bmatrix} = \begin{bmatrix} -2 \\ 3 \end{bmatrix}
$$

#### Geometric Check:
* Initial vector: $[1, 1]^T$ (length $\sqrt{2}$, angle $45^\circ$).
* Rotated $90^\circ$: $[-1, 1]^T$ (angle $135^\circ$).
* Scaled by $s_x = 2, s_y = 3$: $[-2, 3]^T$ ✓.

---

## 11.8 Why this matters in ML

1. **Computer Vision Data Augmentation:** Random rotations, scalings, reflections, and shears applied during training are represented as 2D affine transformation matrices acting on image coordinate grids.
2. **Feature Space Rotations:** Principal Component Analysis (PCA) is fundamentally a **change of basis** that rotates the feature axes to align with directions of maximum data variance.
3. **Graph Neural Networks & Word Embeddings:** Transforming node features or aligning multilingual embedding spaces involves finding optimal orthogonal transformation matrices $Q$.

---

## 11.9 Common mistakes

* **Treating Affine Maps as Linear Maps:** A function $f(x) = mx + b$ with $b \neq 0$ is NOT a linear transformation because $f(0) = b \neq 0$ (fails additivity: $f(x_1 + x_2) \neq f(x_1) + f(x_2)$). It is an affine transformation.
* **Incorrect Composition Order:** Applying transformation $A$ first, then $B$, produces matrix $BA$, not $AB$. Remember that transformations evaluate right-to-left: $(BA)\mathbf{x} = B(A\mathbf{x})$.

---

> 📖 **Navigation:** [← Previous: Part 10: Four Fundamental Subspaces](./10_four_fundamental_subspaces.md) | [🏠 Index](./README.md) | [Next: Part 12: Orthogonality & Orthonormal Bases →](./12_orthogonality_and_bases.md)
