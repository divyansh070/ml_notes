> 📖 **Navigation:** [← Previous: Part 25: Moore-Penrose Pseudoinverse (A^+)](./25_moore_penrose_pseudoinverse.md) | [🏠 Index](./README.md) | [Next: Part 27: 40 Essential Technical Interview Questions & Answers →](./27_interview_questions_and_answers.md)

---

# PART 26 — LINEAR TRANSFORMATIONS & CHANGE OF BASIS

Matrices are more than just grids of numbers: they are geometric engines that rotate, scale, shear, reflect, and project space. Changing the basis allows us to view these transformations from a perspective where their action becomes remarkably simple.

---

## 26.1 Definition of a Linear Transformation

A transformation $T: \mathbb{R}^n \to \mathbb{R}^m$ is **Linear** if and only if it preserves vector addition and scalar multiplication:

1. **Additivity:** $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
2. **Homogeneity (Scalar Scaling):** $T(c\mathbf{u}) = cT(\mathbf{u})$

Combining both gives the linearity test:

$$
T(c\mathbf{u} + d\mathbf{v}) = cT(\mathbf{u}) + dT(\mathbf{v})
$$

### Fundamental Theorem of Linear Transformations
Every linear transformation $T: \mathbb{R}^n \to \mathbb{R}^m$ is uniquely represented by matrix multiplication:

$$
T(\mathbf{x}) = A\mathbf{x}
$$

where the columns of $A$ are simply the outputs of $T$ acting on the standard unit basis vectors $\mathbf{e}_1, \dots, \mathbf{e}_n$:

$$
A = \begin{bmatrix} T(\mathbf{e}_1) & T(\mathbf{e}_2) & \cdots & T(\mathbf{e}_n) \end{bmatrix}
$$

---

## 26.2 Linear vs Affine Transformations in Neural Networks

```
       LINEAR: T(x) = W x (Origin Fixed)            AFFINE: T(x) = W x + b (Origin Shifted)
                y                                            y
                │     ● T(v)                                 │          ● W v + b
                │    ╱                                       │         ╱
                │   ╱                                        │   b ───● (Shifted Origin)
                └──●────► x                                  └───┴────► x
                 Origin Fixed                                  Origin
```

* **Linear Transformation:** $T(\mathbf{x}) = W\mathbf{x}$. Must map the origin to the origin ($T(\mathbf{0}) = \mathbf{0}$).
* **Affine Transformation:** $T(\mathbf{x}) = W\mathbf{x} + \mathbf{b}$. A linear transformation followed by a translation $\mathbf{b}$.
* **Neural Network Dense Layer:** $\mathbf{z} = W\mathbf{x} + \mathbf{b}$ is an **affine map**. The bias vector $\mathbf{b}$ allows the decision hyperplane to shift away from the origin.

---

## 26.3 The 5 Elementary 2D Geometric Transformations

Every $2 \times 2$ matrix transforms the standard unit square by mapping $\mathbf{i} = [1, 0]^T$ and $\mathbf{j} = [0, 1]^T$:

1. **Scaling (Dilation/Contraction):**
   $$
   A_{\text{scale}} = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix} \implies \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 2x \\ 3y \end{bmatrix}
   $$
2. **Rotation (Counter-Clockwise by Angle $\theta$):**
   $$
   R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \implies R_{90^\circ} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}
   $$
3. **Reflection (Across X-Axis):**
   $$
   A_{\text{reflect}} = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \implies \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} x \\ -y \end{bmatrix}
   $$
4. **Shear (Horizontal Sliding):**
   $$
   A_{\text{shear}} = \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix} \implies \begin{bmatrix} 1 & 1.5 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} x + 1.5y \\ y \end{bmatrix}
   $$
5. **Projection (Onto X-Axis):**
   $$
   P = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix} \implies \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} x \\ 0 \end{bmatrix}
   $$

---

## 26.4 Change of Basis

A **basis** $\mathcal{B} = \{\mathbf{v}_1, \dots, \mathbf{v}_n\}$ provides a coordinate system. A vector $\mathbf{x}$ has coordinates $[\mathbf{x}]_{\mathcal{B}} = [c_1, \dots, c_n]^T$ such that:

$$
\mathbf{x} = c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \dots + c_n \mathbf{v}_n = P [\mathbf{x}]_{\mathcal{B}}
$$

where $P = [\mathbf{v}_1 \dots \mathbf{v}_n]$ is the **change-of-basis matrix**.

### Converting Between Coordinate Systems:
* From new basis $\mathcal{B}$ to standard basis: $\mathbf{x} = P [\mathbf{x}]_{\mathcal{B}}$
* From standard basis to new basis $\mathcal{B}$: $[\mathbf{x}]_{\mathcal{B}} = P^{-1} \mathbf{x}$

### How a Matrix Transformation Changes Under a New Basis:
If matrix $A$ represents a linear transformation in the standard basis, its representation in basis $\mathcal{B}$ is:

$$
B = P^{-1} A P
$$

```
                   SIMILARITY TRANSFORMATION COMMUTATIVE DIAGRAM
                    x (Standard)      ──── A ────►      Ax (Standard)
                         │                                    ▲
                      P^-1                                    P
                         ▼                                    │
                    [x]_B             ──── B ────►      B[x]_B = [Ax]_B
```

---

## 26.5 Diagonalization as the Optimal Change of Basis

When the basis vectors are chosen to be the **eigenvectors** of $A$ ($P = Q = [\mathbf{v}_1 \dots \mathbf{v}_n]$), the transformation matrix becomes purely **diagonal**:

$$
\Lambda = Q^{-1} A Q =
\begin{bmatrix}
\lambda_1 & 0 & \cdots & 0 \\
0 & \lambda_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_n
\end{bmatrix}
$$

* **Geometric Meaning:** In the eigenvector basis, the matrix transformation does not rotate or shear space at all — it simply stretches each coordinate axis independently by factor $\lambda_i$.
* **ML Meaning:** In PCA, projecting data onto the covariance eigenvectors diagonalizes the covariance matrix, completely decorrelating all features!

---

## 26.6 Advanced / Optional — Do not study until Core track is complete

### Similarity Transformations and Matrix Invariants
Two matrices $A$ and $B$ are **similar** ($B = P^{-1} A P$) if they represent the exact same linear transformation under different coordinate bases. Similar matrices share identical:
1. **Determinant:** $\det(B) = \det(P^{-1} A P) = \det(P)^{-1} \det(A) \det(P) = \det(A)$.
2. **Trace:** $\text{Tr}(B) = \text{Tr}(P^{-1} A P) = \text{Tr}(A P P^{-1}) = \text{Tr}(A)$.
3. **Eigenvalues:** $\det(B - \lambda I) = \det(P^{-1}(A - \lambda I)P) = \det(A - \lambda I) = 0$.

---

> 📖 **Navigation:** [← Previous: Part 25: Moore-Penrose Pseudoinverse (A^+)](./25_moore_penrose_pseudoinverse.md) | [🏠 Index](./README.md) | [Next: Part 27: 40 Essential Technical Interview Questions & Answers →](./27_interview_questions_and_answers.md)
