> 📖 **Navigation:** [← Previous: Part 21: Paper-and-Pencil Exam Checklist](./21_paper_and_pencil_checklist.md) | [🏠 Index](./README.md) | [Next: Part 23: Gram-Schmidt Orthogonalization & QR Decomposition →](./23_gram_schmidt_and_qr_decomposition.md)

---

# PART 22 — FOUR FUNDAMENTAL SUBSPACES (STRANG'S BIG PICTURE)

Linear algebra reaches its conceptual pinnacle in the **Four Fundamental Subspaces** of a matrix $A \in \mathbb{R}^{m \times n}$. Developed and championed by Prof. Gilbert Strang (MIT 18.06), this unifying framework explains systems of linear equations, matrix rank, orthogonality, projections, least-squares regression, and SVD under one geometric roof.

---

## 22.1 The Big Picture Overview

Every matrix $A \in \mathbb{R}^{m \times n}$ acts as a linear map from an **Input Space $\mathbb{R}^n$** to an **Output Space $\mathbb{R}^m$**:

$$
\mathbf{x} \in \mathbb{R}^n \quad \xrightarrow{\quad A \quad} \quad A\mathbf{x} \in \mathbb{R}^m
$$

These two Euclidean spaces each split into **two mutually perpendicular (orthogonal complement) subspaces**:

```
        INPUT SPACE R^n (n dimensions)                     OUTPUT SPACE R^m (m dimensions)
   ┌─────────────────────────────────────┐            ┌─────────────────────────────────────┐
   │                                     │            │                                     │
   │           ROW SPACE                 │            │          COLUMN SPACE               │
   │           C(A^T)                    │   ──A──►   │          C(A)                       │
   │         Dimension = r               │  1-to-1    │        Dimension = r                │
   │                                     │  isomorphic│                                     │
   │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │            │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
   │           NULL SPACE                │            │       LEFT NULL SPACE               │
   │           N(A)                      │   ──A──►   │          N(A^T)                     │
   │        Dimension = n - r            │  maps to   │        Dimension = m - r            │
   │                                     │  vector 0  │                                     │
   └─────────────────────────────────────┘            └─────────────────────────────────────┘
        Orthogonal Complements:                            Orthogonal Complements:
           C(A^T) ⊥ N(A)                                      C(A) ⊥ N(A^T)
           C(A^T) ⊕ N(A) = R^n                                C(A) ⊕ N(A^T) = R^m
```

### Summary Table of the Four Subspaces

| Subspace | Symbol | Ambient Space | Dimension | Orthogonal Complement | Meaning in ML / Systems |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Column Space** | $C(A)$ | $\mathbb{R}^m$ | $r = \text{rank}(A)$ | $N(A^T)$ | All reachable outputs $A\mathbf{x}$; model predictions $\hat{\mathbf{y}}$ |
| **Null Space** | $N(A)$ | $\mathbb{R}^n$ | $n - r$ | $C(A^T)$ | Feature directions squashed to zero; collinear redundancies |
| **Row Space** | $C(A^T)$ | $\mathbb{R}^n$ | $r = \text{rank}(A)$ | $N(A)$ | True information directions in the feature space |
| **Left Null Space** | $N(A^T)$ | $\mathbb{R}^m$ | $m - r$ | $C(A)$ | Unreachable output directions; residual error vectors $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$ |

---

## 22.2 Deep Dive into Each Subspace

### 1. Column Space $C(A)$
* **Definition:** The set of all linear combinations of the columns of $A$:
  $$
  C(A) = \{\mathbf{b} \in \mathbb{R}^m \mid \mathbf{b} = A\mathbf{x} \text{ for some } \mathbf{x} \in \mathbb{R}^n\}
  $$
* **Solvability Condition:** The linear system $A\mathbf{x} = \mathbf{b}$ has an exact solution if and only if $\mathbf{b} \in C(A)$.
* **Dimension:** $\dim(C(A)) = r$ (column rank).

### 2. Null Space $N(A)$ (Kernel)
* **Definition:** The set of all input vectors $\mathbf{x} \in \mathbb{R}^n$ that $A$ sends to the zero vector $\mathbf{0} \in \mathbb{R}^m$:
  $$
  N(A) = \{\mathbf{x} \in \mathbb{R}^n \mid A\mathbf{x} = \mathbf{0}\}
  $$
* **Dimension:** $\dim(N(A)) = n - r$ (**nullity**).
* **Rank-Nullity Theorem:**
  $$
  \text{rank}(A) + \text{nullity}(A) = n
  $$

### 3. Row Space $C(A^T)$
* **Definition:** The span of the row vectors of $A$ (which are the columns of $A^T$):
  $$
  C(A^T) = \{\mathbf{v} \in \mathbb{R}^n \mid \mathbf{v} = A^T \mathbf{y} \text{ for some } \mathbf{y} \in \mathbb{R}^m\}
  $$
* **Fundamental Miracle:** Row rank equals column rank:
  $$
  \dim(C(A^T)) = \dim(C(A)) = r
  $$

### 4. Left Null Space $N(A^T)$ (Cokernel)
* **Definition:** The null space of $A^T$:
  $$
  N(A^T) = \{\mathbf{y} \in \mathbb{R}^m \mid A^T \mathbf{y} = \mathbf{0}\} \iff \{\mathbf{y} \in \mathbb{R}^m \mid \mathbf{y}^T A = \mathbf{0}^T\}
  $$
* **Why "Left"?** Multiplying both sides shows that $\mathbf{y}^T$ multiplies $A$ from the *left*.
* **Dimension:** $\dim(N(A^T)) = m - r$.

---

## 22.3 Orthogonal Complements and Vector Decomposition

The two pairs of subspaces are **orthogonal complements**:

$$
C(A^T) \perp N(A) \quad \text{in } \mathbb{R}^n, \qquad C(A) \perp N(A^T) \quad \text{in } \mathbb{R}^m
$$

### Why $C(A^T) \perp N(A)$?
Let $\mathbf{x} \in N(A)$. By definition, $A\mathbf{x} = \mathbf{0}$:

$$
\begin{bmatrix}
\text{---} & \mathbf{r}_1^T & \text{---} \\
\text{---} & \mathbf{r}_2^T & \text{---} \\
& \vdots & \\
\text{---} & \mathbf{r}_m^T & \text{---}
\end{bmatrix}
\mathbf{x} =
\begin{bmatrix}
\mathbf{r}_1 \cdot \mathbf{x} \\
\mathbf{r}_2 \cdot \mathbf{x} \\
\vdots \\
\mathbf{r}_m \cdot \mathbf{x}
\end{bmatrix} =
\begin{bmatrix}
0 \\
0 \\
\vdots \\
0
\end{bmatrix}
$$

Every row $\mathbf{r}_i$ has dot product 0 with $\mathbf{x}$. Since any vector in the row space is a linear combination of rows, **every row vector is perpendicular to every null vector**.

### Orthogonal Decomposition of Any Input Vector $\mathbf{x}$
Every vector $\mathbf{x} \in \mathbb{R}^n$ splits uniquely into:

$$
\mathbf{x} = \mathbf{x}_{\text{row}} + \mathbf{x}_{\text{null}}
$$

where $\mathbf{x}_{\text{row}} \in C(A^T)$ and $\mathbf{x}_{\text{null}} \in N(A)$. When $A$ multiplies $\mathbf{x}$:

$$
A\mathbf{x} = A(\mathbf{x}_{\text{row}} + \mathbf{x}_{\text{null}}) = A\mathbf{x}_{\text{row}} + A\mathbf{x}_{\text{null}} = A\mathbf{x}_{\text{row}} + \mathbf{0} = A\mathbf{x}_{\text{row}}
$$

> [!IMPORTANT]
> $A$ is a **bijective (1-to-1 and onto) linear map** from the Row Space $C(A^T)$ to the Column Space $C(A)$. The Null Space $N(A)$ contributes nothing to the output.

---

## 22.4 Non-Square Hand Calculation Example ($2 \times 3$ Matrix)

To clearly see how the ambient spaces $\mathbb{R}^n$ and $\mathbb{R}^m$ differ, let's analyze a rectangular matrix where $m = 2$ and $n = 3$:

$$
A =
\begin{bmatrix}
1 & 0 & 2 \\
0 & 1 & -1
\end{bmatrix} \in \mathbb{R}^{2 \times 3}
$$

### Step 1: Identify Dimensions and Rank
* $m = 2$ (output space is $\mathbb{R}^2$).
* $n = 3$ (input space is $\mathbb{R}^3$).
* The rows are clearly linearly independent, so rank $r = 2$.

### Step 2: Row Space $C(A^T) \subset \mathbb{R}^3$
* Spanned by the two independent rows of $A$:
  $$
  C(A^T) = \text{span}\left( \begin{bmatrix} 1 \\ 0 \\ 2 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \\ -1 \end{bmatrix} \right) \subset \mathbb{R}^3
  $$
* $\dim(C(A^T)) = r = 2$ (a 2D plane in 3D space).

### Step 3: Null Space $N(A) \subset \mathbb{R}^3$
Solve $A\mathbf{x} = \mathbf{0}$:

$$
\begin{bmatrix}
1 & 0 & 2 \\
0 & 1 & -1
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2 \\ x_3
\end{bmatrix} =
\begin{bmatrix}
0 \\ 0
\end{bmatrix}
\implies
\begin{cases}
x_1 + 2x_3 = 0 \implies x_1 = -2x_3 \\
x_2 - x_3 = 0 \implies x_2 = x_3
\end{cases}
$$

$x_3$ is a free variable ($x_3 = t$):

$$
\mathbf{x} = t \begin{bmatrix} -2 \\ 1 \\ 1 \end{bmatrix} \implies N(A) = \text{span}\left( \begin{bmatrix} -2 \\ 1 \\ 1 \end{bmatrix} \right) \subset \mathbb{R}^3
$$

* $\dim(N(A)) = n - r = 3 - 2 = 1$ (a 1D line in 3D space).
* **Orthogonality Check in $\mathbb{R}^3$:**
  * Row 1 dot Null: $(1)(-2) + (0)(1) + (2)(1) = -2 + 0 + 2 = 0 \quad \checkmark$
  * Row 2 dot Null: $(0)(-2) + (1)(1) + (-1)(1) = 0 + 1 - 1 = 0 \quad \checkmark$

### Step 4: Column Space $C(A) \subset \mathbb{R}^2$
* The columns span all of $\mathbb{R}^2$:
  $$
  C(A) = \text{span}\left( \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \end{bmatrix} \right) = \mathbb{R}^2
  $$
* $\dim(C(A)) = r = 2$.

### Step 5: Left Null Space $N(A^T) \subset \mathbb{R}^2$
Solve $A^T \mathbf{y} = \mathbf{0}$:

$$
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
2 & -1
\end{bmatrix}
\begin{bmatrix}
y_1 \\ y_2
\end{bmatrix} =
\begin{bmatrix}
0 \\ 0 \\ 0
\end{bmatrix}
\implies y_1 = 0, y_2 = 0
$$

* $N(A^T) = \{\mathbf{0}\} \subset \mathbb{R}^2$.
* $\dim(N(A^T)) = m - r = 2 - 2 = 0$.
* Check: $\dim(C(A)) + \dim(N(A^T)) = 2 + 0 = 2 = m \quad \checkmark$.

---

## 22.5 ML Connection: Least Squares and Residual Errors

In linear regression, we observe features $X \in \mathbb{R}^{n \times d}$ and targets $\mathbf{y} \in \mathbb{R}^n$.

1. **Target Decomposition:**
   The observed target vector $\mathbf{y} \in \mathbb{R}^n$ splits into an in-subspace prediction and an out-of-subspace error:
   $$
   \mathbf{y} = \hat{\mathbf{y}} + \mathbf{e}
   $$
   * $\hat{\mathbf{y}} = X\mathbf{w}_{\text{LS}} \in C(X)$ (the projection onto the Column Space).
   * $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} \in N(X^T)$ (the residual error vector in the **Left Null Space**).

2. **Why Normal Equations Hold:**
   Because the error $\mathbf{e}$ is orthogonal to $C(X)$, it must lie in $N(X^T)$:
   $$
   X^T \mathbf{e} = \mathbf{0} \implies X^T (\mathbf{y} - X\mathbf{w}) = \mathbf{0} \implies X^T X \mathbf{w} = X^T \mathbf{y}
   $$

```
                           THE GEOMETRY OF LEAST SQUARES
                                        y (Target in R^m)
                                       ╱│
                                      ╱ │
                                     ╱  │  e = y - y_hat ∈ N(X^T)
                                    ╱   │  (Residual Error in Left Null Space)
                                   ╱    │
                                  ▼     ▼
                                 0 ─────●────────────► C(X) in R^m
                                     y_hat = X w       (Column Space / Model Predictions)
```

> [!TIP]
> **Common Interview Question:** *"Where does the residual error vector $\mathbf{e} = \mathbf{y} - X\mathbf{w}$ live in the four fundamental subspaces?"*
> **Answer:** It lives in the **Left Null Space** $N(X^T)$. The normal equations require $\mathbf{e}$ to be orthogonal to every feature column in $X$, which means $X^T \mathbf{e} = \mathbf{0}$, the exact definition of $N(X^T)$.

---

## 22.6 Summary Checklist

- [x] Know the ambient space and dimension of each of the 4 subspaces.
- [x] Know that $C(A^T) \perp N(A)$ in $\mathbb{R}^n$ and $C(A) \perp N(A^T)$ in $\mathbb{R}^m$.
- [x] Know that $\dim(C(A)) = \dim(C(A^T)) = r = \text{rank}(A)$.
- [x] Understand how $\mathbf{y} = \hat{\mathbf{y}} + \mathbf{e}$ decomposes the regression target between $C(X)$ and $N(X^T)$.

---

> 📖 **Navigation:** [← Previous: Part 21: Paper-and-Pencil Exam Checklist](./21_paper_and_pencil_checklist.md) | [🏠 Index](./README.md) | [Next: Part 23: Gram-Schmidt Orthogonalization & QR Decomposition →](./23_gram_schmidt_and_qr_decomposition.md)
