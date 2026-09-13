> 📖 **Navigation:** [← Previous: Part 09: Vector Spaces & Subspaces](./09_vector_spaces_and_subspaces.md) | [🏠 Index](./README.md) | [Next: Part 11: Linear Transformations →](./11_linear_transformations.md)

---

# PART 10 — THE FOUR FUNDAMENTAL SUBSPACES

Linear algebra reaches its conceptual pinnacle in the **Four Fundamental Subspaces** of a matrix $A \in \mathbb{R}^{m \times n}$. Championed by Prof. Gilbert Strang (MIT 18.06), this unifying framework explains linear systems, matrix rank, orthogonality, projections, and least squares under a single geometric roof.

---

## 10.1 Strang's Big Picture: Two Spaces, Four Subspaces

Every matrix $A \in \mathbb{R}^{m \times n}$ acts as a linear mapping from an **Input Space $\mathbb{R}^n$** to an **Output Space $\mathbb{R}^m$**:

$$
\mathbf{x} \in \mathbb{R}^n \quad \xrightarrow{\quad A \quad} \quad A\mathbf{x} \in \mathbb{R}^m
$$

Both Euclidean spaces split into **two mutually perpendicular (orthogonal complement) subspaces**:

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

---

## 10.2 Summary Table of the Four Subspaces

For any matrix $A \in \mathbb{R}^{m \times n}$ with $\text{rank}(A) = r$:

| Subspace | Symbol | Ambient Space | Dimension | Orthogonal Complement | Meaning in ML / Systems |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Column Space** | $C(A)$ | $\mathbb{R}^m$ | $r$ | $N(A^T)$ | All reachable outputs $A\mathbf{x}$; model predictions $\hat{\mathbf{y}}$ |
| **Null Space** | $N(A)$ | $\mathbb{R}^n$ | $n - r$ | $C(A^T)$ | Feature directions squashed to zero; collinear redundancies |
| **Row Space** | $C(A^T)$ | $\mathbb{R}^n$ | $r$ | $N(A)$ | True information directions in the feature space |
| **Left Null Space**| $N(A^T)$ | $\mathbb{R}^m$ | $m - r$ | $C(A)$ | Unreachable output directions; residual errors $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$ |

---

## 10.3 The Rank-Nullity Theorem & Orthogonal Complements

### 1. The Rank-Nullity Theorem (Conservation Law)
For any $m \times n$ matrix with $n$ input columns:

$$
\text{rank}(A) + \text{nullity}(A) = n
$$

*(Surviving Output Dimensions) $+$ (Destroyed Input Dimensions) $=$ (Total Input Dimensions).*

### 2. Why $C(A^T) \perp N(A)$ (Proof of Orthogonality)
Let $\mathbf{x} \in N(A)$. By definition, $A\mathbf{x} = \mathbf{0}$:

$$
\begin{bmatrix}
\text{---} & \mathbf{r}_1^T & \text{---} \\
\text{---} & \mathbf{r}_2^T & \text{---} \\
& \vdots & \\
\text{---} & \mathbf{r}_m^T & \text{---}
\end{bmatrix} \mathbf{x} =
\begin{bmatrix}
\mathbf{r}_1 \cdot \mathbf{x} \\
\mathbf{r}_2 \cdot \mathbf{x} \\
\vdots \\
\mathbf{r}_m \cdot \mathbf{x}
\end{bmatrix} =
\begin{bmatrix} 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix}
$$

Every single row $\mathbf{r}_i$ has dot product $0$ with $\mathbf{x}$. Because any vector in the row space is a linear combination of rows, **every row vector is strictly perpendicular to every null space vector**:

$$
C(A^T) \perp N(A)
$$

### 3. Unique Vector Decomposition:
Every input vector $\mathbf{x} \in \mathbb{R}^n$ splits uniquely into:

$$
\mathbf{x} = \mathbf{x}_{\text{row}} + \mathbf{x}_{\text{null}}
$$

where $\mathbf{x}_{\text{row}} \in C(A^T)$ and $\mathbf{x}_{\text{null}} \in N(A)$. When $A$ multiplies $\mathbf{x}$:

$$
A\mathbf{x} = A(\mathbf{x}_{\text{row}} + \mathbf{x}_{\text{null}}) = A\mathbf{x}_{\text{row}} + \mathbf{0} = A\mathbf{x}_{\text{row}}
$$

The null space component contributes nothing to the output!

---

## 10.4 Complete Non-Square Worked Numerical Example ($2 \times 3$ Matrix)

Analyze all four fundamental subspaces for:

$$
A = \begin{bmatrix}
1 & 0 & 2 \\
0 & 1 & -1
\end{bmatrix} \in \mathbb{R}^{2 \times 3}
$$

* $m = 2$ (Output space is $\mathbb{R}^2$).
* $n = 3$ (Input space is $\mathbb{R}^3$).
* The matrix is already in RREF. Both rows have pivots, so rank $r = 2$.

---

### Subspace 1: Row Space $C(A^T) \subseteq \mathbb{R}^3$
* **Dimension:** $r = 2$.
* **Basis:** The non-zero rows of the RREF matrix:

$$
\mathcal{B}_{C(A^T)} = \left\lbrace \begin{bmatrix} 1 \\ 0 \\ 2 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \\ -1 \end{bmatrix} \right\rbrace
$$

  *(A 2D plane passing through the origin in $\mathbb{R}^3$).*

---

### Subspace 2: Null Space $N(A) \subseteq \mathbb{R}^3$
* **Dimension:** $n - r = 3 - 2 = 1$.
* Solve $A\mathbf{x} = \mathbf{0}$:
  * $x_1 + 2x_3 = 0 \implies x_1 = -2t$
  * $x_2 - x_3 = 0 \implies x_2 = t$
  * $x_3 = t$ (free variable)
* **Basis:**

$$
\mathcal{B}_{N(A)} = \left\lbrace \begin{bmatrix} -2 \\ 1 \\ 1 \end{bmatrix} \right\rbrace
$$

  *(A 1D line passing through the origin in $\mathbb{R}^3$).*

#### Orthogonality Check in $\mathbb{R}^3$:
* Row 1 dot Null: $(1)(-2) + (0)(1) + (2)(1) = -2 + 0 + 2 = 0$ ✓
* Row 2 dot Null: $(0)(-2) + (1)(1) + (-1)(1) = 0 + 1 - 1 = 0$ ✓
* Check Rank-Nullity: $r + (n - r) = 2 + 1 = 3 = n$ ✓.

---

### Subspace 3: Column Space $C(A) \subseteq \mathbb{R}^2$
* **Dimension:** $r = 2$.
* Since $C(A) \subseteq \mathbb{R}^2$ and has dimension $2$, it spans the **entire output space**:

$$
C(A) = \mathbb{R}^2
$$

* **Basis:** Any two independent columns, such as standard basis vectors $[1, 0]^T$ and $[0, 1]^T$.

---

### Subspace 4: Left Null Space $N(A^T) \subseteq \mathbb{R}^2$
* **Dimension:** $m - r = 2 - 2 = 0$.
* The left null space contains **only the trivial zero vector**:

$$
N(A^T) = \left\lbrace \begin{bmatrix} 0 \\ 0 \end{bmatrix} \right\rbrace
$$

* Check orthogonal sum in $\mathbb{R}^2$: $\dim(C(A)) + \dim(N(A^T)) = 2 + 0 = 2 = m$ ✓.

---

## 10.5 Why this matters in ML

1. **Ordinary Least Squares Residuals:** In regression $\mathbf{y} = X\mathbf{w} + \mathbf{e}$:
   * The model prediction $\hat{\mathbf{y}} = X\mathbf{w}$ lies in the Column Space $C(X)$.
   * The residual error vector $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$ lies in the Left Null Space $N(X^T)$.
   * Because $C(X) \perp N(X^T)$, the residual error is **strictly orthogonal to every feature column**:

$$
X^T \mathbf{e} = \mathbf{0}
$$

2. **Feature Collinearity:** If design matrix $X$ has redundant features, $N(X)$ is non-trivial ($\dim(N(X)) > 0$), meaning multiple weight vectors produce identical predictions.

---

## 10.6 Common mistakes

* **Confusing Ambient Spaces:** Assuming $C(A)$ and $C(A^T)$ live in the same space. If $A$ is $2 \times 3$, $C(A^T)$ lives in $\mathbb{R}^3$ while $C(A)$ lives in $\mathbb{R}^2$. They only share the same ambient space if $A$ is square.
* **Forgetting that Row Rank Equals Column Rank:** A rectangular $1000 \times 3$ matrix can NEVER have a row space with dimension greater than 3, because column rank is at most 3.

---

> 📖 **Navigation:** [← Previous: Part 09: Vector Spaces & Subspaces](./09_vector_spaces_and_subspaces.md) | [🏠 Index](./README.md) | [Next: Part 11: Linear Transformations →](./11_linear_transformations.md)
