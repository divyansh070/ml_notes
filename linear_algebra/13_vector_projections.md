> 📖 **Navigation:** [← Previous: Part 12: Orthogonality & Orthonormal Bases](./12_orthogonality_and_bases.md) | [🏠 Index](./README.md) | [Next: Part 14: Least Squares & Linear Regression →](./14_least_squares_and_linear_regression.md)

---

# PART 13 — VECTOR PROJECTIONS & PROJECTION MATRICES

**Projection** drops an orthogonal shadow from a vector onto a target line or subspace. In data science, projection is the mathematical engine behind Linear Regression, PCA, and Gram-Schmidt orthogonalization.

---

## 13.1 Projection onto a Line (Vector onto Vector)

```
                                      b
                                     ╱│
                                    ╱ │ Error e = (b - p) is orthogonal to a!
                                   ╱  │
                                  ●───┴──────────► a
                                  0   p = proj_a(b)
```

We wish to project vector $\mathbf{b}$ onto vector $\mathbf{a}$:
1. The projection $\mathbf{p} = \operatorname{proj}_{\mathbf{a}}(\mathbf{b})$ lies on the line spanned by $\mathbf{a}$, so $\mathbf{p} = c \mathbf{a}$ for some scalar $c \in \mathbb{R}$.
2. The error vector $\mathbf{e} = \mathbf{b} - c\mathbf{a}$ must be **perpendicular (orthogonal)** to $\mathbf{a}$:
   $$
   \mathbf{a} \cdot (\mathbf{b} - c\mathbf{a}) = 0 \implies \mathbf{a}^T \mathbf{b} - c (\mathbf{a}^T \mathbf{a}) = 0
   $$
3. Solving for the scalar coefficient $c$:
   $$
   c = \frac{\mathbf{a}^T \mathbf{b}}{\mathbf{a}^T \mathbf{a}} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2^2}
   $$
4. Therefore, the vector projection formula is:
   $$
   \mathbf{p} = \operatorname{proj}_{\mathbf{a}}(\mathbf{b}) = \left( \frac{\mathbf{a}^T \mathbf{b}}{\mathbf{a}^T \mathbf{a}} \right) \mathbf{a}
   $$

---

## 13.2 Optimization Meaning: Projection is the Closest Point

> [!IMPORTANT]
> **The Closest Point Theorem:**
> The orthogonal projection $\mathbf{p} = \operatorname{proj}_S(\mathbf{b})$ is the **unique point in subspace $S$ that minimizes the Euclidean distance to $\mathbf{b}$**:
> $$
> \operatorname{proj}_S(\mathbf{b}) = \arg\min_{\mathbf{z} \in S} \|\mathbf{b} - \mathbf{z}\|_2^2
> $$

* The shortest distance from a point to a line or plane is along the perpendicular line. Any non-orthogonal point creates a right triangle where the hypotenuse is strictly longer than the perpendicular leg (Pythagorean Theorem).

---

## 13.3 Projection Matrix onto a Subspace ($P$)

When projecting vector $\mathbf{b} \in \mathbb{R}^m$ onto the Column Space of a tall matrix $A \in \mathbb{R}^{m \times n}$ ($m > n$, full column rank):

1. The projected vector $\mathbf{p}$ lies in $\operatorname{Col}(A)$, so $\mathbf{p} = A\mathbf{x}$ for some weight vector $\mathbf{x} \in \mathbb{R}^n$.
2. The error vector $\mathbf{e} = \mathbf{b} - A\mathbf{x}$ must be perpendicular to **every column of $A$**:
   $$
   A^T (\mathbf{b} - A\mathbf{x}) = \mathbf{0} \implies A^T A \mathbf{x} = A^T \mathbf{b}
   $$
3. Since $A$ has full column rank, $A^T A$ is invertible:
   $$
   \mathbf{x} = (A^T A)^{-1} A^T \mathbf{b}
   $$
4. The projection vector $\mathbf{p}$ is therefore:
   $$
   \mathbf{p} = A\mathbf{x} = \underbrace{A (A^T A)^{-1} A^T}_{P} \mathbf{b}
   $$

The matrix $P \in \mathbb{R}^{m \times m}$ is the **Projection Matrix**:
$$
P = A (A^T A)^{-1} A^T
$$

---

## 13.4 Master Properties of Projection Matrices

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE 4 PROJECTION MATRIX IDENTITIES                       │
├───────────────────────────────────┬─────────────────────────────────────────┤
│ 1. Symmetry:                      │ P^T = P                                 │
│ 2. Idempotency:                   │ P^2 = P  (Projecting twice does nothing)│
│ 3. Binary Spectrum:               │ Eigenvalues λ_i ∈ {0, 1} strictly       │
│ 4. Trace equals Subspace Rank:    │ Tr(P) = rank(P) = dim(Subspace S)       │
└───────────────────────────────────┴─────────────────────────────────────────┘
```

> [!NOTE]
> **Why $P^2 = P$ (Idempotency)?**
> Geometrically: once a vector is projected onto subspace $S$, it already lives in $S$. Dropping a shadow a second time leaves it completely unchanged.
> Algebraically:
> $$
> P^2 = [A (A^T A)^{-1} A^T] [A (A^T A)^{-1} A^T] = A (A^T A)^{-1} [(A^T A) (A^T A)^{-1}] A^T = A (A^T A)^{-1} A^T = P \quad \checkmark
> $$

---

## 13.5 The Orthogonal Complement Projector ($I - P$)

If $P$ projects onto subspace $S = \operatorname{Col}(A)$, then $(I - P)$ is the **orthogonal projection matrix onto the residual subspace $S^\perp = N(A^T)$**:

$$
\mathbf{b} = \underbrace{P \mathbf{b}}_{\mathbf{p} \text{ (Fitted Component)}} + \underbrace{(I - P) \mathbf{b}}_{\mathbf{e} \text{ (Residual Error)}}
$$

* Note that $(I - P)$ is also symmetric and idempotent: $(I - P)^2 = I - 2P + P^2 = I - 2P + P = I - P$.

---

## 13.6 Complete Worked Numerical Example

### Task 1: Project a vector onto a line
Project $\mathbf{b} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$ onto $\mathbf{a} = \begin{bmatrix} 4 \\ 0 \end{bmatrix}$:
* $\mathbf{a}^T \mathbf{b} = (4)(3) + (0)(4) = 12$
* $\mathbf{a}^T \mathbf{a} = 4^2 + 0^2 = 16$
* Projection:
  $$
  \mathbf{p} = \frac{12}{16} \begin{bmatrix} 4 \\ 0 \end{bmatrix} = \frac{3}{4} \begin{bmatrix} 4 \\ 0 \end{bmatrix} = \begin{bmatrix} 3 \\ 0 \end{bmatrix}
  $$
* Residual error:
  $$
  \mathbf{e} = \mathbf{b} - \mathbf{p} = \begin{bmatrix} 3 \\ 4 \end{bmatrix} - \begin{bmatrix} 3 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 4 \end{bmatrix}
  $$
* Orthogonality verification: $\mathbf{e} \cdot \mathbf{a} = (0)(4) + (4)(0) = 0 \quad \checkmark$.

---

### Task 2: Construct Projection Matrix for Subspace
Let $A = \begin{bmatrix} 1 \\ 2 \end{bmatrix} \in \mathbb{R}^{2 \times 1}$. Compute $P$:
1. $A^T A = [1, 2] \begin{bmatrix} 1 \\ 2 \end{bmatrix} = 1^2 + 2^2 = 5$.
2. $(A^T A)^{-1} = \frac{1}{5} = 0.2$.
3. $P = A (A^T A)^{-1} A^T = \frac{1}{5} \begin{bmatrix} 1 \\ 2 \end{bmatrix} [1, 2] = \frac{1}{5} \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix} = \begin{bmatrix} 0.2 & 0.4 \\ 0.4 & 0.8 \end{bmatrix}$.
4. Check symmetry: $P^T = P \quad \checkmark$.
5. Check idempotency:
   $$
   P^2 = \begin{bmatrix} 0.2 & 0.4 \\ 0.4 & 0.8 \end{bmatrix} \begin{bmatrix} 0.2 & 0.4 \\ 0.4 & 0.8 \end{bmatrix} = \begin{bmatrix} 0.04 + 0.16 & 0.08 + 0.32 \\ 0.08 + 0.32 & 0.16 + 0.64 \end{bmatrix} = \begin{bmatrix} 0.2 & 0.4 \\ 0.4 & 0.8 \end{bmatrix} = P \quad \checkmark
   $$

---

## 13.7 Why this matters in ML

1. **Ordinary Least Squares is Pure Projection:** In regression, the design matrix $X$ rarely reaches target $\mathbf{y}$. The fitted prediction $\hat{\mathbf{y}} = X\mathbf{w} = X(X^T X)^{-1} X^T \mathbf{y} = P \mathbf{y}$ is literally the orthogonal projection of $\mathbf{y}$ onto the feature space!
2. **The Hat Matrix in Statistics:** In econometrics and statistics, $P = X(X^T X)^{-1} X^T$ is known as the **Hat Matrix** $H$ because it "puts the hat on $\mathbf{y}$" ($\hat{\mathbf{y}} = H\mathbf{y}$). Diagonal elements $h_{ii}$ measure sample leverage.

---

## 13.8 Common mistakes

* **The "Cancellation Fallacy":** Writing $A(A^T A)^{-1} A^T = A A^{-1} (A^T)^{-1} A^T = I$. This is **FATALLY WRONG** when $A$ is rectangular ($m > n$) because individual non-square matrices $A$ do not possess ordinary inverses $A^{-1}$. You cannot distribute the inverse over $A^T A$!
* **Projecting onto Non-Unit Vectors without Dividing by Length:** Writing $\mathbf{p} = (\mathbf{a} \cdot \mathbf{b})\mathbf{a}$. This is only true if $\|\mathbf{a}\|_2 = 1$. Otherwise, you must divide by $\|\mathbf{a}\|_2^2$.

---

> 📖 **Navigation:** [← Previous: Part 12: Orthogonality & Orthonormal Bases](./12_orthogonality_and_bases.md) | [🏠 Index](./README.md) | [Next: Part 14: Least Squares & Linear Regression →](./14_least_squares_and_linear_regression.md)
