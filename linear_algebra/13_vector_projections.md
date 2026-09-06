> 📖 **Navigation:** [← Previous: Part 12: Orthogonality & Orthonormal Bases](./12_orthogonality_and_bases.md) | [🏠 Index](./README.md) | [Next: Part 14: Linear Regression Matrix Mathematics →](./14_linear_regression_matrix_math.md)

---

# PART 13 — VECTOR PROJECTIONS & PROJECTION MATRICES

**Projection** drops an orthogonal shadow from a vector onto a target line or subspace. In data science, projection is the mathematical engine behind Linear Regression, PCA, and dimensionality reduction.

---

## 13.1 Projection onto a Line (Vector onto Vector)

```
                                      a
                                     ╱│
                                    ╱ │ Error e = (a - p) is orthogonal to b!
                                   ╱  │
                                  ●───┴──────────► b
                                  0   p = proj_b(a)
```

We wish to project vector $\mathbf{a}$ onto vector $\mathbf{b}$:
1. The projection $\mathbf{p} = \text{proj}_{\mathbf{b}}(\mathbf{a})$ lies along $\mathbf{b}$, so $\mathbf{p} = c \mathbf{b}$ for scalar $c$.
2. The error vector $\mathbf{e} = (\mathbf{a} - c\mathbf{b})$ must be **orthogonal** to $\mathbf{b}$:
   $$
   \mathbf{b} \cdot (\mathbf{a} - c\mathbf{b}) = 0 \implies \mathbf{b}^T \mathbf{a} - c(\mathbf{b}^T \mathbf{b}) = 0 \implies c = \frac{\mathbf{a}^T \mathbf{b}}{\mathbf{b}^T \mathbf{b}}
   $$
3. Therefore:
   $$
   \mathbf{p} = \text{proj}_{\mathbf{b}}(\mathbf{a}) = \left( \frac{\mathbf{a}^T \mathbf{b}}{\mathbf{b}^T \mathbf{b}} \right) \mathbf{b}
   $$

### Hand Calculation Example:
Project $\mathbf{a} = [3, 4]^T$ onto $\mathbf{b} = [4, 0]^T$:
* $\mathbf{a}^T \mathbf{b} = (3 \times 4) + (4 \times 0) = 12$
* $\mathbf{b}^T \mathbf{b} = (4 \times 4) + (0 \times 0) = 16$
* $\mathbf{p} = \frac{12}{16} [4, 0]^T = [3, 0]^T$
* Error: $\mathbf{e} = \mathbf{a} - \mathbf{p} = [0, 4]^T \implies \mathbf{e} \cdot \mathbf{b} = (0 \times 4) + (4 \times 0) = 0 \quad \checkmark$.

---

## 13.2 Optimization Interpretation: Projection is the Closest Point

> [!IMPORTANT]
> **The Closest Point Theorem:**
> The projection $\mathbf{p} = \text{proj}_S(\mathbf{x})$ is the **unique point in subspace $S$ that minimizes the Euclidean distance to $\mathbf{x}$**:
> $$
> \text{proj}_S(\mathbf{x}) = \arg\min_{\mathbf{z} \in S} \|\mathbf{x} - \mathbf{z}\|_2^2
> $$

* The shortest distance from a point to a plane is along the perpendicular line. Thus, the error $(\mathbf{x} - \mathbf{p})$ is strictly orthogonal to subspace $S$.

---

## 13.3 Projection Matrix onto a Multi-Dimensional Subspace ($P$)

When projecting onto the column space of a matrix $X \in \mathbb{R}^{n \times d}$ (where $n > d$ and $X$ has full column rank):

$$
P = X (X^T X)^{-1} X^T
$$

* **Projected vector:** $\hat{\mathbf{y}} = P \mathbf{y} = X (X^T X)^{-1} X^T \mathbf{y}$.

---

## 13.4 Master Properties of Projection Matrices

```
                  THE 4 PROJECTION MATRIX PROPERTIES
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 1. Symmetry:           P^T = P                                         │
  │ 2. Idempotency:        P^2 = P  (Projecting a second time changes zero)│
  │ 3. Binary Spectrum:    Eigenvalues λ_i ∈ {0, 1} strictly               │
  │ 4. Rank equals Trace:  rank(P) = Tr(P) = dim(Subspace S)               │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 13.5 The Orthogonal Complement Projector ($I - P$)

If $P$ projects onto subspace $S = \text{Col}(X)$, then $(I - P)$ is the **orthogonal projection matrix onto the residual subspace $S^\perp = \text{Null}(X^T)$**:

$$
\mathbf{y} = \underbrace{P \mathbf{y}}_{\hat{\mathbf{y}} \text{ (Fitted Prediction)}} + \underbrace{(I - P) \mathbf{y}}_{\mathbf{e} \text{ (Residual Error)}}
$$

---

## 13.6 Direct Connection to Linear Regression Geometry

```
                                 y (Actual Target)
                                ╱│
                               ╱ │ Residual e = (I - P)y
                              ╱  │ (Orthogonal to Feature Space!)
                             ╱   │
                            ●────┴────────► Col(X) (Feature Subspace)
                            0    ŷ = Py = Xw
```

In Ordinary Least Squares, because $X\mathbf{w} = \mathbf{y}$ has no exact solution ($\mathbf{y} \notin \text{Col}(X)$), we solve for the **closest point in feature space** $\hat{\mathbf{y}} = P\mathbf{y} = X(X^T X)^{-1} X^T \mathbf{y}$, giving normal equation weights $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$!

---

> 📖 **Navigation:** [← Previous: Part 12: Orthogonality & Orthonormal Bases](./12_orthogonality_and_bases.md) | [🏠 Index](./README.md) | [Next: Part 14: Linear Regression Matrix Mathematics →](./14_linear_regression_matrix_math.md)
