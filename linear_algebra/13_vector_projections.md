> 📖 **Navigation:** [← Previous: Part 12: Orthogonality & Orthonormal Bases](./12_orthogonality_and_bases.md) | [🏠 Index](./README.md) | [Next: Part 14: Linear Regression Matrix Mathematics →](./14_linear_regression_matrix_math.md)

---

# PART 12 — VECTOR PROJECTIONS

---

## 12.1 Derivation of the Scalar & Vector Projection Formula

```
                                      a
                                     ╱│
                                    ╱ │ Error (a - p) is orthogonal to b!
                                   ╱  │
                                  ●───┴──────────► b
                                  0   p = proj_b(a)
```

We wish to drop a perpendicular shadow from vector $\mathbf{a}$ onto vector $\mathbf{b}$.
1. The projection $\mathbf{p} = \text{proj}_{\mathbf{b}}(\mathbf{a})$ lies along the direction of $\mathbf{b}$, so $\mathbf{p} = c \mathbf{b}$ for some scalar $c$.
2. The error vector $(\mathbf{a} - c\mathbf{b})$ must be **orthogonal** to $\mathbf{b}$:

$$
\mathbf{b} \cdot (\mathbf{a} - c\mathbf{b}) = 0
$$

3. Expand the dot product:

$$
\mathbf{b} \cdot \mathbf{a} - c(\mathbf{b} \cdot \mathbf{b}) = 0 \implies c = \frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{b} \cdot \mathbf{b}} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{b}\|_2^2}
$$

4. Multiply scalar $c$ by vector $\mathbf{b}$:

$$
\text{proj}_{\mathbf{b}}(\mathbf{a}) = \left(\frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{b} \cdot \mathbf{b}}\right) \mathbf{b}
$$

---

## 12.2 Step-by-Step Numerical Hand Calculation

Project vector $\mathbf{a} = [3, 4]^T$ onto vector $\mathbf{b} = [4, 0]^T$:
1. Compute $\mathbf{a} \cdot \mathbf{b} = (3 \times 4) + (4 \times 0) = 12 + 0 = 12$.
2. Compute $\mathbf{b} \cdot \mathbf{b} = (4 \times 4) + (0 \times 0) = 16 + 0 = 16$.
3. Compute projection vector:

$$
\mathbf{p} = \frac{12}{16}
\begin{bmatrix}
4 \\
0
\end{bmatrix}
= \frac{3}{4}
\begin{bmatrix}
4 \\
0
\end{bmatrix} =
\begin{bmatrix}
3 \\
0
\end{bmatrix}
$$

4. Verification of Orthogonal Error: $\mathbf{e} = \mathbf{a} - \mathbf{p} = [3, 4]^T - [3, 0]^T = [0, 4]^T$.
   * $\mathbf{e} \cdot \mathbf{b} = (0 \times 4) + (4 \times 0) = 0 \quad \checkmark$

---

## 12.3 Projection Matrices & Subspaces

The **Projection Matrix** $P$ that projects any arbitrary vector onto the column space of a matrix $X$:

$$
P = X (X^T X)^{-1} X^T
$$

* **Property:** $P^2 = P$ (Projecting a second time changes nothing).
* **ML Connection:** Linear regression predictions $\hat{\mathbf{y}} = X \mathbf{w} = X(X^T X)^{-1} X^T \mathbf{y} = P \mathbf{y}$ is the orthogonal projection of target vector $\mathbf{y}$ onto the column space of feature matrix $X$.

---

> 📖 **Navigation:** [← Previous: Part 12: Orthogonality & Orthonormal Bases](./12_orthogonality_and_bases.md) | [🏠 Index](./README.md) | [Next: Part 14: Linear Regression Matrix Mathematics →](./14_linear_regression_matrix_math.md)
