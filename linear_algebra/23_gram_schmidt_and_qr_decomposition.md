> 📖 **Navigation:** [← Previous: Part 22: Four Fundamental Subspaces (Strang's Big Picture)](./22_four_fundamental_subspaces.md) | [🏠 Index](./README.md) | [Next: Part 24: Positive Definite & Positive Semidefinite Matrices →](./24_positive_definite_matrices.md)

---

# PART 23 — GRAM-SCHMIDT ORTHOGONALIZATION & QR DECOMPOSITION

---

## 23.1 Why We Need Orthogonal Bases

When basis vectors are mutually orthogonal ($\mathbf{q}_i \cdot \mathbf{q}_j = 0 \text{ for } i \neq j$) and normalized ($\|\mathbf{q}_i\|_2 = 1$):
1. **No Cross-Talk:** Projecting a vector onto coordinate axis $i$ is completely independent of coordinate axis $j$:

$$
c_i = \mathbf{x} \cdot \mathbf{q}_i
$$

2. **Trivial Matrix Inversion:** Any orthogonal matrix satisfies:

$$
Q^T Q = I \implies Q^{-1} = Q^T
$$

3. **Numerical Precision:** Orthogonal transformations preserve $L_2$ lengths and do not amplify floating-point rounding errors.

---

## 23.2 Gram-Schmidt Derivation

The **Gram-Schmidt Process** converts a set of linearly independent vectors $\mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_k$ into an orthonormal set $\mathbf{q}_1, \mathbf{q}_2, \dots, \mathbf{q}_k$.

```
        VECTOR 1: Set u1 = a1                VECTOR 2: Subtract projection onto u1
                  y                                           y
                  │     a1 = u1                               │     a2
                  │    ╱                                      │    ╱│
                1 ┼───●                                     1 ┼───● │  u2 = a2 - proj_u1(a2)
                  │  ╱                                        │   │ └──► (Orthogonal to u1!)
                  └──┴─────► x                                └───┴────► x
                     1                                            1
```

### Complete Step-by-Step Hand Calculation Example
Orthonormalize the two vectors:

$$
\mathbf{a}_1 =
\begin{bmatrix}
1 \\
1
\end{bmatrix},
\quad
\mathbf{a}_2 =
\begin{bmatrix}
1 \\
0
\end{bmatrix}
$$

---

### Step 1: Set the First Orthogonal Vector $\mathbf{u}_1$

$$
\mathbf{u}_1 = \mathbf{a}_1 =
\begin{bmatrix}
1 \\
1
\end{bmatrix}
$$

---

### Step 2: Compute the Second Orthogonal Vector $\mathbf{u}_2$
Subtract the projection of $\mathbf{a}_2$ onto $\mathbf{u}_1$:

$$
\mathbf{u}_2 = \mathbf{a}_2 - \text{proj}_{\mathbf{u}_1}(\mathbf{a}_2) = \mathbf{a}_2 - \left(\frac{\mathbf{a}_2 \cdot \mathbf{u}_1}{\mathbf{u}_1 \cdot \mathbf{u}_1}\right) \mathbf{u}_1
$$

1. Dot products:
   * $\mathbf{a}_2 \cdot \mathbf{u}_1 = (1)(1) + (0)(1) = 1$.
   * $\mathbf{u}_1 \cdot \mathbf{u}_1 = 1^2 + 1^2 = 2$.
2. Projection fraction: $\frac{\mathbf{a}_2 \cdot \mathbf{u}_1}{\mathbf{u}_1 \cdot \mathbf{u}_1} = \frac{1}{2}$.
3. Subtract:

$$
\mathbf{u}_2 =
\begin{bmatrix}
1 \\
0
\end{bmatrix} - \frac{1}{2}
\begin{bmatrix}
1 \\
1
\end{bmatrix} =
\begin{bmatrix}
1 - 1/2 \\
0 - 1/2
\end{bmatrix} =
\begin{bmatrix}
1/2 \\
-1/2
\end{bmatrix}
$$

4. **Orthogonality Check:**
   $\mathbf{u}_1 \cdot \mathbf{u}_2 = (1)(1/2) + (1)(-1/2) = 1/2 - 1/2 = 0 \quad \checkmark$.

---

### Step 3: Normalize Both Vectors to Unit Length ($L_2 = 1$)
1. Normalize $\mathbf{u}_1$:
   * $\|\mathbf{u}_1\|_2 = \sqrt{1^2 + 1^2} = \sqrt{2}$.

$$
\mathbf{q}_1 = \frac{\mathbf{u}_1}{\|\mathbf{u}_1\|_2} =
\begin{bmatrix}
1/\sqrt{2} \\
1/\sqrt{2}
\end{bmatrix}
$$

2. Normalize $\mathbf{u}_2$:
   * $\|\mathbf{u}_2\|_2 = \sqrt{(1/2)^2 + (-1/2)^2} = \sqrt{1/4 + 1/4} = \sqrt{2/4} = \frac{1}{\sqrt{2}}$.

$$
\mathbf{q}_2 = \frac{\mathbf{u}_2}{\|\mathbf{u}_2\|_2} = \sqrt{2}
\begin{bmatrix}
1/2 \\
-1/2
\end{bmatrix} =
\begin{bmatrix}
1/\sqrt{2} \\
-1/\sqrt{2}
\end{bmatrix}
$$

---

## 23.3 Build Q and R ($A = QR$)

Any matrix $A$ with linearly independent columns can be factored into:

$$
A = QR
$$

Where:
* $Q$: An **orthogonal matrix** ($Q^T Q = I$) containing the orthonormal basis vectors as columns.
* $R$: An **upper-triangular matrix** ($R_{ij} = \mathbf{q}_i^T \mathbf{a}_j$, with $R_{ij} = 0$ for $i \gt j$) storing the projection coefficients.

### Constructing $Q$ and $R$ from our Hand Example:
1. Form $Q$:

$$
Q =
\begin{bmatrix}
\mathbf{q}_1 & \mathbf{q}_2
\end{bmatrix} =
\begin{bmatrix}
1/\sqrt{2} & 1/\sqrt{2} \\
1/\sqrt{2} & -1/\sqrt{2}
\end{bmatrix}
$$

2. Compute entries of $R = Q^T A$:
   * $R_{11} = \mathbf{q}_1 \cdot \mathbf{a}_1 = (1/\sqrt{2})(1) + (1/\sqrt{2})(1) = 2/\sqrt{2} = \sqrt{2}$.
   * $R_{12} = \mathbf{q}_1 \cdot \mathbf{a}_2 = (1/\sqrt{2})(1) + (1/\sqrt{2})(0) = 1/\sqrt{2}$.
   * $R_{21} = \mathbf{q}_2 \cdot \mathbf{a}_1 = (1/\sqrt{2})(1) + (-1/\sqrt{2})(1) = 0$.
   * $R_{22} = \mathbf{q}_2 \cdot \mathbf{a}_2 = (1/\sqrt{2})(1) + (-1/\sqrt{2})(0) = 1/\sqrt{2}$.

$$
R =
\begin{bmatrix}
\sqrt{2} & 1/\sqrt{2} \\
0 & 1/\sqrt{2}
\end{bmatrix}
$$

3. **Verification** ($A = QR$):

$$
QR =
\begin{bmatrix}
1/\sqrt{2} & 1/\sqrt{2} \\
1/\sqrt{2} & -1/\sqrt{2}
\end{bmatrix}
\begin{bmatrix}
\sqrt{2} & 1/\sqrt{2} \\
0 & 1/\sqrt{2}
\end{bmatrix} =
\begin{bmatrix}
1 + 0 & 1/2 + 1/2 \\
1 + 0 & 1/2 - 1/2
\end{bmatrix} =
\begin{bmatrix}
1 & 1 \\
1 & 0
\end{bmatrix} = A \quad \checkmark
$$

---

## 23.4 Why QR Matters in ML

In Ordinary Least Squares, the Normal Equation is $X^T X \mathbf{w} = X^T \mathbf{y}$.
1. Substitute $X = QR$:

$$
(QR)^T (QR) \mathbf{w} = (QR)^T \mathbf{y} \implies R^T (Q^T Q) R \mathbf{w} = R^T Q^T \mathbf{y}
$$

2. Since $Q^T Q = I$:

$$
R^T R \mathbf{w} = R^T Q^T \mathbf{y} \implies R \mathbf{w} = Q^T \mathbf{y}
$$

3. **Why this is superior to** $(X^T X)^{-1}$:
   * Because $R$ is **upper-triangular**, $R\mathbf{w} = Q^T \mathbf{y}$ is solved instantly using **back-substitution** without inverting any matrices!
   * Forming $X^T X$ squares the condition number: $\kappa(X^T X) = \kappa(X)^2$. If $\kappa(X) = 10^4$, then $\kappa(X^T X) = 10^8$, ruining numerical accuracy. QR solves least squares with condition number $\kappa(X)$, providing maximum numerical stability.

> [!TIP]
> **Common Interview Question:** *"Why do production machine learning libraries use QR decomposition to solve Ordinary Least Squares rather than inverting $X^T X$ directly?"*
> **Answer:** Explicitly forming $X^T X$ squares the condition number ($\kappa(X^T X) = \kappa(X)^2$), which magnifies floating-point roundoff errors and can cause numerical singularity. QR decomposition solves $R\mathbf{w} = Q^T \mathbf{y}$ via simple back-substitution with condition number $\kappa(X)$, providing maximum numerical stability without computing any matrix inverse.

> [!WARNING]
> **Common Mistake:** Forgetting that Gram-Schmidt produces **orthogonal** vectors ($\mathbf{u}_k$) that must be individually normalized by dividing by their $L_2$ norm ($\mathbf{q}_k = \mathbf{u}_k / \|\mathbf{u}_k\|_2$) to form an **orthonormal** matrix $Q$ where $Q^T Q = I$.

---

> 📖 **Navigation:** [← Previous: Part 22: Four Fundamental Subspaces (Strang's Big Picture)](./22_four_fundamental_subspaces.md) | [🏠 Index](./README.md) | [Next: Part 24: Positive Definite & Positive Semidefinite Matrices →](./24_positive_definite_matrices.md)
