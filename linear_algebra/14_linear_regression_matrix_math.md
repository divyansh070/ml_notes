> 📖 **Navigation:** [← Previous: Part 13: Vector Projections](./13_vector_projections.md) | [🏠 Index](./README.md) | [Next: Part 15: Gradients & Derivatives for Optimization →](./15_gradients_and_derivatives.md)

---

# PART 13 — LINEAR REGRESSION: MATRIX MATHEMATICS

---

## 13.1 The Matrix System: $y = Xw + \epsilon$

$$
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_N
\end{bmatrix} =
\begin{bmatrix}
1 & x_{11} & \dots & x_{1p} \\
1 & x_{21} & \dots & x_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
1 & x_{N1} & \dots & x_{Np}
\end{bmatrix}
\begin{bmatrix}
w_0 \\
w_1 \\
\vdots \\
w_p
\end{bmatrix}
+
\begin{bmatrix}
\epsilon_1 \\
\epsilon_2 \\
\vdots \\
\epsilon_N
\end{bmatrix}
$$

* Target vector: $\mathbf{y} \in \mathbb{R}^N$
* Design matrix: $X \in \mathbb{R}^{N \times (p+1)}$
* Weights vector: $\mathbf{w} \in \mathbb{R}^{p+1}$

---

## 13.2 Minimizing the Sum of Squared Residuals (Least Squares)

Residual error vector: $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} = \mathbf{y} - X\mathbf{w}$.

The Ordinary Least Squares (OLS) Loss function is the squared $L_2$ norm of the error vector:

$$
\mathcal{L}(\mathbf{w}) = \|\mathbf{y} - X\mathbf{w}\|_2^2 = (\mathbf{y} - X\mathbf{w})^T (\mathbf{y} - X\mathbf{w})
$$

Expand the matrix transpose product:

$$
\mathcal{L}(\mathbf{w}) = (\mathbf{y}^T - \mathbf{w}^T X^T)(\mathbf{y} - X\mathbf{w}) = \mathbf{y}^T \mathbf{y} - \mathbf{y}^T X \mathbf{w} - \mathbf{w}^T X^T \mathbf{y} + \mathbf{w}^T X^T X \mathbf{w}
$$

Since $\mathbf{y}^T X \mathbf{w}$ is a scalar, $(\mathbf{y}^T X \mathbf{w})^T = \mathbf{w}^T X^T \mathbf{y}$. Combining terms:

$$
\mathcal{L}(\mathbf{w}) = \mathbf{y}^T \mathbf{y} - 2\mathbf{w}^T X^T \mathbf{y} + \mathbf{w}^T X^T X \mathbf{w}
$$

---

## 13.3 Derivation of the Normal Equation ($X^T X w = X^T y$)

To find the minimum loss, take the matrix derivative with respect to $\mathbf{w}$ and set it to $\mathbf{0}$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = -2 X^T \mathbf{y} + 2 X^T X \mathbf{w} = \mathbf{0}
$$

Divide by 2 and rearrange:

$$
X^T X \mathbf{w} = X^T \mathbf{y}
$$

Assuming $X^T X$ is invertible, multiply both sides by $(X^T X)^{-1}$:

$$
\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}
$$

> [!IMPORTANT]
> **Key Conditions & Practical Reality:**
> 1. **Invertibility Assumption:** The closed-form expression $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$ strictly assumes that $X^T X$ is full rank and invertible ($\det(X^T X) \neq 0$).
> 2. **Numerical Solvers vs. Matrix Inversion:** The formula is essential for mathematical derivations and understanding OLS. In production numerical software (e.g. Scikit-Learn, LAPACK), explicitly computing $(X^T X)^{-1}$ is avoided due to $\mathcal{O}(p^3)$ cost and numerical instability. Instead, systems use **QR Decomposition** ($X = QR$), **SVD**, or **iterative Gradient Descent**.

---

## 13.4 Why $(X^T X)^{-1}$ Fails in Practice (Condition Number & QR Solvers)

1. **Computational Cost:** Matrix inversion of $(p \times p)$ scales as $\mathcal{O}(p^3)$. If $p = 50,000$ features, computing $(X^T X)^{-1}$ requires $\approx 1.25 \times 10^{14}$ operations.
2. **Ill-Conditioned / Multicollinearity:** If features are highly correlated, $(X^T X)$ has a condition number near $\infty$. Tiny floating-point rounding errors cause learned weights to blow up to massive unstable numbers ($w_1 = +10^6, w_2 = -10^6$).
3. **Industry Standard Alternatives:**
   * **QR Decomposition:** $X = QR \implies R \mathbf{w} = Q^T \mathbf{y}$ (Solves via fast back-substitution without explicit inversion).
   * **Gradient Descent:** Iteratively steps down the loss gradient in $\mathcal{O}(Np)$ per iteration.

---

> 📖 **Navigation:** [← Previous: Part 13: Vector Projections](./13_vector_projections.md) | [🏠 Index](./README.md) | [Next: Part 15: Gradients & Derivatives for Optimization →](./15_gradients_and_derivatives.md)
