> 📖 **Navigation:** [← Previous: Part 13: Vector Projections & Projection Matrices](./13_vector_projections.md) | [🏠 Index](./README.md) | [Next: Part 15: Eigenvalues & Eigenvectors →](./15_eigenvalues_and_eigenvectors.md)

---

# PART 14 — LEAST SQUARES & LINEAR REGRESSION

Linear regression is the foundational predictive model in data science. When an observed target vector cannot be reached by a linear system, **Ordinary Least Squares (OLS)** finds the provably optimal solution by projecting the target onto the feature subspace.

---

## 14.1 Why $A\mathbf{x} = \mathbf{b}$ Has No Exact Solution in Real Data

In real-world machine learning, we collect $m$ samples (often thousands) across $n$ features ($m \gg n$). The design matrix $X \in \mathbb{R}^{m \times n}$ has far more equations than parameters:

$$
X\mathbf{w} = \mathbf{y}
$$

Because observed targets contain noise and non-linearities, the vector $\mathbf{y}$ almost never lies within the Column Space $\text{Col}(X)$. The system is **inconsistent**—no exact parameter vector $\mathbf{w}$ exists.

Instead of seeking an impossible exact solution, we solve the **Least-Squares Problem**:

$$
\min_{\mathbf{w}} \|\mathbf{y} - X\mathbf{w}\|_2^2
$$

---

## 14.2 The Geometric Derivation: Residual Orthogonality

```
                                 y (Actual Target)
                                ╱│
                               ╱ │ Residual e = y - Xw
                              ╱  │ (Orthogonal to Col(X)!)
                             ╱   │
                            ●────┴────────► Col(X) (Feature Subspace)
                            0    ŷ = Xw = Py
```

1. We want the predicted point $\hat{\mathbf{y}} = X\mathbf{w} \in \text{Col}(X)$ that is closest to $\mathbf{y}$.
2. By the Closest Point Theorem, the shortest distance occurs when the residual error vector $\mathbf{e} = \mathbf{y} - X\mathbf{w}$ is **strictly perpendicular (orthogonal)** to the feature column space $\text{Col}(X)$.
3. This requires that $\mathbf{e}$ be perpendicular to **every column of $X$**:

$$
X^T \mathbf{e} = \mathbf{0}
$$

4. Substitute $\mathbf{e} = \mathbf{y} - X\mathbf{w}$:

$$
X^T (\mathbf{y} - X\mathbf{w}) = \mathbf{0} \implies X^T \mathbf{y} - X^T X \mathbf{w} = \mathbf{0}
$$

5. Rearranging gives the **Normal Equations**:

$$
X^T X \mathbf{w} = X^T \mathbf{y}
$$

6. If $X$ has full column rank ($\text{rank}(X) = n$), the $n \times n$ matrix $X^T X$ is invertible:

$$
\mathbf{w}_{\text{OLS}} = (X^T X)^{-1} X^T \mathbf{y}
$$

---

## 14.3 Concise Calculus Derivation (Loss Minimization)

The Ordinary Least Squares loss function is the sum of squared errors:

$$
\mathcal{L}(\mathbf{w}) = \|\mathbf{y} - X\mathbf{w}\|_2^2 = (\mathbf{y} - X\mathbf{w})^T (\mathbf{y} - X\mathbf{w}) = \mathbf{y}^T \mathbf{y} - 2 \mathbf{w}^T X^T \mathbf{y} + \mathbf{w}^T X^T X \mathbf{w}
$$

Differentiate with respect to parameter vector $\mathbf{w}$ and set to zero:

$$
\nabla_{\mathbf{w}} \mathcal{L} = -2 X^T \mathbf{y} + 2 X^T X \mathbf{w} = \mathbf{0} \implies X^T X \mathbf{w} = X^T \mathbf{y}
$$

*(Both geometric projection and matrix calculus produce the identical normal equations!).*

---

## 14.4 Residual Properties & The Intercept Column

1. **Orthogonality to Features:** $X^T \mathbf{e} = \mathbf{0}$. Residual errors are completely uncorrelated with any feature.
2. **Mean Residual is Zero:** When an intercept column $\mathbf{1}$ is included in $X$ (the first column is all 1s):

$$
\mathbf{1}^T \mathbf{e} = \sum_{i=1}^{m} e_i = 0
$$

   **The sum and mean of OLS residuals is always exactly zero.**

---

## 14.5 Complete Worked Numerical Example

Fit a linear regression model $y = w_0 + w_1 x$ on 3 data points: $(1, 2), (2, 2), (3, 4)$.

### Step 1: Set up $X$ and $\mathbf{y}$

$$
X = \begin{bmatrix}
1 & 1 \\
1 & 2 \\
1 & 3
\end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} 2 \\ 2 \\ 4 \end{bmatrix}
$$

### Step 2: Compute $X^T X$ and $X^T \mathbf{y}$

$$
X^T X = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix} =
\begin{bmatrix} 1+1+1 & 1+2+3 \\ 1+2+3 & 1+4+9 \end{bmatrix} = \begin{bmatrix} 3 & 6 \\ 6 & 14 \end{bmatrix}
$$

$$
X^T \mathbf{y} = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 2 \\ 2 \\ 4 \end{bmatrix} =
\begin{bmatrix} 2 + 2 + 4 \\ 2 + 4 + 12 \end{bmatrix} = \begin{bmatrix} 8 \\ 18 \end{bmatrix}
$$

### Step 3: Invert $X^T X$ using the $2 \times 2$ formula

$$
\det(X^T X) = (3)(14) - (6)(6) = 42 - 36 = 6
$$

$$
(X^T X)^{-1} = \frac{1}{6} \begin{bmatrix} 14 & -6 \\ -6 & 3 \end{bmatrix} = \begin{bmatrix} 7/3 & -1 \\ -1 & 1/2 \end{bmatrix}
$$

### Step 4: Compute Weights $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$

$$
\mathbf{w} = \frac{1}{6} \begin{bmatrix} 14 & -6 \\ -6 & 3 \end{bmatrix} \begin{bmatrix} 8 \\ 18 \end{bmatrix} =
\frac{1}{6} \begin{bmatrix} 14(8) - 6(18) \\ -6(8) + 3(18) \end{bmatrix} =
\frac{1}{6} \begin{bmatrix} 112 - 108 \\ -48 + 54 \end{bmatrix} =
\frac{1}{6} \begin{bmatrix} 4 \\ 6 \end{bmatrix} = \begin{bmatrix} 2/3 \\ 1 \end{bmatrix} \approx \begin{bmatrix} 0.67 \\ 1.00 \end{bmatrix}
$$

Model equation: $\hat{y} = 0.67 + 1.0 x$.

### Step 5: Verify Predictions and Residuals
* Predictions:

$$
\hat{\mathbf{y}} = X\mathbf{w} = \begin{bmatrix} 2/3 + 1(1) \\ 2/3 + 1(2) \\ 2/3 + 1(3) \end{bmatrix} = \begin{bmatrix} 5/3 \\ 8/3 \\ 11/3 \end{bmatrix}
$$$
* Residuals:

$$

\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} = \begin{bmatrix} 2 - 5/3 \\ 2 - 8/3 \\ 4 - 11/3 \end{bmatrix} = \begin{bmatrix} 1/3 \\ -2/3 \\ 1/3 \end{bmatrix}
$$$
* Check sum of residuals: $1/3 - 2/3 + 1/3 = 0$ ✓
* Check feature orthogonality:

$$
X^T \mathbf{e} = \begin{bmatrix} 1/3 - 2/3 + 1/3 \\ 1(1/3) + 2(-2/3) + 3(1/3) \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$ ✓

---

## 14.6 Multicollinearity & Rank Deficiency: When $(X^T X)^{-1}$ Fails

What happens if feature columns are collinear (e.g. `x2 = 2 * x1`)?
* $\text{rank}(X) < n \implies \det(X^T X) = 0$. $X^T X$ cannot be inverted!
* **The Two Fixes in Machine Learning:**
  1. **Ridge Regression ($L_2$):** Adds $\lambda I$ to shift eigenvalues:

$$

\mathbf{w}_{\text{Ridge}} = (X^T X + \lambda I)^{-1} X^T \mathbf{y}

$$

     Because all eigenvalues of $(X^T X + \lambda I)$ are strictly $\ge \lambda > 0$, it is **always positive definite and invertible**.
  2. **Pseudoinverse ($X^+$):** Uses SVD to select the unique minimum-norm solution $\mathbf{w} = X^+ \mathbf{y}$.

---

## 14.7 Why this matters in ML

1. **Analytical Baseline:** OLS provides the closed-form benchmark for regression models before gradient descent is needed.
2. **Computational Solvers (QR vs Normal Equations):** Production libraries like `scipy.linalg.lstsq` do NOT form $X^T X$ because forming $X^T X$ squares the condition number ($\kappa(X^T X) = (\kappa(X))^2$). Instead, they factor $X = Q R$ and solve the upper-triangular system $R\mathbf{w} = Q^T \mathbf{y}$ with machine-precision stability.

---

## 14.8 Common mistakes

* **Inverting When $m < n$ (More Features Than Samples):** If you have 100 features and only 50 samples, $\text{rank}(X) \le 50 < 100$. $X^T X$ is guaranteed to be singular. OLS will crash without regularization.
* **Confusing Residuals with Errors:** The unobserved true noise is $\boldsymbol{\epsilon} = \mathbf{y} - X\mathbf{w}^*$. The observed sample residual is $\mathbf{e} = \mathbf{y} - X\hat{\mathbf{w}}$. Only the sample residuals $\mathbf{e}$ are guaranteed to sum to zero and be orthogonal to $X$.

---

> 📖 **Navigation:** [← Previous: Part 13: Vector Projections & Projection Matrices](./13_vector_projections.md) | [🏠 Index](./README.md) | [Next: Part 15: Eigenvalues & Eigenvectors →](./15_eigenvalues_and_eigenvectors.md)
