> 📖 **Navigation:** [← Previous: Part 13: Vector Projections](./13_vector_projections.md) | [🏠 Index](./README.md) | [Next: Part 15: Gradients & Derivatives for Optimization →](./15_gradients_and_derivatives.md)

---

# PART 14 — LINEAR REGRESSION: MATRIX MATHEMATICS & NORMAL EQUATIONS

Linear Regression is the quintessential linear algebra model in machine learning. This module provides the complete algebraic and geometric derivation of the **Normal Equations**.

---

## 14.1 The Matrix System: $\mathbf{y} = X\mathbf{w} + \boldsymbol{\epsilon}$

For a dataset of $n$ samples and $d$ features (including the bias intercept column $\mathbf{1}$):

$$
\begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}
=
\begin{bmatrix}
1 & x_{11} & \dots & x_{1d} \\
1 & x_{21} & \dots & x_{2d} \\
\vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & \dots & x_{nd}
\end{bmatrix}
\begin{bmatrix} w_0 \\ w_1 \\ \vdots \\ w_d \end{bmatrix}
+
\begin{bmatrix} \epsilon_1 \\ \epsilon_2 \\ \vdots \\ \epsilon_n \end{bmatrix}
$$

* $\mathbf{y} \in \mathbb{R}^n$: Target vector
* $X \in \mathbb{R}^{n \times (d+1)}$: Design matrix
* $\mathbf{w} \in \mathbb{R}^{d+1}$: Parameter weights vector
* $\boldsymbol{\epsilon} \in \mathbb{R}^n$: Unobserved noise / error vector

---

## 14.2 Derivation of the Normal Equations via Matrix Calculus

We wish to find weights $\mathbf{w}$ that minimize the Ordinary Least Squares (OLS) loss function (Sum of Squared Residuals):

$$
\mathcal{L}(\mathbf{w}) = \|\mathbf{y} - X\mathbf{w}\|_2^2 = (\mathbf{y} - X\mathbf{w})^T (\mathbf{y} - X\mathbf{w})
$$

### Step 1: Expand the Matrix Transpose Product
$$
\mathcal{L}(\mathbf{w}) = (\mathbf{y}^T - \mathbf{w}^T X^T)(\mathbf{y} - X\mathbf{w}) = \mathbf{y}^T \mathbf{y} - \mathbf{y}^T X \mathbf{w} - \mathbf{w}^T X^T \mathbf{y} + \mathbf{w}^T X^T X \mathbf{w}
$$
Because $\mathbf{y}^T X \mathbf{w}$ is a scalar ($1 \times 1$), $(\mathbf{y}^T X \mathbf{w})^T = \mathbf{w}^T X^T \mathbf{y}$:
$$
\mathcal{L}(\mathbf{w}) = \mathbf{y}^T \mathbf{y} - 2 \mathbf{w}^T X^T \mathbf{y} + \mathbf{w}^T X^T X \mathbf{w}
$$

### Step 2: Differentiate with respect to $\mathbf{w}$ and Set to $\mathbf{0}$
Using matrix derivatives ($\nabla_{\mathbf{w}} (\mathbf{w}^T \mathbf{a}) = \mathbf{a}$ and $\nabla_{\mathbf{w}} (\mathbf{w}^T A \mathbf{w}) = 2A\mathbf{w}$):
$$
\nabla_{\mathbf{w}} \mathcal{L} = -2 X^T \mathbf{y} + 2 X^T X \mathbf{w} = \mathbf{0}
$$

### Step 3: Rearrange to the Normal Equations
$$
X^T X \mathbf{w} = X^T \mathbf{y}
$$

Assuming $X$ has full column rank ($\det(X^T X) \neq 0$):
$$
\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}
$$

---

## 14.3 Geometric Meaning: Orthogonality of Residuals ($X^T \mathbf{e} = \mathbf{0}$)

```
                                 y (Actual Target)
                                ╱│
                               ╱ │ Residual e = y - Xw
                              ╱  │ (Orthogonal to Col(X)!)
                             ╱   │
                            ●────┴────────► Col(X) (Feature Subspace)
                            0    ŷ = Xw = Py
```

Rearranging the normal equations gives:
$$
X^T (\mathbf{y} - X\mathbf{w}) = X^T \mathbf{e} = \mathbf{0}
$$

* **Geometric Insight:** The error vector $\mathbf{e}$ is **strictly orthogonal to every feature column in $X$**.
* **Zero Mean Residuals:** Since the first column of $X$ is $\mathbf{1}$, $\mathbf{1}^T \mathbf{e} = \sum_{i=1}^n e_i = 0$. **The sum (and mean) of OLS residuals is always exactly zero!**
* The prediction $\hat{\mathbf{y}} = X\mathbf{w} = X(X^T X)^{-1} X^T \mathbf{y} = P \mathbf{y}$ is the **orthogonal projection of $\mathbf{y}$ onto the column space $\text{Col}(X)$**.

---

## 14.4 Rank-Deficient Regression & Multicollinearity

What happens when $X^T X$ is singular ($\det(X^T X) = 0$)?
* If features are collinear (e.g. $x_2 = 2 x_1$), $X$ loses full column rank $\implies X^T X$ has no inverse!
* There exist **infinitely many weight vectors $\mathbf{w}$** that achieve the exact same minimum MSE.
* **The Two Fixes in Machine Learning:**
  1. **Moore-Penrose Pseudoinverse ($X^+$):** Picks the unique solution $\mathbf{w} = X^+ \mathbf{y}$ with the minimum Euclidean norm $\|\mathbf{w}\|_2$.
  2. **Ridge Regularization ($L_2$):** Adds $\lambda I$ to ensure $(X^T X + \lambda I)$ is strictly positive definite and invertible.

---

## Advanced / Optional — Do not study until Core track is complete

### A.1 The Hat Matrix ($H$) & Leverage Scores ($h_{ii}$)
The projection matrix $H = X(X^T X)^{-1} X^T$ satisfies $\hat{\mathbf{y}} = H\mathbf{y}$. Diagonal entry $h_{ii} = H_{ii}$ is the **leverage** of sample $i$. Samples with high leverage ($h_{ii} > \frac{2d}{n}$) exert disproportionate pull on the regression line (used in Cook's Distance outlier detection).

### A.2 The Gauss-Markov Theorem & BLUE Property
Under standard error assumptions ($\mathbb{E}[\boldsymbol{\epsilon}] = \mathbf{0}, \text{Cov}(\boldsymbol{\epsilon}) = \sigma^2 I$), OLS is **BLUE** (**Best Linear Unbiased Estimator**):
1. Unbiased: $\mathbb{E}[\hat{\mathbf{w}}] = \mathbf{w}$
2. Parameter Covariance: $\text{Cov}(\hat{\mathbf{w}}) = \sigma^2 (X^T X)^{-1}$
3. Minimum Variance: Any other unbiased estimator has a larger covariance matrix in the PSD sense.

---

> 📖 **Navigation:** [← Previous: Part 13: Vector Projections](./13_vector_projections.md) | [🏠 Index](./README.md) | [Next: Part 15: Gradients & Derivatives for Optimization →](./15_gradients_and_derivatives.md)
