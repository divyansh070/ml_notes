> 📖 **Navigation:** [← Previous: Part 08: Eigenvalues in Principal Component Analysis (PCA)](./08_eigenvalues_in_pca.md) | [🏠 Index](./README.md) | [Next: Part 10: Complete PCA Walkthrough from Scratch →](./10_pca_scratch_walkthrough.md)

---

# PART 8 — COVARIANCE MATRIX: COMPLETE HAND CALCULATION

---

## 8.1 Variance vs. Covariance Formulas

* **Variance (Spread of single variable $X$):**

$$
\text{Var}(X) = s_X^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2
$$

* **Covariance (Joint linear association between $X$ and $Y$):**

$$
\text{Cov}(X, Y) = s_{XY} = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})
$$

  * If $\text{Cov}(X,Y) \gt 0$: When $X$ increases, $Y$ tends to increase.
  * If $\text{Cov}(X,Y) \lt 0$: When $X$ increases, $Y$ tends to decrease.
  * If $\text{Cov}(X,Y) = 0$: **No linear association exists between $X$ and $Y$. (A non-linear relationship can still exist!)**

---

## 8.2 Complete Step-by-Step Hand Calculation

We have a tiny dataset of $n=3$ observations on two features ($X=$ Study Hours, $Y=$ Exam Score):
* $X = [1, 2, 3]$
* $Y = [2, 3, 7]$

---

### Step 1: Calculate Sample Means ($\bar{x}, \bar{y}$)

$$
\bar{x} = \frac{1 + 2 + 3}{3} = \frac{6}{3} = 2.0
$$

$$
\bar{y} = \frac{2 + 3 + 7}{3} = \frac{12}{3} = 4.0
$$

---

### Step 2: Build the Centered Deviation Table

| Sample $i$ | $x_i$ | $y_i$ | $(x_i - \bar{x})$ | $(y_i - \bar{y})$ | $(x_i - \bar{x})^2$ | $(y_i - \bar{y})^2$ | $(x_i - \bar{x})(y_i - \bar{y})$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1 | 2 | $1 - 2 = \mathbf{-1}$ | $2 - 4 = \mathbf{-2}$ | $(-1)^2 = \mathbf{1}$ | $(-2)^2 = \mathbf{4}$ | $(-1) \times (-2) = \mathbf{2}$ |
| **2** | 2 | 3 | $2 - 2 = \mathbf{0}$ | $3 - 4 = \mathbf{-1}$ | $(0)^2 = \mathbf{0}$ | $(-1)^2 = \mathbf{1}$ | $(0) \times (-1) = \mathbf{0}$ |
| **3** | 3 | 7 | $3 - 2 = \mathbf{1}$ | $7 - 4 = \mathbf{3}$ | $(1)^2 = \mathbf{1}$ | $(3)^2 = \mathbf{9}$ | $(1) \times (3) = \mathbf{3}$ |
| **SUM** | | | | | $\sum = \mathbf{2}$ | $\sum = \mathbf{14}$ | $\sum = \mathbf{5}$ |

---

### Step 3: Calculate Sample Variances & Covariance (Divide by $n-1 = 2$)

$$
\text{Var}(X) = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n - 1} = \frac{2}{2} = 1.0
$$

$$
\text{Var}(Y) = \frac{\sum_{i=1}^{n} (y_i - \bar{y})^2}{n - 1} = \frac{14}{2} = 7.0
$$

$$
\text{Cov}(X, Y) = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{n - 1} = \frac{5}{2} = 2.5
$$

---

### Step 4: Construct the 2x2 Covariance Matrix

$$
\Sigma =
\begin{bmatrix}
\text{Var}(X) & \text{Cov}(X,Y) \\
\text{Cov}(Y,X) & \text{Var}(Y)
\end{bmatrix} =
\begin{bmatrix}
1.0 & 2.5 \\
2.5 & 7.0
\end{bmatrix}
$$

---

## 8.3 The Symmetry & Quadratic Structure of $\Sigma$

* **Diagonal elements:** Always individual feature variances ($\ge 0$).
* **Off-diagonal elements:** Pairwise covariances. Because $\text{Cov}(X, Y) = \text{Cov}(Y, X)$, the covariance matrix is **always symmetric**:

$$
\Sigma = \Sigma^T
$$

* **Matrix Formulation:** If $X_c$ is the mean-centered data matrix ($(N \times p)$), then:

$$
\Sigma = \frac{1}{n-1} X_c^T X_c
$$

---

## 8.4 Sample ($n-1$) vs. Population ($N$) Bessel's Correction

* **Population Covariance (Divide by $N$):** Used when the dataset contains every single member of the population:

$$
\Sigma_{\text{pop}} = \frac{1}{N} X_c^T X_c \implies \Sigma_{\text{pop}} =
\begin{bmatrix}
2/3 & 5/3 \\
5/3 & 14/3
\end{bmatrix}
\approx
\begin{bmatrix}
0.67 & 1.67 \\
1.67 & 4.67
\end{bmatrix}
$$

* **Sample Covariance (Divide by $n-1$, Bessel's Correction):** Used when the dataset is a sample drawn from a larger population. Dividing by $n-1$ corrects for the fact that sample deviations around the sample mean $\bar{x}$ are systematically smaller than deviations around the true unknown population mean $\mu$, providing an **unbiased estimator**.

---

## 8.5 Mahalanobis Distance (Scale & Correlation Invariant Distance)

Standard Euclidean distance $d(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T (\mathbf{x} - \boldsymbol{\mu})}$ assumes features are uncorrelated and have identical unit variance (spherical contours).

When features have different scales and non-zero correlations, the **Mahalanobis Distance** measures the statistical distance of point $\mathbf{x}$ from distribution mean $\boldsymbol{\mu}$ scaled by the inverse covariance matrix $\Sigma^{-1}$:

$$
D_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})}
$$

```
       EUCLIDEAN DISTANCE (Circles)             MAHALANOBIS DISTANCE (Ellipses)
                 x2                                          x2
                 │                                           │        Point B (Far statistically!)
                 │      Point A                              │       ●
                 │     ●                                     │      ╱
                 │    ╱                                      │    ╭───╮
                 └───●────────► x1                           └───●┴───┴──────► x1
                    Mean                                        Mean    Point A (Near statistically!)
          (Both points look equidistant)                  (Accounts for covariance elongation!)
```

* **Role in ML:** Anomaly detection (multivariate outlier detection), Gaussian Discriminant Analysis (LDA/QDA classification boundaries), and exponent of Multivariate Normal PDF:

$$
p(\mathbf{x}) = \frac{1}{(2\pi)^{p/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} D_M^2(\mathbf{x}, \boldsymbol{\mu}) \right)
$$

---

## 8.6 Total Variance vs. Generalized Variance

* **Total Variance (Trace of $\Sigma$):** The sum of individual feature variances:

$$
\text{Total Variance} = \text{Tr}(\Sigma) = \sum_{i=1}^{p} \text{Var}(X_i) = \sum_{i=1}^{p} \lambda_i
$$

* **Generalized Variance (Determinant of $\Sigma$):** The multidimensional scatter volume spanned by the data cloud:

$$
\text{Generalized Variance} = \det(\Sigma) = \prod_{i=1}^{p} \lambda_i
$$

* If features are perfectly collinear, the scatter ellipse has zero thickness $\implies \det(\Sigma) = 0$, even though $\text{Tr}(\Sigma) > 0$.

---

> 📖 **Navigation:** [← Previous: Part 08: Eigenvalues in Principal Component Analysis (PCA)](./08_eigenvalues_in_pca.md) | [🏠 Index](./README.md) | [Next: Part 10: Complete PCA Walkthrough from Scratch →](./10_pca_scratch_walkthrough.md)
