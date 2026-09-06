> 📖 **Navigation:** [← Previous: Part 09: Covariance Matrix (Complete Hand Calculation)](./09_covariance_matrix.md) | [🏠 Index](./README.md) | [Next: Part 11: Singular Value Decomposition (SVD) →](./11_singular_value_decomposition.md)

---

# PART 9 — COMPLETE PCA WALKTHROUGH FROM SCRATCH

---

## 9.1 The 6-Step End-to-End PCA Algorithm

```
  1. Mean Center Data  ──►  2. Compute Covariance  ──►  3. Solve Eigenvalues
                                                                  │
  6. Reduced Dataset   ◄──  5. Project Data Points ◄──  4. Select Top Eigenvectors
```

---

## 9.2 Step-by-Step Numerical Hand Walkthrough (2D to 1D)

We have a 2D dataset of 3 points lying along the line $y = x$:

$$
X =
\begin{bmatrix}
1 & 1 \\
2 & 2 \\
3 & 3
\end{bmatrix}
$$

---

### Step 1: Center the Data ($\bar{x}=2, \bar{y}=2$)

$$
X_c = X - \mu =
\begin{bmatrix}
1 - 2 & 1 - 2 \\
2 - 2 & 2 - 2 \\
3 - 2 & 3 - 2
\end{bmatrix} =
\begin{bmatrix}
-1 & -1 \\
0 & 0 \\
1 & 1
\end{bmatrix}
$$

---

### Step 2: Compute Sample Covariance Matrix $\Sigma = \frac{1}{n-1} X_c^T X_c$

$$
X_c^T X_c =
\begin{bmatrix}
-1 & 0 & 1 \\
-1 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
-1 & -1 \\
0 & 0 \\
1 & 1
\end{bmatrix} =
\begin{bmatrix}
(-1)^2 + 0 + 1^2 & (-1)(-1) + 0 + (1)(1) \\
(-1)(-1) + 0 + (1)(1) & (-1)^2 + 0 + 1^2
\end{bmatrix} =
\begin{bmatrix}
2 & 2 \\
2 & 2
\end{bmatrix}
$$

$$
\Sigma = \frac{1}{3 - 1}
\begin{bmatrix}
2 & 2 \\
2 & 2
\end{bmatrix} =
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}
$$

---

### Step 3: Compute Eigenvalues of $\Sigma$

$$
\det(\Sigma - \lambda I) = \det
\begin{bmatrix}
1 - \lambda & 1 \\
1 & 1 - \lambda
\end{bmatrix}
= (1 - \lambda)^2 - 1 = \lambda^2 - 2\lambda = \lambda(\lambda - 2) = 0
$$

$$
\lambda_1 = 2, \quad \lambda_2 = 0
$$

---

### Step 4: Compute Top Eigenvector ($\mathbf{u}_1$ for $\lambda_1 = 2$)

$$
\begin{bmatrix}
1 - 2 & 1 \\
1 & 1 - 2
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix} =
\begin{bmatrix}
0 \\
0
\end{bmatrix}
\implies -v_1 + v_2 = 0 \implies v_1 = v_2
$$

Unit Eigenvector:

$$
\mathbf{u}_1 =
\begin{bmatrix}
1/\sqrt{2} \\
1/\sqrt{2}
\end{bmatrix}
$$

---

### Step 5: Project Centered Data onto Principal Component $\mathbf{u}_1$

$$
Z = X_c \mathbf{u}_1 =
\begin{bmatrix}
-1 & -1 \\
0 & 0 \\
1 & 1
\end{bmatrix}
\begin{bmatrix}
1/\sqrt{2} \\
1/\sqrt{2}
\end{bmatrix} =
\begin{bmatrix}
-1(1/\sqrt{2}) - 1(1/\sqrt{2}) \\
0(1/\sqrt{2}) + 0(1/\sqrt{2}) \\
1(1/\sqrt{2}) + 1(1/\sqrt{2})
\end{bmatrix} =
\begin{bmatrix}
-2/\sqrt{2} \\
0 \\
2/\sqrt{2}
\end{bmatrix} =
\begin{bmatrix}
-\sqrt{2} \\
0 \\
\sqrt{2}
\end{bmatrix}
\approx
\begin{bmatrix}
-1.414 \\
0.000 \\
1.414
\end{bmatrix}
$$

---

## 9.3 Calculating Explained Variance Ratio

$$
\text{Explained Variance Ratio} = \frac{\lambda_k}{\sum \lambda_i}
$$

* For Component 1: $\frac{\lambda_1}{\lambda_1 + \lambda_2} = \frac{2}{2 + 0} = \frac{2}{2} = 1.0 \implies$ **100%**.
* **The Takeaway:** By compressing from 2D coordinates $[x, y]$ down to a 1D scalar $z$, we **retained 100% of the variance in this particular dataset** while eliminating 1 redundant feature axis.

---

> 📖 **Navigation:** [← Previous: Part 09: Covariance Matrix (Complete Hand Calculation)](./09_covariance_matrix.md) | [🏠 Index](./README.md) | [Next: Part 11: Singular Value Decomposition (SVD) →](./11_singular_value_decomposition.md)
