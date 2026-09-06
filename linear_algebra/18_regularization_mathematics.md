> 📖 **Navigation:** [← Previous: Part 17: Distance & Similarity Metrics](./17_distance_and_similarity_metrics.md) | [🏠 Index](./README.md) | [Next: Part 19: Entropy & Information Gain Mathematics →](./19_entropy_and_information_gain.md)

---

# PART 17 — REGULARIZATION MATHEMATICS (L1 vs. L2)

---

## 17.1 The Constrained Optimization Formulation

$$
\min_{\mathbf{w}} \text{MSE}(\mathbf{w}) \quad \text{subject to} \quad \|\mathbf{w}\|_p \le C
$$

* **Ridge Regression ($L_2$):** Subject to $w_1^2 + w_2^2 \le C$ (A smooth **circular/spherical ball**).
* **Lasso Regression ($L_1$):** Subject to $|w_1| + |w_2| \le C$ (A **spiky diamond/polytope**).

---

## 17.2 Mathematical Geometry: Why L1 Creates Exact Zeros (Sparsity)

```
        L2 REGULARIZATION (Circle)                  L1 REGULARIZATION (Diamond)
                   w2                                          w2
                   │   * OLS Minimum                           │   * OLS Minimum
                ╭──┼──╮                                       ╱│╲
               │   │   │                                     ╱ │ ╲
           ────┼───●───┼──── w1                          ────●──┼──●──── w1  (Hits sharp corner
               │   │   │                                     ╲ │ ╲    on w1 axis -> w2 = 0!)
                ╰──┼──╯                                       ╲│╱
          (Contact occurs at smooth edge              (Contact occurs at sharp corner
           -> w1 and w2 are small floats)              -> one weight is set to EXACT 0.0)
```

1. **The Elliptical Contours:** The MSE loss contours expand outward from the unconstrained OLS minimum.
2. **The Geometric Contact Point:**
   * **$L_1$ Diamond:** The diamond has sharp, pointy vertices lying directly on the coordinate axes ($w_1=0$ or $w_2=0$). When the growing MSE ellipses hit the constraint, they almost always touch a sharp corner first. This sets the other weight to **exact 0.0**, performing **automatic feature selection**.
   * **$L_2$ Ball:** The circle is uniformly smooth with no corners. The elliptical contours touch the circle along smooth boundaries where neither weight is exactly zero, shrinking weights asymptotically toward zero.

---

## 17.3 Mathematical Geometry: Why L2 Shrinks Smoothly (Weight Decay)

Loss function:

$$
\mathcal{L}_{\text{Ridge}} = \text{MSE} + \frac{\lambda}{2} \sum_{j=1}^{p} w_j^2
$$

Gradient update step:

$$
w_{j}^{(t+1)} = w_j^{(t)} - \alpha \left(\frac{\partial \text{MSE}}{\partial w_j} + \lambda w_j\right) = (1 - \alpha \lambda) w_j^{(t)} - \alpha \frac{\partial \text{MSE}}{\partial w_j}
$$

* Since $(1 - \alpha \lambda) \lt 1$, the weights are multiplied by a decay factor less than 1 at every single step before taking the gradient step! This is why $L_2$ is called **Weight Decay**.

---

## 17.4 Spectral / SVD View of Ridge Shrinkage

Expressing Ridge Regression through the SVD of the centered design matrix $X = U \Sigma V^T$ reveals exactly how $L_2$ regularization handles ill-conditioned data:

$$
\mathbf{w}_{\text{Ridge}} = (X^T X + \lambda I)^{-1} X^T \mathbf{y} = \sum_{i=1}^{p} \left( \frac{\sigma_i}{\sigma_i^2 + \lambda} \right) (\mathbf{u}_i^T \mathbf{y}) \mathbf{v}_i
$$

Compare this directly with unregularized OLS:

$$
\mathbf{w}_{\text{OLS}} = (X^T X)^{-1} X^T \mathbf{y} = \sum_{i=1}^{p} \left( \frac{1}{\sigma_i} \right) (\mathbf{u}_i^T \mathbf{y}) \mathbf{v}_i
$$

```
                   OLS vs. RIDGE COEFFICIENT SHRINKAGE FILTER
       Singular Value σ_i           OLS Filter (1 / σ_i)     Ridge Filter (σ_i / (σ_i^2 + λ))
    ──────────────────────────────────────────────────────────────────────────────────────────
    Large Signal (σ_i = 100, λ=1)       0.010                     100 / 10001 ≈ 0.010 (Preserved!)
    Small Noise  (σ_i = 0.01, λ=1)    100.000 (Explodes!)         0.01 / 1.0001 ≈ 0.010 (DAMPED!)
```

* **The Takeaway:** For large singular values ($\sigma_i^2 \gg \lambda$), Ridge acts like OLS. For small singular values ($\sigma_i^2 \ll \lambda$, corresponding to collinear noise directions), Ridge shrinks the filter weight toward zero, preventing variance explosion!

---

## 17.5 Lasso Subgradient Analysis & Soft-Thresholding Operator

Because the $L_1$ penalty $|w_j|$ is non-differentiable at $w_j = 0$, we use **Subgradient Calculus**.
The subdifferential of absolute value $|w|$ is:

$$
\partial |w| =
\begin{cases}
\{+1\} & \text{if } w > 0 \\
[-1, +1] & \text{if } w = 0 \\
\{-1\} & \text{if } w < 0
\end{cases}
$$

For an orthonormal feature matrix ($X^T X = I$), the closed-form Lasso solution is the **Soft-Thresholding Operator**:

$$
\hat{w}_j^{\text{Lasso}} = S_\lambda(\hat{w}_j^{\text{OLS}}) = \text{sign}(\hat{w}_j^{\text{OLS}}) \cdot \max\left( |\hat{w}_j^{\text{OLS}}| - \lambda, 0 \right)
$$

```
             HARD THRESHOLDING (L0) vs. SOFT THRESHOLDING (L1 - Lasso)
          w_hat                                    w_hat
            ▲      /                                 ▲        /
            │     /                                  │       /
            │    /                                   │      /
         ───┼───●───► w_ols                       ───┼─────●────► w_ols
           /│                                       /│    -λ  +λ
          / │                                      / │
      Hard jump at 0.0                    Continuous soft shrinkage to 0.0!
```

---

## 17.6 ElasticNet Regularization ($L_1 + L_2$)

When features are highly correlated (e.g. gene expressions, financial indicators), Lasso arbitrarily picks one feature and zeroes the others. **ElasticNet** combines both penalties:

$$
\mathcal{L}_{\text{ElasticNet}} = \text{MSE}(\mathbf{w}) + \lambda_1 \|\mathbf{w}\|_1 + \frac{\lambda_2}{2} \|\mathbf{w}\|_2^2
$$

* **Grouping Effect:** ElasticNet encourages grouped selection—correlated features enter or leave the model together.
* **Strict Convexity:** Adding the strictly convex $L_2$ penalty guarantees a unique global minimum even when $p > N$.

---

> 📖 **Navigation:** [← Previous: Part 17: Distance & Similarity Metrics](./17_distance_and_similarity_metrics.md) | [🏠 Index](./README.md) | [Next: Part 19: Entropy & Information Gain Mathematics →](./19_entropy_and_information_gain.md)
