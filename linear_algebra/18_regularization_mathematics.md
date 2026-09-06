> 📖 **Navigation:** [← Previous: Part 17: Distance & Similarity Metrics](./17_distance_and_similarity_metrics.md) | [🏠 Index](./README.md) | [Next: Part 19: Entropy & Information Gain Mathematics →](./19_entropy_and_information_gain.md)

---

# PART 18 — REGULARIZATION MATHEMATICS ($L_1$ vs. $L_2$)

Regularization prevents overfitting by penalizing large parameter weights, balancing the **bias-variance tradeoff**.

---

## 18.1 Constrained Optimization Formulation

$$
\min_{\mathbf{w}} \text{MSE}(\mathbf{w}) \quad \text{subject to} \quad \|\mathbf{w}\|_p \le C
$$

* **Ridge Regression ($L_2$ Penalty):** Subject to $w_1^2 + w_2^2 \le C$ (A smooth **circular/spherical ball**).
* **Lasso Regression ($L_1$ Penalty):** Subject to $|w_1| + |w_2| \le C$ (A **spiky diamond/polytope**).

---

## 18.2 Why $L_1$ (Lasso) Creates Exact Zeros (Sparsity)

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

1. **The Sharp Corner Geometry:** The $L_1$ diamond has sharp corners lying directly on the coordinate axes ($w_1 = 0$ or $w_2 = 0$). When the expanding MSE loss contours touch the constraint boundary, they almost always make contact at a corner, forcing weights to **exact 0.0** (automatic feature selection).
2. **Subgradient Calculus:** The gradient of $|w|$ is a constant $\pm \lambda$ regardless of how small $w$ is, driving weights all the way to zero. In $L_2$, the gradient $\lambda w$ shrinks as $w \to 0$, producing asymptotic decay without exact zeros.

---

## 18.3 Ridge Regression ($L_2$) & Why Adding $\lambda I$ Fixes Singularity

The closed-form Ridge solution is:

$$
\mathbf{w}_{\text{Ridge}} = (X^T X + \lambda I)^{-1} X^T \mathbf{y}
$$

* **Why Adding $\lambda I$ Guarantees Invertibility:**
  * Even if $X^T X$ is singular (rank-deficient or $d > n$), its eigenvalues satisfy $\lambda_i(X^T X) \ge 0$.
  * Adding $\lambda I$ shifts all eigenvalues by $+\lambda$: $\lambda_i(X^T X + \lambda I) = \lambda_i + \lambda \ge \lambda > 0$.
  * Because all eigenvalues are strictly positive ($> 0$), $(X^T X + \lambda I)$ is **strictly Positive Definite and ALWAYS invertible!**

---

## 18.4 ElasticNet Regularization ($L_1 + L_2$)

$$
\mathcal{L}_{\text{ElasticNet}} = \text{MSE}(\mathbf{w}) + \lambda_1 \|\mathbf{w}\|_1 + \frac{\lambda_2}{2} \|\mathbf{w}\|_2^2
$$

* **Grouping Effect:** When two features are strongly correlated, Lasso arbitrarily picks one and zeroes the other. ElasticNet groups them, selecting both features together while maintaining sparsity.

---

## Advanced / Optional — Do not study until Core track is complete

### A.1 SVD / Spectral Shrinkage View of Ridge
Using SVD $X = U \Sigma V^T$:
$$
\mathbf{w}_{\text{Ridge}} = \sum_{i=1}^{d} \left( \frac{\sigma_i}{\sigma_i^2 + \lambda} \right) (\mathbf{u}_i^T \mathbf{y}) \mathbf{v}_i
$$
*(For noisy directions with small singular values $\sigma_i \ll \lambda$, Ridge damps the filter weight to $\approx \frac{\sigma_i}{\lambda} \to 0$, preventing variance explosion).*

---

> 📖 **Navigation:** [← Previous: Part 17: Distance & Similarity Metrics](./17_distance_and_similarity_metrics.md) | [🏠 Index](./README.md) | [Next: Part 19: Entropy & Information Gain Mathematics →](./19_entropy_and_information_gain.md)
