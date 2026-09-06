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

> 📖 **Navigation:** [← Previous: Part 17: Distance & Similarity Metrics](./17_distance_and_similarity_metrics.md) | [🏠 Index](./README.md) | [Next: Part 19: Entropy & Information Gain Mathematics →](./19_entropy_and_information_gain.md)
