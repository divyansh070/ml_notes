> 📖 **Navigation:** [← Previous: Part 16: The Chain Rule & Backpropagation](./16_chain_rule_and_backprop.md) | [🏠 Index](./README.md) | [Next: Part 18: Regularization Mathematics →](./18_regularization_mathematics.md)

---

# PART 17 — DISTANCE & SIMILARITY METRICS

Distance and similarity metrics quantify the geometric separation between data points in feature space.

---

## 17.1 Core Distance Metrics

| Metric | Mathematical Formula | Geometric Meaning |
| :--- | :--- | :--- |
| **Euclidean ($L_2$)** | $d_2(\mathbf{p}, \mathbf{q}) = \sqrt{\sum (p_i - q_i)^2}$ | Straight-line geometric distance. |
| **Manhattan ($L_1$)** | $d_1(\mathbf{p}, \mathbf{q}) = \sum \|p_i - q_i\|$ | Grid / city-block distance along coordinate axes. |
| **Chebyshev ($L_\infty$)** | $d_\infty(\mathbf{p}, \mathbf{q}) = \max_i \|p_i - q_i\|$ | Maximum difference along any single dimension (Chessboard distance). |
| **Minkowski ($L_p$)** | $d_p(\mathbf{p}, \mathbf{q}) = \left(\sum \|p_i - q_i\|^p\right)^{1/p}$ | Generalized $L_p$ metric ($p=1 \implies L_1, p=2 \implies L_2$). |
| **Cosine Similarity** | $\cos\theta = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$ | Angular alignment regardless of magnitude ($\in [-1, 1]$). |
| **Hamming Distance** | $d_H(\mathbf{p}, \mathbf{q}) = \sum \mathbb{I}(p_i \neq q_i)$ | Number of positions where categorical / binary tokens disagree. |

---

## 17.2 Practical Machine Learning Selection Guide

```
                       PRACTICAL METRIC SELECTION MATRIX
  ┌─────────────────────────────────┬───────────────────────────────┬───────────────────────────────┐
  │ Data Type / Scenario            │ Recommended Metric            │ Why?                          │
  ├─────────────────────────────────┼───────────────────────────────┼───────────────────────────────┤
  │ Continuous spatial / physical   │ Euclidean (L2)                │ Isotropic, rotation-invariant │
  │ Tabular grid / robust to outliers│ Manhattan (L1)               │ Does not square extreme errors│
  │ NLP / LLM Text Embeddings       │ Cosine Similarity             │ Normalizes document length    │
  │ Binary vectors / One-Hot tokens │ Hamming Distance              │ Counts bit mismatches         │
  │ Correlated Gaussian features    │ Mahalanobis Distance          │ Accounts for covariance tilt  │
  └─────────────────────────────────┴───────────────────────────────┴───────────────────────────────┘
```

---

> 📖 **Navigation:** [← Previous: Part 16: The Chain Rule & Backpropagation](./16_chain_rule_and_backprop.md) | [🏠 Index](./README.md) | [Next: Part 18: Regularization Mathematics →](./18_regularization_mathematics.md)
