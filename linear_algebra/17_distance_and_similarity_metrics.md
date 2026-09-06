> 📖 **Navigation:** [← Previous: Part 16: The Chain Rule & Backpropagation](./16_chain_rule_and_backprop.md) | [🏠 Index](./README.md) | [Next: Part 18: Regularization Mathematics (L1 vs. L2) →](./18_regularization_mathematics.md)

---

# PART 16 — DISTANCE & SIMILARITY METRICS

---

## 16.1 Euclidean Distance ($L_2$) vs. Manhattan Distance ($L_1$)

Let Point $A = (1, 2)$ and Point $B = (4, 6)$:

1. **Euclidean Distance ($L_2$):**

$$
d_{L2}(A, B) = \sqrt{(4 - 1)^2 + (6 - 2)^2} = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = \mathbf{5.0}
$$

2. **Manhattan Distance ($L_1$):**

$$
d_{L1}(A, B) = |4 - 1| + |6 - 2| = |3| + |4| = 3 + 4 = \mathbf{7.0}
$$

3. **Minkowski Distance ($L_p$ Generalization):**

$$
d_{Lp}(A, B) = \left(\sum_{i=1}^{n} |a_i - b_i|^p\right)^{1/p}
$$

   * $p=1 \implies$ Manhattan
   * $p=2 \implies$ Euclidean

---

## 16.2 Cosine Distance vs. Cosine Similarity

$$
\text{Cosine Distance} = 1 - \text{Cosine Similarity} = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}
$$

* If vectors are identical ($\theta = 0^\circ$): Similarity $= 1.0 \implies$ Distance $= 0.0$.
* If vectors are orthogonal ($\theta = 90^\circ$): Similarity $= 0.0 \implies$ Distance $= 1.0$.

---

## 16.3 Comparison Table & Real-World Selection Matrix

| Metric | Formula | Sensitive to Scale? | When to Use (ML Applications) |
| :--- | :--- | :--- | :--- |
| **Euclidean ($L_2$)** | $\sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$ | **High** | Physical spatial coordinates, image pixel grids, KNN on standardized features. |
| **Manhattan ($L_1$)** | $\sum_{i=1}^{n} \lvert p_i - q_i \rvert$ | **High** | High-dimensional data (less vulnerable to Curse of Dimensionality than $L_2$), grid layouts. |
| **Cosine Similarity** | $\frac{\mathbf{u} \cdot \mathbf{v}}{\lVert \mathbf{u} \rVert_2 \lVert \mathbf{v} \rVert_2}$ | **Zero (Invariant)** | Text embeddings, recommendation systems, semantic search (focuses on angle, not magnitude). |

---

> 📖 **Navigation:** [← Previous: Part 16: The Chain Rule & Backpropagation](./16_chain_rule_and_backprop.md) | [🏠 Index](./README.md) | [Next: Part 18: Regularization Mathematics (L1 vs. L2) →](./18_regularization_mathematics.md)
