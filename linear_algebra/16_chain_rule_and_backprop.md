> 📖 **Navigation:** [← Previous: Part 15: Gradients & Derivatives for Optimization](./15_gradients_and_derivatives.md) | [🏠 Index](./README.md) | [Next: Part 17: Distance & Similarity Metrics →](./17_distance_and_similarity_metrics.md)

---

# PART 15 — THE CHAIN RULE & BACKPROPAGATION

---

## 15.1 Single-Variable & Multivariable Chain Rule

For a composite nested function $y = f(u)$ where $u = g(x)$:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

---

## 15.2 Hand Trace on a Nested Function

Let $y = (3x + 2)^2$. Find $\frac{dy}{dx}$ at $x = 1$.
1. Break down into sub-nodes:
   * Inner function: $u = 3x + 2$
   * Outer function: $y = u^2$
2. Compute individual derivatives:
   * $\frac{du}{dx} = 3$
   * $\frac{dy}{du} = 2u$
3. Apply chain rule:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = (2u) \times 3 = 2(3x + 2) \times 3 = 6(3x + 2) = 18x + 12
$$

4. Evaluate at $x = 1$:

$$
\frac{dy}{dx}\Big|_{x=1} = 18(1) + 12 = 30
$$

---

## 15.3 Neural Network Computational Graphs

In deep learning, every neural network layer is a composite function:

$$
\text{Input } x \quad \xrightarrow{\quad \text{Linear} \quad} \quad z = wx + b \quad \xrightarrow{\quad \text{Activation} \quad} \quad a = \sigma(z) \quad \xrightarrow{\quad \text{Loss} \quad} \quad \mathcal{L}(a, y)
$$

```
     FORWARD PASS:   x ──► [ z = w*x + b ] ──► [ a = sigma(z) ] ──► [ Loss L ]
                                                                        │
     BACKWARD PASS:  dL/dw = (dL/da) * (da/dz) * (dz/dw)  ◄─────────────┘
```

By the Chain Rule:

$$
\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

* $\frac{\partial z}{\partial w} = x$
* $\frac{\partial a}{\partial z} = \sigma'(z)$
* $\frac{\partial \mathcal{L}}{\partial a} = \text{Loss gradient}$
* **The Insight:** **Backpropagation is nothing more than the repeated application of the Chain Rule from output to input!**

---

> 📖 **Navigation:** [← Previous: Part 15: Gradients & Derivatives for Optimization](./15_gradients_and_derivatives.md) | [🏠 Index](./README.md) | [Next: Part 17: Distance & Similarity Metrics →](./17_distance_and_similarity_metrics.md)
