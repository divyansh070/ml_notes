> 📖 **Navigation:** [← Previous: Part 15: Gradients & Derivatives for Optimization](./15_gradients_and_derivatives.md) | [🏠 Index](./README.md) | [Next: Part 17: Distance & Similarity Metrics →](./17_distance_and_similarity_metrics.md)

---

# PART 16 — THE CHAIN RULE & BACKPROPAGATION

**Backpropagation** is the computational engine of Deep Learning. It is simply the **Multivariable Chain Rule** applied efficiently across a directed computational graph.

---

## 16.1 The Multivariable Chain Rule

For a composite function $\mathcal{L}(a(z(w)))$:
$$
\frac{d\mathcal{L}}{dw} = \frac{d\mathcal{L}}{da} \cdot \frac{da}{dz} \cdot \frac{dz}{dw}
$$

```
               FORWARD PASS (Computing Activations & Loss):
      Input x ──► [ z = w*x + b ] ──► [ a = σ(z) ] ──► [ Loss L = (a - y)^2 ]
                        │                   │                  │
                        ▼                   ▼                  ▼
               BACKWARD PASS (Propagating Error Gradients):
      dL/dw = (dL/da) * (da/dz) * (dz/dw)  ◄───────────────────┘
```

---

## 16.2 Complete Hand Calculation on a Single Neuron

Let us trace a complete numerical forward and backward pass on a single neuron:
* Input: $x = 2.0$, Target: $y = 1.0$
* Parameters: $w = 0.5$, $b = 0.1$
* Activation: Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ (where $\sigma'(z) = \sigma(z)(1 - \sigma(z))$)
* Loss: Squared Error $\mathcal{L} = \frac{1}{2}(a - y)^2$

### Forward Pass:
1. $z = w x + b = (0.5)(2.0) + 0.1 = \mathbf{1.1}$
2. $a = \sigma(1.1) = \frac{1}{1 + e^{-1.1}} \approx \mathbf{0.750}$
3. $\mathcal{L} = \frac{1}{2}(0.750 - 1.0)^2 = \frac{1}{2}(-0.25)^2 = \mathbf{0.03125}$

### Backward Pass:
1. **Loss gradient w.r.t. activation:**
   $$
   \frac{\partial \mathcal{L}}{\partial a} = (a - y) = 0.750 - 1.0 = \mathbf{-0.250}
   $$
2. **Activation gradient w.r.t. pre-activation:**
   $$
   \frac{\partial a}{\partial z} = a(1 - a) = (0.750)(1 - 0.750) = (0.750)(0.250) = \mathbf{0.1875}
   $$
3. **Pre-activation gradient w.r.t. weight:**
   $$
   \frac{\partial z}{\partial w} = x = \mathbf{2.0}
   $$
4. **Assemble Weight Gradient via Chain Rule:**
   $$
   \frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w} = (-0.250) \times (0.1875) \times (2.0) = \mathbf{-0.09375}
   $$

* **Weight Update ($\alpha = 0.1$):**
  $$
  w_{\text{new}} = w - \alpha \frac{\partial \mathcal{L}}{\partial w} = 0.5 - 0.1(-0.09375) = \mathbf{0.509375}
  $$

---

## 16.3 The Jacobian Matrix & Vector-Valued Backpropagation

When a layer transforms vector $\mathbf{x} \in \mathbb{R}^n$ into output vector $\mathbf{f}(\mathbf{x}) \in \mathbb{R}^m$, the **Jacobian Matrix** $J \in \mathbb{R}^{m \times n}$ collects all pairwise partial derivatives:

$$
J_{ij} = \frac{\partial f_i}{\partial x_j}
$$

* **Vector-Jacobian Product (VJP):** Modern autodiff frameworks (PyTorch, JAX) never construct massive Jacobian matrices in memory; they directly evaluate the **Vector-Jacobian Product** $\mathbf{v}^T J$, propagating upstream gradients backward in $\mathcal{O}(n)$ time.

---

> 📖 **Navigation:** [← Previous: Part 15: Gradients & Derivatives for Optimization](./15_gradients_and_derivatives.md) | [🏠 Index](./README.md) | [Next: Part 17: Distance & Similarity Metrics →](./17_distance_and_similarity_metrics.md)
