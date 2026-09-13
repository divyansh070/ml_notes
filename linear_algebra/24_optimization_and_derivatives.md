> 📖 **Navigation:** [← Previous: Part 23: Gram-Schmidt & QR Decomposition](./23_gram_schmidt_and_qr_decomposition.md) | [🏠 Index](./README.md) | [Next: Part 25: Information Theory for ML →](./25_information_theory_for_ml.md)

---

# PART 24 — OPTIMIZATION & DERIVATIVES FOR ML

*Multivariate Calculus & Optimization Companion Module*

Optimization is the process of adjusting parameters to minimize a loss function. While linear algebra provides the coordinate geometry, **calculus** provides the downhill compass directing gradient descent and backpropagation.

---

## 24.1 1D Derivatives, Partial Derivatives & The Gradient

* **1D Derivative:** Rate of instantaneous change:

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

* **Partial Derivative ($\frac{\partial f}{\partial x_i}$):** Measures the rate of change along one coordinate axis $x_i$ while holding all other variables constant.
* **The Gradient Vector ($\nabla f$):** Packages all partial derivatives into a single vector pointing in the **direction of steepest ascent**:

$$
\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_d} \end{bmatrix} \in \mathbb{R}^d
$$

```
                         THE GRADIENT DESCENT DIRECTION
                                     Loss L
                                       │       ● Start (w_0)
                                       │      /
                   -∇L (Steepest) ◄────┼─────●  (Downhill Step)
                                       │    /
                                       └───┴────────► Parameter w
```

### Gradient Descent Update Rule:
To minimize loss $\mathcal{L}(\mathbf{w})$, step in the direction of **steepest descent** ($-\nabla \mathcal{L}$):

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla \mathcal{L}(\mathbf{w}_t)
$$

where $\alpha > 0$ is the learning rate.

---

## 24.2 Master Matrix Calculus Identities

In machine learning derivations, working with scalar summation indices is slow and error-prone. Use these standard matrix calculus identities directly:

| Scalar Function $f(\mathbf{x})$ | Gradient $\nabla_{\mathbf{x}} f(\mathbf{x})$ | Note / Assumption |
| :--- | :--- | :--- |
| $\mathbf{a}^T \mathbf{x} = \mathbf{x}^T \mathbf{a}$ | $\mathbf{a}$ | Linear inner product |
| $\mathbf{x}^T A \mathbf{x}$ | $(A + A^T)\mathbf{x}$ | General quadratic form |
| $\mathbf{x}^T A \mathbf{x}$ | $2A\mathbf{x}$ | When $A$ is symmetric ($A = A^T$) |
| $\|\mathbf{x}\|_2^2 = \mathbf{x}^T \mathbf{x}$ | $2\mathbf{x}$ | Squared Euclidean norm |
| $\|\mathbf{y} - X\mathbf{w}\|_2^2$ | $-2X^T (\mathbf{y} - X\mathbf{w})$ | OLS loss gradient w.r.t $\mathbf{w}$ |
| $\text{Tr}(A X)$ | $A^T$ | Trace derivative w.r.t matrix $X$ |
| $\log \det(X)$ | $X^{-1}$ | Log-determinant for symmetric $X \succ 0$ |

---

## 24.3 The Hessian Matrix & Loss Curvature

The **Hessian** $H = \nabla^2 \mathcal{L}(\mathbf{w}) \in \mathbb{R}^{d \times d}$ collects all second-order partial derivatives:

$$
H_{ij} = \frac{\partial^2 \mathcal{L}}{\partial w_i \partial w_j}
$$

* **Second-Order Taylor Approximation:**

$$
\mathcal{L}(\mathbf{w} + \Delta \mathbf{w}) \approx \mathcal{L}(\mathbf{w}) + \nabla \mathcal{L}^T \Delta \mathbf{w} + \frac{1}{2} \Delta \mathbf{w}^T H \Delta \mathbf{w}
$$

* **Eigenvalues of $H$ Govern Optimization:**
  * If $\lambda_{\max}(H) \gg \lambda_{\min}(H)$ (Condition number $\kappa(H) \gg 1$), the loss surface is an **elongated ravine**. Gradient descent oscillates wildly across the canyon walls rather than progressing down the valley floor.
  * Momentum and Adam resolve this by scaling step sizes adaptively along each principal curvature axis.

---

## 24.4 The Multivariable Chain Rule & Backpropagation

**Backpropagation** is simply the multivariable chain rule applied systematically across a directed computational graph.

```
               FORWARD PASS (Computing Activations & Loss):
      Input x ──► [ z = w*x + b ] ──► [ a = σ(z) ] ──► [ Loss L = (a - y)^2 ]
                        │                   │                  │
                        ▼                   ▼                  ▼
               BACKWARD PASS (Propagating Error Gradients):
      dL/dw = (dL/da) * (da/dz) * (dz/dw)  ◄───────────────────┘
```

For composite function $\mathcal{L}(a(z(w)))$:

$$
\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

---

## 24.5 Complete Hand Calculation: Single Neuron Forward & Backward Pass

* Input: $x = 2.0$, Target: $y = 1.0$
* Parameters: $w = 0.5$, $b = 0.1$
* Activation: Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ (with derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z))$)
* Loss: Squared Error $\mathcal{L} = \frac{1}{2}(a - y)^2$

### Forward Pass:
1. Pre-activation: $z = w x + b = (0.5)(2.0) + 0.1 = \mathbf{1.1}$
2. Activation: $a = \sigma(1.1) = \frac{1}{1 + e^{-1.1}} \approx \mathbf{0.750}$
3. Loss: $\mathcal{L} = \frac{1}{2}(0.750 - 1.0)^2 = \frac{1}{2}(-0.25)^2 = \mathbf{0.03125}$

### Backward Pass (Backprop):
1. **Loss gradient w.r.t activation $a$:**

$$
\frac{\partial \mathcal{L}}{\partial a} = a - y = 0.750 - 1.0 = \mathbf{-0.250}
$$

2. **Activation gradient w.r.t pre-activation $z$:**

$$
\frac{\partial a}{\partial z} = a(1 - a) = (0.750)(1 - 0.750) = (0.750)(0.250) = \mathbf{0.1875}
$$

3. **Pre-activation gradient w.r.t parameter weight $w$:**

$$
\frac{\partial z}{\partial w} = x = \mathbf{2.0}
$$

4. **Pre-activation gradient w.r.t bias $b$:**

$$
\frac{\partial z}{\partial b} = 1.0
$$

### Assemble Gradients via Chain Rule:

$$
\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w} = (-0.250) \times (0.1875) \times (2.0) = \mathbf{-0.09375}
$$

$$
\frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial b} = (-0.250) \times (0.1875) \times (1.0) = \mathbf{-0.046875}
$$

### Parameter Updates (Learning Rate $\alpha = 0.1$):
* $w_{\text{new}} = 0.5 - 0.1(-0.09375) = 0.5 + 0.009375 = \mathbf{0.509375}$
* $b_{\text{new}} = 0.1 - 0.1(-0.046875) = 0.1 + 0.0046875 = \mathbf{0.1046875}$

---

> 📖 **Navigation:** [← Previous: Part 23: Gram-Schmidt & QR Decomposition](./23_gram_schmidt_and_qr_decomposition.md) | [🏠 Index](./README.md) | [Next: Part 25: Information Theory for ML →](./25_information_theory_for_ml.md)
