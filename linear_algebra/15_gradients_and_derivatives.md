> 📖 **Navigation:** [← Previous: Part 14: Linear Regression Matrix Mathematics](./14_linear_regression_matrix_math.md) | [🏠 Index](./README.md) | [Next: Part 16: The Chain Rule & Backpropagation →](./16_chain_rule_and_backprop.md)

---

# PART 15 — GRADIENTS & DERIVATIVES FOR OPTIMIZATION

Optimization is the process of adjusting parameters to minimize a loss function. **Gradients** provide the downhill compass directing this optimization.

---

## 15.1 1D Derivatives & Partial Derivatives

* **1D Derivative:** Rate of instantaneous change: $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$.
* **Partial Derivative ($\frac{\partial f}{\partial x_i}$):** Measures the rate of change along one variable $x_i$ while holding all other variables constant.

---

## 15.2 The Gradient Vector ($\nabla f$) & Steepest Descent

The **Gradient** $\nabla f(\mathbf{x})$ bundles all partial derivatives into a single vector:

$$
\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_d} \end{bmatrix}
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

* **Steepest Ascent:** $\nabla f(\mathbf{x})$ points in the direction of **maximum rate of increase**.
* **Steepest Descent:** $-\nabla f(\mathbf{x})$ points in the direction of **maximum rate of decrease**.
* **Gradient Descent Update Rule:**
  $$
  \mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla \mathcal{L}(\mathbf{w}_t)
  $$
  where $\alpha > 0$ is the scalar learning rate.

---

## 15.3 Deriving the Linear Regression Loss Gradient

For MSE loss $\mathcal{L}(\mathbf{w}) = \|\mathbf{y} - X\mathbf{w}\|_2^2 = (\mathbf{y} - X\mathbf{w})^T (\mathbf{y} - X\mathbf{w})$:

$$
\nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}) = 2 X^T (X\mathbf{w} - \mathbf{y}) = 2 X^T (\hat{\mathbf{y}} - \mathbf{y}) = -2 X^T \mathbf{e}
$$

* **Batch Gradient Descent Step for Linear Regression:**
  $$
  \mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \cdot \frac{2}{n} X^T (X\mathbf{w}_t - \mathbf{y})
  $$

---

## 15.4 Master Matrix Calculus Identities (Quick Reference)

```
                    ESSENTIAL MATRIX CALCULUS IDENTITIES
  ┌──────────────────────────────────────────┬─────────────────────────────┐
  │ Scalar Function f                        │ Gradient                    │
  ├──────────────────────────────────────────┼─────────────────────────────┤
  │ a^T x                                    │ ∇_x = a                     │
  │ x^T A x  (for symmetric A = A^T)         │ ∇_x = 2 A x                 │
  │ ||A x - b||_2^2                          │ ∇_x = 2 A^T (A x - b)       │
  │ Tr(A X)                                  │ ∇_X = A^T                   │
  │ log det(X)  (for X ≻ 0)                  │ ∇_X = X^-T                  │
  └──────────────────────────────────────────┴─────────────────────────────┘
```

---

## 15.5 The Hessian Matrix & Multivariable Curvature

The **Hessian** $H \in \mathbb{R}^{d \times d}$ collects all second-order partial derivatives ($H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$), measuring the **local curvature** of the loss surface:

```
                        CRITICAL POINT CLASSIFICATION
   Hessian Eigenvalues                     Geometry & Classification
   ────────────────────────────────────────────────────────────────────────
   All λ_i > 0  (H ≻ 0, Positive Definite)  Strict Local Minimum (Upward Bowl)
   All λ_i < 0  (H ≺ 0, Negative Definite)  Strict Local Maximum (Downward Dome)
   Mixed signs  (H is Indefinite)           SADDLE POINT (Mountain Pass)
```

---

## Advanced / Optional — Do not study until Core track is complete

### A.1 Newton-Raphson Second-Order Optimization
Newton's Method fits a local quadratic model and jumps directly to its minimum:
$$
\mathbf{x}_{k+1} = \mathbf{x}_k - [H(\mathbf{x}_k)]^{-1} \nabla f(\mathbf{x}_k)
$$
*(Quadratic convergence rate, but requires $\mathcal{O}(d^3)$ FLOPs to invert the Hessian, making it impractical for deep learning with millions of parameters).*

---

> 📖 **Navigation:** [← Previous: Part 14: Linear Regression Matrix Mathematics](./14_linear_regression_matrix_math.md) | [🏠 Index](./README.md) | [Next: Part 16: The Chain Rule & Backpropagation →](./16_chain_rule_and_backprop.md)
