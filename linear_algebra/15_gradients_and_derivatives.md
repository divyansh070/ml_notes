> 📖 **Navigation:** [← Previous: Part 14: Linear Regression Matrix Mathematics](./14_linear_regression_matrix_math.md) | [🏠 Index](./README.md) | [Next: Part 16: The Chain Rule & Backpropagation →](./16_chain_rule_and_backprop.md)

---

# PART 14 — GRADIENTS & DERIVATIVES FOR OPTIMIZATION

---

## 14.1 1D Derivatives & Slope of Tangent Line

The derivative $\frac{df}{dx}$ measures the instantaneous rate of change of $f(x)$ with respect to $x$:

$$
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

* *Hand Example:* For $f(x) = x^2$:

$$
f'(x) = \lim_{h \to 0} \frac{(x+h)^2 - x^2}{h} = \lim_{h \to 0} \frac{x^2 + 2xh + h^2 - x^2}{h} = \lim_{h \to 0} (2x + h) = 2x
$$

  * At $x = 3$: Slope $= 2(3) = +6$ (Function is increasing steeply).
  * At $x = 0$: Slope $= 2(0) = 0$ (Minimum point / flat slope).
  * At $x = -2$: Slope $= 2(-2) = -4$ (Function is decreasing).

---

## 14.2 Partial Derivatives (Multivariable Functions)

When a function depends on multiple variables $f(x_1, x_2, \dots, x_n)$, the **partial derivative** $\frac{\partial f}{\partial x_i}$ measures how $f$ changes when varying $x_i$ while treating **all other variables as constants**.

### Hand Calculation Example
Let $f(x, y) = x^2 + 3xy + y^2$:
1. **Compute $\frac{\partial f}{\partial x}$ (treat $y$ as a constant number):**

$$
\frac{\partial f}{\partial x} = \frac{d}{dx}[x^2] + \frac{d}{dx}[3y \cdot x] + \frac{d}{dx}[y^2] = 2x + 3y + 0 = 2x + 3y
$$

2. **Compute $\frac{\partial f}{\partial y}$ (treat $x$ as a constant number):**

$$
\frac{\partial f}{\partial y} = \frac{d}{dy}[x^2] + \frac{d}{dy}[3x \cdot y] + \frac{d}{dy}[y^2] = 0 + 3x + 2y = 3x + 2y
$$

---

## 14.3 The Gradient Vector (Direction of Steepest Ascent)

The **Gradient** $\nabla f$ bundles all partial derivatives into a single vector:

$$
\nabla f(x, y) =
\begin{bmatrix}
\frac{\partial f}{\partial x} \\
\frac{\partial f}{\partial y}
\end{bmatrix}
$$

* **Fundamental Theorem:** The gradient vector $\nabla f$ points in the **direction of greatest rate of increase (steepest uphill slope)**.
* **Negative Gradient ($-\nabla f$):** Points in the **direction of steepest descent (fastest downhill path to the minimum)**.

---

## 14.4 Gradient Descent: 3-Step Numerical Hand Trace

We wish to find the minimum of $f(x) = x^2$ using Gradient Descent.
* Update Rule: $x_{t+1} = x_t - \alpha \nabla f(x_t)$
* Gradient: $\nabla f(x) = 2x$
* Settings: Start at initial guess $x_0 = 4.0$, Learning Rate $\alpha = 0.1$.

```
     Iteration 0: x = 4.0  ──►  f(x) = 16.0   (Gradient = 8.0)
     Iteration 1: x = 3.2  ──►  f(x) = 10.24  (Gradient = 6.4)
     Iteration 2: x = 2.56 ──►  f(x) = 6.55   (Gradient = 5.12)
     Iteration 3: x = 2.05 ──►  f(x) = 4.19   (Converging smoothly toward x=0!)
```

### Iteration 1:
1. Compute gradient at $x_0 = 4.0$:

$$
\nabla f(4.0) = 2(4.0) = 8.0
$$

2. Update parameter:

$$
x_1 = 4.0 - 0.1(8.0) = 4.0 - 0.8 = \mathbf{3.2}
$$

### Iteration 2:
1. Compute gradient at $x_1 = 3.2$:

$$
\nabla f(3.2) = 2(3.2) = 6.4
$$

2. Update parameter:

$$
x_2 = 3.2 - 0.1(6.4) = 3.2 - 0.64 = \mathbf{2.56}
$$

### Iteration 3:
1. Compute gradient at $x_2 = 2.56$:

$$
\nabla f(2.56) = 2(2.56) = 5.12
$$

2. Update parameter:

$$
x_3 = 2.56 - 0.1(5.12) = 2.56 - 0.512 = \mathbf{2.048}
$$

---

> 📖 **Navigation:** [← Previous: Part 14: Linear Regression Matrix Mathematics](./14_linear_regression_matrix_math.md) | [🏠 Index](./README.md) | [Next: Part 16: The Chain Rule & Backpropagation →](./16_chain_rule_and_backprop.md)
