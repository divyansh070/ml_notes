> 📖 **Navigation:** [← Previous: Part 20: ML Mathematics Roadmap Table](./20_ml_mathematics_roadmap.md) | [🏠 Index](./README.md) | [Next: Part 22: Four Fundamental Subspaces (Strang's Big Picture) →](./22_four_fundamental_subspaces.md)

---

# PART 21 — "WHAT I SHOULD BE ABLE TO DO ON PAPER" CHECKLIST

Before technical interviews and coding assessments, test yourself on a blank sheet of paper with no calculator or libraries. The skills are divided into **three progressive mastery tiers**.

---

## 🟢 Level A: Hand Calculations on $2 \times 2$ Matrices (Crunch the Numbers)

- [ ] **1. Vector Dot Product & Angle:** Given $\mathbf{a} = [2, 3]^T, \mathbf{b} = [4, 1]^T$, compute $\mathbf{a} \cdot \mathbf{b} = 11$ and $\cos\theta = 11 / (\sqrt{13}\sqrt{17})$.
- [ ] **2. Vector Norms:** Given $\mathbf{v} = [3, -4]^T$, compute $\|\mathbf{v}\|_2 = 5$, $\|\mathbf{v}\|_1 = 7$, and $\|\mathbf{v}\|_\infty = 4$.
- [ ] **3. Cosine Similarity:** Given $\mathbf{u} = [1, 2]^T, \mathbf{v} = [2, 4]^T$, show that $\text{sim}(\mathbf{u}, \mathbf{v}) = 1.0$ (parallel vectors).
- [ ] **4. Matrix Multiplication:** Given $A = \left[\begin{smallmatrix} 1 & 2 \\ 3 & 4 \end{smallmatrix}\right]$ and $B = \left[\begin{smallmatrix} 5 & 6 \\ 7 & 8 \end{smallmatrix}\right]$, compute $AB = \left[\begin{smallmatrix} 19 & 22 \\ 43 & 50 \end{smallmatrix}\right]$.
- [ ] **5. Matrix Transpose Product:** Verify on small matrices that $(AB)^T = B^T A^T \neq A^T B^T$.
- [ ] **6. 2x2 Determinant:** Given $A = \left[\begin{smallmatrix} 5 & 3 \\ 2 & 4 \end{smallmatrix}\right]$, compute $\det(A) = (5)(4) - (3)(2) = 14$.
- [ ] **7. 2x2 Matrix Inversion:** Invert $A = \left[\begin{smallmatrix} 4 & 7 \\ 2 & 6 \end{smallmatrix}\right] \implies A^{-1} = \frac{1}{10}\left[\begin{smallmatrix} 6 & -7 \\ -2 & 4 \end{smallmatrix}\right] = \left[\begin{smallmatrix} 0.6 & -0.7 \\ -0.2 & 0.4 \end{smallmatrix}\right]$.
- [ ] **8. 2x2 Linear System ($A\mathbf{x} = \mathbf{b}$):** Solve $2x_1 + x_2 = 5$ and $x_1 - x_2 = 1$ using Gaussian elimination to find $\mathbf{x} = [2, 1]^T$.
- [ ] **9. Check Linear Independence:** Prove that $\mathbf{a}_1 = [1, 2]^T$ and $\mathbf{a}_2 = [2, 4]^T$ are linearly dependent by showing $c_1 \mathbf{a}_1 + c_2 \mathbf{a}_2 = \mathbf{0}$ for $c_1 = -2, c_2 = 1$.
- [ ] **10. Sample Covariance Calculation:** For $X = [1, 2, 3]$ and $Y = [2, 3, 7]$, compute $\text{Var}(X) = 1.0$, $\text{Var}(Y) = 7.0$, and $\text{Cov}(X, Y) = 2.5$ using Bessel's $n-1$ denominator.
- [ ] **11. Characteristic Polynomial:** For $A = \left[\begin{smallmatrix} 2 & 1 \\ 1 & 2 \end{smallmatrix}\right]$, write $\det(A - \lambda I) = \lambda^2 - 4\lambda + 3 = 0$.
- [ ] **12. Eigenvalues & Eigenvectors:** Factor $\lambda_1 = 3, \lambda_2 = 1$. Find eigenvectors $\mathbf{v}_1 = [1, 1]^T$ and $\mathbf{v}_2 = [1, -1]^T$.
- [ ] **13. Vector Projection:** Project $\mathbf{a} = [3, 4]^T$ onto line $\mathbf{b} = [4, 0]^T$ to find $\mathbf{p} = [3, 0]^T$ and residual $\mathbf{e} = [0, 4]^T$.
- [ ] **14. 1D Gradient Descent Step:** For $f(x) = x^2$, start at $x_0 = 4$ with learning rate $\alpha = 0.1$. Compute $x_1 = 4 - 0.1(8) = 3.2$ and $x_2 = 2.56$.
- [ ] **15. Shannon Entropy Calculation:** Compute $H(0.5, 0.5) = 1.0 \text{ Bit}$ and $H(1.0, 0.0) = 0.0 \text{ Bits}$.

---

## 🔵 Level B: Conceptual & Geometric Explanations (Explain & Draw)

- [ ] **16. Column Picture vs. Row Picture:** Draw the column picture of $A\mathbf{x} = \mathbf{b}$ as a linear combination of column vectors landing on $\mathbf{b}$.
- [ ] **17. Geometric Meaning of Determinant:** Explain why $\det(A) = 0$ means the transformation flattens space, causing information loss.
- [ ] **18. Geometry of PCA:** Draw a 2D data ellipse, sketch the first principal component $\mathbf{v}_1$ along the major axis, and show why $\text{Var}(Z_1) = \lambda_1$.
- [ ] **19. SVD as Three Transformations:** Draw the unit circle transformed via $A = U \Sigma V^T$: Rotate by $V^T \to$ Stretch along axes by $\sigma_i \to$ Rotate into target space by $U$.
- [ ] **20. Geometry of $L_1$ vs. $L_2$ Regularization:** Draw the diamond $L_1$ ball and circular $L_2$ ball intersecting elliptical loss contours to explain why $L_1$ causes sparsity.
- [ ] **21. Strang's 4 Subspaces on Paper:** Draw the master diagram showing $C(A^T) \perp N(A)$ in $\mathbb{R}^n$ and $C(A) \perp N(A^T)$ in $\mathbb{R}^m$.
- [ ] **22. Geometry of Linear Regression:** Draw $\mathbf{y}$ splitting into prediction $\hat{\mathbf{y}} \in C(X)$ and residual error $\mathbf{e} \in N(X^T)$.
- [ ] **23. Why We Don't Invert $(X^T X)$:** Explain why QR decomposition avoids squaring the condition number $\kappa(X^T X) = \kappa(X)^2$.
- [ ] **24. Definiteness Surfaces:** Sketch the 3D surface of a Positive Definite matrix (bowl), Negative Definite (dome), and Indefinite (saddle point).

---

## 🟣 Level C: Step-by-Step Derivations (From First Principles)

- [ ] **25. Normal Equation Derivation:** Starting from $\mathcal{L}(\mathbf{w}) = \|\mathbf{y} - X\mathbf{w}\|_2^2$, expand to $\mathbf{y}^T\mathbf{y} - 2\mathbf{w}^T X^T\mathbf{y} + \mathbf{w}^T X^T X \mathbf{w}$, take gradient $\nabla_{\mathbf{w}} \mathcal{L} = \mathbf{0}$, and derive $(X^T X)\mathbf{w} = X^T \mathbf{y}$.
- [ ] **26. PCA Variance Maximization Derivation:** Set up Lagrangian $\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^T \Sigma \mathbf{w} - \lambda(\mathbf{w}^T\mathbf{w} - 1)$, compute $\nabla_{\mathbf{w}} \mathcal{L} = \mathbf{0}$, and derive $\Sigma\mathbf{w} = \lambda\mathbf{w}$.
- [ ] **27. Proof that Covariance $\Sigma$ is PSD:** Show that for any non-zero vector $\mathbf{u}$, $\mathbf{u}^T \Sigma \mathbf{u} = \frac{1}{n-1}\|X_c \mathbf{u}\|_2^2 \ge 0$.
- [ ] **28. Gram-Schmidt to QR Derivation:** Orthonormalize $\mathbf{a}_1 = [1, 1]^T, \mathbf{a}_2 = [1, 0]^T$ to find $\mathbf{q}_1, \mathbf{q}_2$ and construct upper-triangular matrix $R = Q^T A$.
- [ ] **29. Moore-Penrose Pseudoinverse via Normal Equation:** Derive $A^+ = (A^T A)^{-1} A^T$ by solving the least-squares normal equations for full column rank matrix $A$.
- [ ] **30. Neural Network Backprop via Chain Rule:** Derive $\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \sigma'(z_2) \cdot W_2^T \cdot \phi'(z_1) \cdot \mathbf{x}^T$ for a 2-layer MLP.

---

> 📖 **Navigation:** [← Previous: Part 20: ML Mathematics Roadmap Table](./20_ml_mathematics_roadmap.md) | [🏠 Index](./README.md) | [Next: Part 22: Four Fundamental Subspaces (Strang's Big Picture) →](./22_four_fundamental_subspaces.md)
