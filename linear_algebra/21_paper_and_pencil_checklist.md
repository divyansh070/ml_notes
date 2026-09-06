> 📖 **Navigation:** [← Previous: Part 20: ML Mathematics Roadmap Table](./20_ml_mathematics_roadmap.md) | [🏠 Index](./README.md) | [Next: Part 22: Four Fundamental Subspaces (Strang's Big Picture) →](./22_four_fundamental_subspaces.md)

---

# PART 20 — "WHAT I SHOULD BE ABLE TO DO ON PAPER" CHECKLIST

Before technical interviews and coding assessments, test yourself on a blank sheet of paper:

- [ ] **1. Vector Dot Product:** Given $\mathbf{a} = [2, 3]^T, \mathbf{b} = [4, 1]^T$, compute $\mathbf{a} \cdot \mathbf{b} = 11$.
- [ ] **2. Vector Norms:** Given $\mathbf{v} = [3, -4]^T$, compute $\|\mathbf{v}\|_2 = 5$ and $\|\mathbf{v}\|_1 = 7$.
- [ ] **3. Euclidean Distance:** Given $\mathbf{p} = [1, 2]^T, \mathbf{q} = [4, 6]^T$, compute $d(\mathbf{p}, \mathbf{q}) = 5$.
- [ ] **4. Cosine Similarity:** Given $\mathbf{u} = [1, 2]^T, \mathbf{v} = [2, 4]^T$, prove similarity $= 1.0$.
- [ ] **5. Matrix Multiplication:** Given $A$ and $B$, compute $AB$:

$$
A =
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix},
\quad
B =
\begin{bmatrix}
5 & 6 \\
7 & 8
\end{bmatrix}
\implies
AB =
\begin{bmatrix}
19 & 22 \\
43 & 50
\end{bmatrix}
$$

- [ ] **6. Matrix Transpose:** Transpose a non-square matrix and prove $(AB)^T = B^T A^T$.
- [ ] **7. 2x2 Determinant:** Given $A$, compute $\det(A) = 14$:

$$
A =
\begin{bmatrix}
5 & 3 \\
2 & 4
\end{bmatrix}
$$

- [ ] **8. 2x2 Matrix Inverse:** Given $A$, compute $A^{-1}$ and verify $A A^{-1} = I$:

$$
A =
\begin{bmatrix}
4 & 7 \\
2 & 6
\end{bmatrix}
\implies
A^{-1} =
\begin{bmatrix}
0.6 & -0.7 \\
-0.2 & 0.4
\end{bmatrix}
$$

- [ ] **9. Solve Linear System:** Convert $2x+y=5, x-y=1$ into $A\mathbf{x}=\mathbf{b}$ and solve $\mathbf{x} = [2, 1]^T$.
- [ ] **10. Check Linear Independence:** Prove that $[1, 2]^T$ and $[2, 4]^T$ are dependent ($\det = 0$).
- [ ] **11. Sample Covariance Table:** Given $X = [1, 2, 3], Y = [2, 3, 7]$, compute $\text{Var}(X)=1, \text{Var}(Y)=7, \text{Cov}(X,Y)=2.5$.
- [ ] **12. Build Covariance Matrix:** Construct $\Sigma$ from scratch:

$$
\Sigma =
\begin{bmatrix}
1.0 & 2.5 \\
2.5 & 7.0
\end{bmatrix}
$$

- [ ] **13. Characteristic Equation:** Set up $\det(A - \lambda I) = 0$ for $A$ and get $\lambda^2 - 4\lambda + 3 = 0$:

$$
A =
\begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}
$$

- [ ] **14. Derive Eigenvalues:** Solve $\lambda_1 = 3, \lambda_2 = 1$.
- [ ] **15. Derive Eigenvectors:** Substitute $\lambda=3$ to find $\mathbf{v}_1 = [1, 1]^T$ and $\lambda=1$ to find $\mathbf{v}_2 = [1, -1]^T$.
- [ ] **16. Verify Eigenvectors:** Multiply $A \mathbf{v}_1$ and show it equals $3 \mathbf{v}_1$.
- [ ] **17. Project Vector:** Project $\mathbf{a} = [3, 4]^T$ onto $\mathbf{b} = [4, 0]^T$ to get $\mathbf{p} = [3, 0]^T$.
- [ ] **18. Normal Equation:** Write $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$ and explain every term.
- [ ] **19. Partial Derivatives:** Compute $\frac{\partial}{\partial x}[x^2 + 3xy + y^2] = 2x + 3y$.
- [ ] **20. Gradient Descent Step:** Given $f(x)=x^2, x_0=4, \alpha=0.1$, compute step $x_1 = 3.2, x_2 = 2.56$.
- [ ] **21. Chain Rule:** Differentiate $y = (3x + 2)^2 \implies \frac{dy}{dx} = 18x + 12$.
- [ ] **22. Entropy Calculation:** Compute $H(0.5, 0.5) = 1.0 \text{ Bit}$ and $H(1.0, 0.0) = 0.0 \text{ Bits}$.
- [ ] **23. Four Subspaces & Null Space:** For the $2 \times 2$ matrix with rows $[1, 2]$ and $[2, 4]$, find $N(A) = \text{span}([-2, 1]^T)$ and verify $\text{rank}(A) + \text{nullity}(A) = 2$.
- [ ] **24. Gram-Schmidt Orthogonalization:** Orthonormalize $\mathbf{a}_1 = [1, 1]^T, \mathbf{a}_2 = [1, 0]^T$ to find $\mathbf{q}_1 = [1/\sqrt{2}, 1/\sqrt{2}]^T, \mathbf{q}_2 = [1/\sqrt{2}, -1/\sqrt{2}]^T$.
- [ ] **25. Construct QR Decomposition:** Factor $A = QR$ and construct the upper-triangular matrix $R$ with diagonal entries $[\sqrt{2}, 1/\sqrt{2}]$.
- [ ] **26. Positive Definite Test:** Compute quadratic form $\mathbf{x}^T A \mathbf{x} = 2x_1^2 + 2x_1 x_2 + 2x_2^2$ for matrix $A$ (eigenvalues $\lambda_1=3, \lambda_2=1 \gt 0$) and prove it is Positive Definite.
- [ ] **27. Why Covariance is PSD:** Prove $\mathbf{x}^T \Sigma \mathbf{x} = \frac{1}{n-1} \|X_c \mathbf{x}\|_2^2 \ge 0$ for any vector $\mathbf{x}$.
- [ ] **28. Moore-Penrose Pseudoinverse:** For column vector $A = [1, 2]^T$, compute $A^+ = (A^T A)^{-1} A^T = [0.2, 0.4]$ and verify $A^+ A = [1.0]$.
- [ ] **29. Linear Transformation Geometry:** Apply 2D Rotation $R_{90^\circ}$ (mapping $[1, 0]^T \to [0, 1]^T$) and Scaling to vectors on paper.
- [ ] **30. SVD Geometric Decomposition:** Interpret $A = U \Sigma V^T$ as Rotate $\to$ Stretch $\to$ Rotate.

---

---

> 📖 **Navigation:** [← Previous: Part 20: ML Mathematics Roadmap Table](./20_ml_mathematics_roadmap.md) | [🏠 Index](./README.md) | [Next: Part 22: Four Fundamental Subspaces (Strang's Big Picture) →](./22_four_fundamental_subspaces.md)
