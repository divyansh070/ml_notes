> 📖 **Navigation:** [← Previous: Part 25: Information Theory for ML](./25_information_theory_for_ml.md) | [🏠 Index](./README.md) | [Next: Part 27: Interview Questions & Answers →](./27_interview_questions_and_answers.md)

---

# PART 26 — PAPER & PENCIL SELF-TEST CHECKLIST

Before technical interviews, machine learning coding rounds, or research assessments, test yourself on a blank sheet of paper without calculators, libraries, or search engines. The skills are organized into **three progressive mastery tiers**.

---

## 🟢 Level A: Hand Calculations (Crunch the Numbers)

- [ ] **1. Vector Dot Product & Norms:** Given $\mathbf{a} = \begin{bmatrix} 3 \\ -4 \end{bmatrix}$ and $\mathbf{b} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$, compute $\mathbf{a} \cdot \mathbf{b} = 2$, $\|\mathbf{a}\|_1 = 7$, $\|\mathbf{a}\|_2 = 5$, $\|\mathbf{a}\|_\infty = 4$, and $\cos\theta = \frac{2}{5\sqrt{5}}$.
- [ ] **2. Matrix Multiplication:** Multiply $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and $B = \begin{bmatrix} 0 & 1 \\ -1 & 2 \end{bmatrix}$ to obtain $AB = \begin{bmatrix} -2 & 5 \\ -4 & 11 \end{bmatrix}$, and verify $(AB)^T = B^T A^T$.
- [ ] **3. $2 \times 2$ Determinant & Inverse:** For $A = \begin{bmatrix} 4 & 7 \\ 2 & 6 \end{bmatrix}$, compute $\det(A) = 10$ and $A^{-1} = \frac{1}{10}\begin{bmatrix} 6 & -7 \\ -2 & 4 \end{bmatrix} = \begin{bmatrix} 0.6 & -0.7 \\ -0.2 & 0.4 \end{bmatrix}$.
- [ ] **4. Gaussian Elimination to Solve $A\mathbf{x} = \mathbf{b}$:** Solve the $2 \times 2$ system $2x_1 + x_2 = 5$ and $x_1 - x_2 = 1$ via forward elimination to row echelon form and back-substitution to get $\mathbf{x} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$.
- [ ] **5. Gauss-Jordan Matrix Inversion $[A \mid I] \to [I \mid A^{-1}]$:** Invert $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ using row operations on $\left[\begin{array}{cc|cc} 1 & 2 & 1 & 0 \\ 3 & 4 & 0 & 1 \end{array}\right]$ to obtain $A^{-1} = \begin{bmatrix} -2 & 1 \\ 1.5 & -0.5 \end{bmatrix}$.
- [ ] **6. Inversion by Adjugate / Cofactors ($3 \times 3$):** Given an upper triangular matrix $A = \begin{bmatrix} 1 & 2 & 0 \\ 0 & 3 & 1 \\ 0 & 0 & 2 \end{bmatrix}$, compute all 9 minors $M_{ij}$, cofactors $C_{ij} = (-1)^{i+j}M_{ij}$, construct the cofactor matrix $C$, transpose to get $\operatorname{adj}(A) = C^T$, and divide by $\det(A) = 6$ to find $A^{-1}$.
- [ ] **7. Linear Independence Check:** Determine whether $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$ and $\mathbf{v}_2 = \begin{bmatrix} 2 \\ 4 \end{bmatrix}$ are linearly dependent by checking if $\det([\mathbf{v}_1 \mid \mathbf{v}_2]) = 0$ or by finding non-trivial scalars $c_1, c_2$.
- [ ] **8. Rank via Row Echelon Form:** Reduce $A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 0 & 1 & 2 \end{bmatrix}$ to REF to identify the pivot count $\operatorname{rank}(A) = 2$ and nullity $\dim(N(A)) = 3 - 2 = 1$.
- [ ] **9. Vector Projection:** Project $\mathbf{b} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$ onto $\mathbf{a} = \begin{bmatrix} 4 \\ 0 \end{bmatrix}$ using $\mathbf{p} = \frac{\mathbf{a}^T\mathbf{b}}{\mathbf{a}^T\mathbf{a}}\mathbf{a} = \begin{bmatrix} 3 \\ 0 \end{bmatrix}$, and verify orthogonality $\mathbf{a}^T(\mathbf{b} - \mathbf{p}) = 0$.
- [ ] **10. Eigenvalues & Eigenvectors ($2 \times 2$):** For $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$, solve $\det(A - \lambda I) = \lambda^2 - 4\lambda + 3 = 0$ to get $\lambda_1 = 3, \lambda_2 = 1$. Solve $(A - \lambda I)\mathbf{v} = \mathbf{0}$ to find eigenvectors $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \mathbf{v}_2 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$.
- [ ] **11. Diagonalization & Matrix Power:** For $A$ above, diagonalize $A = P D P^{-1}$ and compute $A^3 = P D^3 P^{-1}$.
- [ ] **12. SVD Hand Calculation ($2 \times 2$ rank-1):** For $A = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$, compute eigenvalues of $A^T A = \begin{bmatrix} 2 & 2 \\ 2 & 2 \end{bmatrix}$ ($\lambda_1 = 4, \lambda_2 = 0$), find singular value $\sigma_1 = \sqrt{4} = 2$, right singular vector $\mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix}$, and left singular vector $\mathbf{u}_1 = \frac{1}{\sigma_1}A\mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix}$. Verify $A = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T$.
- [ ] **13. Sample Covariance & Variance:** For 3 centered 2D data points, calculate the empirical sample covariance matrix $S = \frac{1}{n-1} X^T X$.
- [ ] **14. Shannon Entropy & Gini Impurity:** For a binary dataset with 4 positive and 12 negative samples ($p = 0.25$), calculate $H(0.25) = -(0.25\log_2 0.25 + 0.75\log_2 0.75) \approx 0.811\text{ bits}$ and $\text{Gini} = 1 - (0.25^2 + 0.75^2) = 0.375$.

---

## 🔵 Level B: Conceptual & Geometric Explanations (Explain & Draw)

- [ ] **15. Column Picture vs. Row Picture:** Draw the row picture (intersection of hyperplanes) and column picture (linear combination of feature vectors hitting target $\mathbf{b}$) for $A\mathbf{x} = \mathbf{b}$.
- [ ] **16. Four Fundamental Subspaces Diagram (Strang's Big Picture):** Draw the two domain/codomain spaces showing Row Space $C(A^T) \perp$ Null Space $N(A)$ in $\mathbb{R}^n$, and Column Space $C(A) \perp$ Left Null Space $N(A^T)$ in $\mathbb{R}^m$.
- [ ] **17. Geometric Determinant:** Explain why $\det(A) = 0$ geometrically means the transformation collapses an $n$-dimensional volume into lower dimensions, destroying invertibility.
- [ ] **18. Geometry of Least Squares:** Sketch how target $\mathbf{y} \notin C(X)$ is projected onto the column space $C(X)$ along an error vector $\mathbf{e}$ orthogonal to every column of $X$ ($X^T \mathbf{e} = \mathbf{0}$).
- [ ] **19. SVD as Action on the Unit Sphere:** Draw how $A = U \Sigma V^T$ maps the unit sphere into a hyper-ellipse: rotate by $V^T$, stretch along axes by singular values $\sigma_i$, and rotate into target coordinates by $U$.
- [ ] **20. Full vs. Compact vs. Truncated SVD:** Sketch the block sizes of matrices for Full SVD ($m \times m$, $m \times n$, $n \times n$), Compact SVD ($m \times r$, $r \times r$, $r \times n$), and Truncated rank-$k$ approximation ($m \times k$, $k \times k$, $k \times n$).
- [ ] **21. Two-Sided vs. Left vs. Right Inverse:** Explain why tall matrices ($m > n$, full column rank) have left inverses $(A^T A)^{-1}A^T$, wide matrices ($m < n$, full row rank) have right inverses $A^T(A A^T)^{-1}$, and non-square matrices cannot have a two-sided inverse.
- [ ] **22. Geometry of PCA:** Draw a 2D point cloud, draw the orthogonal principal eigenvectors of covariance matrix $\Sigma$, and explain why the first eigenvector points in the direction of maximum variance.
- [ ] **23. Definiteness Surfaces (Quadratic Forms):** Draw the 3D surface $z = \mathbf{x}^T A \mathbf{x}$ for Positive Definite (bowl pointing up), Negative Definite (dome pointing down), and Indefinite (saddle point).
- [ ] **24. $L_1$ vs. $L_2$ Regularization Contours:** Sketch the diamond $L_1$ ball and circular $L_2$ ball intersecting the elliptical MSE loss contours to explain why $L_1$ yields sparse weights (corner hits).

---

## 🟣 Level C: First-Principles Derivations

- [ ] **25. Normal Equation via Residual Orthogonality:** Derive $(X^T X)\mathbf{w}^* = X^T \mathbf{y}$ starting strictly from the geometric requirement that residual $\mathbf{e} = \mathbf{y} - X\mathbf{w}^*$ must be orthogonal to the column space $C(X)$ ($X^T \mathbf{e} = \mathbf{0}$).
- [ ] **26. Normal Equation via Matrix Calculus:** Expand $\mathcal{L}(\mathbf{w}) = (\mathbf{y} - X\mathbf{w})^T(\mathbf{y} - X\mathbf{w})$, compute gradient $\nabla_{\mathbf{w}} \mathcal{L}$, set to $\mathbf{0}$, and solve for $\mathbf{w}^*$.
- [ ] **27. Proof that Covariance $\Sigma$ is Positive Semi-Definite:** Show that for any non-zero vector $\mathbf{u} \in \mathbb{R}^d$, $\mathbf{u}^T \Sigma \mathbf{u} = \frac{1}{n-1}\|X_c \mathbf{u}\|_2^2 \ge 0$.
- [ ] **28. PCA First Principal Component via Lagrange Multipliers:** Maximize $\mathbf{w}^T \Sigma \mathbf{w}$ subject to $\mathbf{w}^T \mathbf{w} = 1$ using Lagrangian $\mathcal{L}(\mathbf{w}, \lambda) = \mathbf{w}^T \Sigma \mathbf{w} - \lambda(\mathbf{w}^T\mathbf{w} - 1)$ to prove $\Sigma \mathbf{w} = \lambda \mathbf{w}$.
- [ ] **29. Gram-Schmidt to QR Factorization:** Orthonormalize $\mathbf{a}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \mathbf{a}_2 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ to find $\mathbf{q}_1, \mathbf{q}_2$, and construct upper-triangular matrix $R = Q^T A$.
- [ ] **30. Matrix Backpropagation:** For linear layer $\mathbf{z} = W\mathbf{x} + \mathbf{b}$, derive the matrix gradients $\frac{\partial \mathcal{L}}{\partial W} = \boldsymbol{\delta} \mathbf{x}^T$ and $\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = W^T \boldsymbol{\delta}$ where $\boldsymbol{\delta} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}}$.

---

> 📖 **Navigation:** [← Previous: Part 25: Information Theory for ML](./25_information_theory_for_ml.md) | [🏠 Index](./README.md) | [Next: Part 27: Interview Questions & Answers →](./27_interview_questions_and_answers.md)
