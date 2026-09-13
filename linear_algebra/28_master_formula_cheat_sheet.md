> 📖 **Navigation:** [← Previous: Part 27: 20 Essential Technical Interview Questions & Answers](./27_interview_questions_and_answers.md) | [🏠 Index](./README.md) | [Return to Index →](./README.md)

---

# PART 28 — MASTER FORMULA CHEAT SHEET & QUICK REFERENCE

A one-page high-density reference sheet of essential mathematical formulas, identities, and properties across Linear Algebra, Optimization, and Machine Learning.

---

## 1. Vector Operations & Norms

| Operation | Formula | Geometric Meaning / ML Note |
|---|---|---|
| **Dot Product** | $\mathbf{a} \cdot \mathbf{b} = \mathbf{a}^T \mathbf{b} = \sum_{i=1}^d a_i b_i = \|\mathbf{a}\|_2 \|\mathbf{b}\|_2 \cos\theta$ | Projection & directional alignment; Transformer attention $Q K^T$ |
| **Cauchy-Schwarz** | $|\mathbf{a}^T \mathbf{b}| \le \|\mathbf{a}\|_2 \|\mathbf{b}\|_2$ | Guarantees Cosine Similarity $\in [-1, 1]$ |
| **$L_1$ Norm** | $\|\mathbf{x}\|_1 = \sum_{i=1}^d \|x_i\|$ | Manhattan distance; promotes sparse weights (Lasso) |
| **$L_2$ Norm** | $\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^d x_i^2} = \sqrt{\mathbf{x}^T \mathbf{x}}$ | Euclidean length; weight decay / Ridge penalty |
| **$L_\infty$ Norm** | $\|\mathbf{x}\|_\infty = \max_{i} \|x_i\|$ | Chebyshev distance; adversarial perturbation bound ($\epsilon$-ball) |
| **General $L_p$ Norm** | $\|\mathbf{x}\|_p = \left(\sum_{i=1}^d \|x_i\|^p\right)^{1/p}$ | Valid norm for $p \ge 1$ (Minkowski inequality) |
| **Cosine Similarity** | $\cos\theta = \frac{\mathbf{a}^T \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}$ | Angle-only similarity independent of vector magnitude |

---

## 2. Matrix Operations & Properties

| Concept | Formula | Key Properties |
|---|---|---|
| **Transpose of Product** | $(A B)^T = B^T A^T$ | Reverse order: $(A B C)^T = C^T B^T A^T$ |
| **Trace** | $\text{Tr}(A) = \sum_{i=1}^n A_{ii} = \sum_{i=1}^n \lambda_i$ | Cyclic: $\text{Tr}(A B C) = \text{Tr}(B C A) = \text{Tr}(C A B)$ |
| **Frobenius Norm** | $\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{\text{Tr}(A^T A)} = \sqrt{\sum_{i=1}^r \sigma_i^2}$ | Matrix Euclidean norm; invariant under orthogonal transforms |
| **Spectral Norm** | $\|A\|_2 = \sigma_{\max}(A) = \sqrt{\lambda_{\max}(A^T A)}$ | Maximum magnification factor $\|A\mathbf{x}\|_2 / \|\mathbf{x}\|_2$ |
| **Nuclear Norm** | $\|A\|_* = \sum_{i=1}^r \sigma_i$ | Convex relaxation of matrix rank |
| **Hadamard Product** | $[A \odot B]_{ij} = A_{ij} B_{ij}$ | Element-wise product (used in LSTM gates, activations) |

---

## 3. Systems of Linear Equations & Elimination

* **System Matrix Form:** $A\mathbf{x} = \mathbf{b}$ where $A \in \mathbb{R}^{m \times n}$.
* **Row View:** Intersection of $m$ hyperplanes in $\mathbb{R}^n$.
* **Column View:** Target $\mathbf{b}$ expressed as a linear combination of feature columns $\sum x_j \mathbf{a}_j$.
* **Gaussian Elimination:** $[A \mid \mathbf{b}] \to [U \mid \mathbf{c}]$ (Row Echelon Form) via forward elimination ($O(\frac{1}{3}n^3)$), followed by back-substitution.
* **Gauss-Jordan Elimination:** Forward and backward elimination to Reduced Row Echelon Form (RREF) $[R \mid \mathbf{d}]$ where pivots are $1$ and pivot columns are standard basis vectors ($O(\frac{1}{2}n^3)$).
* **Solvability Criteria:**
  * Consistent $\iff \text{rank}(A) = \text{rank}([A \mid \mathbf{b}])$.
  * Unique solution $\iff \text{rank}(A) = \text{rank}([A \mid \mathbf{b}]) = n$.
  * Infinite solutions $\iff \text{rank}(A) = \text{rank}([A \mid \mathbf{b}]) < n$ (has $n - r$ free variables).
  * Inconsistent (no solution) $\iff \text{rank}(A) < \text{rank}([A \mid \mathbf{b}])$ (pivot in augmented column: $0 = 1$).

---

## 4. Matrix Inverses & Identities

* **Two-Sided Inverse ($n \times n$ Square, Full Rank):**

$$
A A^{-1} = A^{-1} A = I_n, \quad (A B)^{-1} = B^{-1} A^{-1}, \quad (A^T)^{-1} = (A^{-1})^T
$$

* **$2 \times 2$ Inverse Formula:**

$$
\begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
$$

* **Inversion by Adjugate / Cofactors:**

$$
A^{-1} = \frac{1}{\det(A)}\text{adj}(A) = \frac{1}{\det(A)} C^T
$$

  where $C_{ij} = (-1)^{i+j} M_{ij}$ are the cofactors, satisfying $A \text{adj}(A) = \det(A) I$.
* **Gauss-Jordan Matrix Inversion:**

$$
[A \mid I] \xrightarrow{\text{row operations}} [I \mid A^{-1}]
$$

* **Left Inverse (Tall $m > n$, Full Column Rank):**

$$
A_{\text{left}}^{-1} = (A^T A)^{-1} A^T \implies A_{\text{left}}^{-1} A = I_n \quad (\text{Least Squares: } \mathbf{x}_{\text{LS}} = A_{\text{left}}^{-1}\mathbf{b})
$$

* **Right Inverse (Wide $m < n$, Full Row Rank):**

$$
A_{\text{right}}^{-1} = A^T (A A^T)^{-1} \implies A A_{\text{right}}^{-1} = I_m \quad (\text{Minimum Norm: } \mathbf{x}_{\text{min}} = A_{\text{right}}^{-1}\mathbf{b})
$$

* **Moore-Penrose Pseudoinverse ($A^+$ for ANY matrix):**

$$
A^+ = V \Sigma^+ U^T \quad \text{where } [\Sigma^+]_{ii} = \begin{cases} 1/\sigma_i & \text{if } \sigma_i > 0 \\ 0 & \text{if } \sigma_i = 0 \end{cases}
$$

* **Sherman-Morrison Formula (Rank-1 Update):**

$$
(A + \mathbf{u}\mathbf{v}^T)^{-1} = A^{-1} - \frac{A^{-1}\mathbf{u}\mathbf{v}^T A^{-1}}{1 + \mathbf{v}^T A^{-1}\mathbf{u}}
$$

* **Woodbury Matrix Identity (Rank-$k$ Update):**

$$
(A + U C V)^{-1} = A^{-1} - A^{-1} U (C^{-1} + V A^{-1} U)^{-1} V A^{-1}
$$

---

## 5. Determinants

* **$2 \times 2$ Determinant:** $\det \begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$.
* **Laplace Expansion ($n \times n$):** $\det(A) = \sum_{j=1}^n A_{ij} C_{ij} = \sum_{i=1}^n A_{ij} C_{ij}$.
* **Triangular Matrix:** $\det(A) = \prod_{i=1}^n A_{ii}$ (product of diagonal entries/pivots).
* **Core Determinant Rules:**
  1. $\det(A B) = \det(A)\det(B)$
  2. $\det(A^{-1}) = \frac{1}{\det(A)}$
  3. $\det(A^T) = \det(A)$
  4. $\det(c A) = c^n \det(A)$ for $n \times n$ matrix
  5. $\det(A) = \prod_{i=1}^n \lambda_i$
  6. $A$ invertible $\iff \det(A) \neq 0$

---

## 6. Four Fundamental Subspaces (Strang's Big Picture)

For matrix $A \in \mathbb{R}^{m \times n}$ with $\text{rank}(A) = r$:

| Subspace | Notation | Lives in | Dimension | Orthogonal Complement |
|---|---|---|---|---|
| **Column Space** | $C(A)$ | $\mathbb{R}^m$ | $r$ | Left Null Space: $C(A) \perp N(A^T)$ |
| **Null Space** | $N(A)$ | $\mathbb{R}^n$ | $n - r$ | Row Space: $N(A) \perp C(A^T)$ |
| **Row Space** | $C(A^T)$ | $\mathbb{R}^n$ | $r$ | Null Space: $C(A^T) \perp N(A)$ |
| **Left Null Space** | $N(A^T)$ | $\mathbb{R}^m$ | $m - r$ | Column Space: $N(A^T) \perp C(A)$ |

* **Rank-Nullity Theorem:** $\dim(C(A^T)) + \dim(N(A)) = r + (n - r) = n$.
* **Fundamental Orthogonality:** $\mathbf{x}_{\text{row}} \cdot \mathbf{x}_{\text{null}} = 0$ and $\mathbf{y}_{\text{col}} \cdot \mathbf{y}_{\text{left-null}} = 0$.

---

## 7. Projections & Least Squares

* **Projection of vector $\mathbf{b}$ onto vector $\mathbf{a}$:**

$$
\mathbf{p} = \frac{\mathbf{a}^T \mathbf{b}}{\mathbf{a}^T \mathbf{a}}\mathbf{a}, \quad \text{Error: } \mathbf{e} = \mathbf{b} - \mathbf{p} \perp \mathbf{a}
$$

* **Projection Matrix onto Column Space $C(X)$:**

$$
P = X(X^T X)^{-1} X^T \quad (P^T = P, \; P^2 = P)
$$

* **Orthogonal Complement Projector:** $P_\perp = I - P$ (projects onto $N(X^T)$).
* **Normal Equation:**

$$
(X^T X)\mathbf{w}^* = X^T \mathbf{y} \implies \mathbf{w}^* = (X^T X)^{-1}X^T \mathbf{y}
$$

* **Residual Orthogonality:**

$$
\mathbf{e} = \mathbf{y} - X\mathbf{w}^* \in N(X^T) \implies X^T \mathbf{e} = \mathbf{0}
$$

* **Ridge Regularization:** $\mathbf{w}_{\text{ridge}}^* = (X^T X + \lambda I)^{-1} X^T \mathbf{y}$ (always invertible for $\lambda > 0$).

---

## 8. Spectral Theory & Symmetric Matrices

* **Eigenvalue Equation:** $A\mathbf{v} = \lambda \mathbf{v}$ with $\mathbf{v} \neq \mathbf{0}$.
* **Characteristic Equation:** $\det(A - \lambda I) = 0$.
* **Diagonalization:** $A = P D P^{-1}$ where $P = [\mathbf{v}_1 \mid \cdots \mid \mathbf{v}_n]$ and $D = \text{diag}(\lambda_1, \dots, \lambda_n)$.
* **Matrix Power:** $A^k = P D^k P^{-1} = P \text{diag}(\lambda_1^k, \dots, \lambda_n^k) P^{-1}$.
* **Spectral Theorem for Real Symmetric Matrices ($A = A^T$):**

$$
A = Q \Lambda Q^T = \sum_{i=1}^n \lambda_i \mathbf{q}_i \mathbf{q}_i^T \quad (Q^T Q = I, \; \lambda_i \in \mathbb{R})
$$

* **Trace & Determinant:** $\text{Tr}(A) = \sum \lambda_i$, $\det(A) = \prod \lambda_i$.
* **Rayleigh Quotient:** $R(A, \mathbf{x}) = \frac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T \mathbf{x}} \in [\lambda_{\min}, \lambda_{\max}]$.

---

## 9. Positive Definite Matrices

For real symmetric matrix $A \in \mathbb{R}^{n \times n}$:

| Classification | Quadratic Form | Eigenvalues | Leading Minors | Cholesky | Surface Shape |
|---|---|---|---|---|---|
| **Positive Definite ($A \succ 0$)** | $\mathbf{x}^T A \mathbf{x} > 0$ ($\mathbf{x} \neq \mathbf{0}$) | All $\lambda_i > 0$ | All $> 0$ | $A = L L^T$ exists | Upward bowl (strict minimum) |
| **Positive Semi-Definite ($A \succeq 0$)** | $\mathbf{x}^T A \mathbf{x} \ge 0$ | All $\lambda_i \ge 0$ | All $\ge 0$ | $A = L L^T$ | Upward trough / flat valley |
| **Negative Definite ($A \prec 0$)** | $\mathbf{x}^T A \mathbf{x} < 0$ ($\mathbf{x} \neq \mathbf{0}$) | All $\lambda_i < 0$ | Alternating $-, +, -, \dots$ | Does not exist | Downward dome (strict maximum) |
| **Indefinite** | $\mathbf{x}^T A \mathbf{x}$ changes sign | Mixed $+ / -$ | Violates tests | Does not exist | Saddle point |

* **Sample Covariance:** $\Sigma = \frac{1}{n-1}X_c^T X_c$ is always **Positive Semi-Definite** ($\mathbf{u}^T \Sigma \mathbf{u} = \frac{1}{n-1}\|X_c \mathbf{u}\|_2^2 \ge 0$).

---

## 10. Singular Value Decomposition (SVD) & PCA

* **Full SVD:** $A = U \Sigma V^T$ where $U \in \mathbb{R}^{m \times m}$ ($U^T U = I_m$), $\Sigma \in \mathbb{R}^{m \times n}$, $V \in \mathbb{R}^{n \times n}$ ($V^T V = I_n$).
* **Compact SVD:** $A = U_r \Sigma_r V_r^T = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T$ with $r = \text{rank}(A)$.
* **Singular Values:** $\sigma_i = \sqrt{\lambda_i(A^T A)} = \sqrt{\lambda_i(A A^T)}$, ordered $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0$.
* **Geometric Action:** Rotate ($V^T$) $\to$ Scale by $\sigma_i$ ($\Sigma$) $\to$ Rotate ($U$).
* **Eckart-Young Optimal Low-Rank Approximation:**

$$
A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T \implies \min_{\text{rank}(B)=k}\|A - B\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}, \quad \|A - A_k\|_2 = \sigma_{k+1}
$$

* **PCA via SVD:** For centered data $X_c = U \Sigma V^T$:
  * Principal directions: Columns of $V$ (eigenvectors of $\Sigma_{\text{cov}}$).
  * Principal variances: $\lambda_i = \frac{\sigma_i^2}{n-1}$.
  * Projected coordinates: $Z = X_c V = U \Sigma$.

---

## 11. Matrix Calculus Identities

| Expression | Gradient $\nabla_{\mathbf{x}}$ or Derivative |
|---|---|
| $\mathbf{a}^T \mathbf{x}$ | $\mathbf{a}$ |
| $\mathbf{x}^T A \mathbf{x}$ (general $A$) | $(A + A^T)\mathbf{x}$ |
| $\mathbf{x}^T A \mathbf{x}$ ($A = A^T$ symmetric) | $2 A \mathbf{x}$ |
| $\|\mathbf{x}\|_2^2 = \mathbf{x}^T \mathbf{x}$ | $2 \mathbf{x}$ |
| $\|A\mathbf{x} - \mathbf{b}\|_2^2$ | $2 A^T (A\mathbf{x} - \mathbf{b})$ |
| $\text{Tr}(A X)$ with respect to matrix $X$ | $A^T$ |
| $\log \det(X)$ with respect to matrix $X$ | $X^{-T}$ |
| Linear layer backprop $\mathbf{z} = W\mathbf{x} + \mathbf{b}$ | $\frac{\partial \mathcal{L}}{\partial W} = \boldsymbol{\delta} \mathbf{x}^T, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{x}} = W^T \boldsymbol{\delta} \quad \left(\boldsymbol{\delta} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}}\right)$ |

---

## 12. Information Theory & Classification Loss

* **Shannon Entropy:** $H(X) = -\sum_{i=1}^C p_i \log_2 p_i$.
* **Binary Entropy:** $H(p) = -p \log_2 p - (1-p)\log_2(1-p)$. Maximum at $p=0.5$ ($1.0\text{ bit}$).
* **Gini Impurity:** $\text{Gini}(S) = 1 - \sum_{i=1}^C p_i^2$. Maximum at $p=0.5$ ($0.50$).
* **Information Gain:** $IG(S, A) = H(S) - \sum_{v} \frac{|S_v|}{|S|} H(S_v)$.
* **KL Divergence:** $D_{\text{KL}}(P \parallel Q) = \sum_{i=1}^C p_i \log \left(\frac{p_i}{q_i}\right) \ge 0$.
* **Cross-Entropy Loss:** $\mathcal{L}_{\text{CE}} = -\sum_{i=1}^C y_i \log \hat{y}_i = H(P) + D_{\text{KL}}(P \parallel Q)$.

---

> 📖 **Navigation:** [← Previous: Part 27: 20 Essential Technical Interview Questions & Answers](./27_interview_questions_and_answers.md) | [🏠 Index](./README.md) | [Return to Index →](./README.md)
