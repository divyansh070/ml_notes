# LINEAR ALGEBRA & MATHEMATICS FOR MACHINE LEARNING — PART 2
### Advanced Subspaces, Matrix Decompositions, Positive Definite Forms, Pseudoinverse & Interview Q&A

> 📖 **Navigation:** [← Return to Part 1: Fundamentals, Eigendecomposition, Calculus & Optimization](LINEAR_ALGEBRA_AND_MATH.md)

---

## 📑 TABLE OF CONTENTS (PART 2)

21. [PART 21 — 40 ESSENTIAL TECHNICAL INTERVIEW QUESTIONS](#part-21--40-essential-technical-interview-questions)
    - [Linear Algebra Core](#linear-algebra-core)
    - [Eigenvalues, Covariance & PCA](#eigenvalues-covariance--pca)
    - [Calculus & Optimization](#calculus--optimization)
    - [Machine Learning Mathematics](#machine-learning-mathematics)
    - [Advanced Linear Algebra & Decompositions](#advanced-linear-algebra--decompositions)
22. [PART 22 — FOUR FUNDAMENTAL SUBSPACES](#part-22--four-fundamental-subspaces)
    - [22.1 Column Space](#221-column-space)
    - [22.2 Row Space](#222-row-space)
    - [22.3 Null Space](#223-null-space)
    - [22.4 Left Null Space](#224-left-null-space)
    - [22.5 The Big Picture: Orthogonal Decomposition](#225-the-big-picture-orthogonal-decomposition)
    - [22.6 Rank-Nullity Theorem](#226-rank-nullity-theorem)
    - [22.7 ML Connection](#227-ml-connection)
23. [PART 23 — GRAM-SCHMIDT ORTHOGONALIZATION & QR DECOMPOSITION](#part-23--gram-schmidt-orthogonalization--qr-decomposition)
    - [23.1 Why We Need Orthogonal Bases](#231-why-we-need-orthogonal-bases)
    - [23.2 Gram-Schmidt Derivation](#232-gram-schmidt-derivation)
    - [23.3 Build Q and R](#233-build-q-and-r)
    - [23.4 Why QR Matters in ML](#234-why-qr-matters-in-ml)
24. [PART 24 — POSITIVE DEFINITE & POSITIVE SEMIDEFINITE MATRICES](#part-24--positive-definite--positive-semidefinite-matrices)
    - [24.1 Quadratic Form](#241-quadratic-form)
    - [24.2 Positive Definite](#242-positive-definite)
    - [24.3 Positive Semidefinite](#243-positive-semidefinite)
    - [24.4 Negative Definite and Indefinite](#244-negative-definite-and-indefinite)
    - [24.5 Geometric Intuition](#245-geometric-intuition)
    - [24.6 ML Connections](#246-ml-connections)
25. [PART 25 — MOORE-PENROSE PSEUDOINVERSE](#part-25--moore-penrose-pseudoinverse)
    - [25.1 Why Ordinary Inverse Fails](#251-why-ordinary-inverse-fails)
    - [25.2 What the Pseudoinverse Does](#252-what-the-pseudoinverse-does)
    - [25.3 Least Squares Connection](#253-least-squares-connection)
    - [25.4 Numerical Hand Calculation](#254-numerical-hand-calculation)
    - [25.5 ML Applications](#255-ml-applications)
26. [PART 26 — LINEAR TRANSFORMATIONS](#part-26--linear-transformations)
    - [26.1 Definition](#261-definition)
    - [26.2 The Two Defining Properties](#262-the-two-defining-properties)
    - [26.3 Matrices as Geometric Transformations](#263-matrices-as-geometric-transformations)
    - [26.4 Connection to Eigenvectors](#264-connection-to-eigenvectors)
    - [26.5 Connection to SVD](#265-connection-to-svd)
    - [26.6 ML Connection](#266-ml-connection)
27. [FINAL SECTION — ONE-PAGE MASTER FORMULA CHEAT SHEET](#final-section--one-page-master-formula-cheat-sheet)

---

# PART 21 — 40 ESSENTIAL TECHNICAL INTERVIEW QUESTIONS

### Linear Algebra Core
1. **Q:** What does a vector represent in Machine Learning?
   * **A:** A point or feature representation in $p$-dimensional space, where each coordinate corresponds to a measurable feature of a single observation.
2. **Q:** What is the geometric meaning of the dot product?
   * **A:** $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|_2 \|\mathbf{b}\|_2 \cos\theta$. It measures directional alignment between two vectors and is proportional to the length of the projection of $\mathbf{a}$ onto $\mathbf{b}$.
3. **Q:** Why does matrix multiplication appear everywhere in Deep Learning?
   * **A:** A layer in a neural network is an affine transformation $\mathbf{z} = W\mathbf{x} + \mathbf{b}$. Matrix multiplication transforms all inputs and feature dimensions simultaneously across batches via hardware-accelerated parallel BLAS routines.
4. **Q:** What is matrix rank and why does it matter?
   * **A:** The maximum number of linearly independent rows or columns in a matrix. It measures the true dimensionality of information. If rank is less than the number of features, redundant feature columns exist.
5. **Q:** What does a determinant represent geometrically?
   * **A:** The scaling factor by which a linear transformation multiplies the area in 2D or volume in 3D of a unit geometric region.
6. **Q:** When does a matrix have an inverse?
   * **A:** When it is square and non-singular ($\det(A) \neq 0$), meaning it has full rank and does not collapse space into a lower dimension.
7. **Q:** What is the difference between an orthogonal and an orthonormal matrix?
   * **A:** In an orthogonal matrix, column vectors are mutually perpendicular. In an orthonormal matrix, column vectors are both perpendicular and unit length ($\|\mathbf{q}_i\|_2 = 1$), satisfying $Q^T Q = I \implies Q^{-1} = Q^T$.

### Eigenvalues, Covariance & PCA
8. **Q:** What is an eigenvector and an eigenvalue?
   * **A:** An eigenvector $\mathbf{v}$ is a non-zero vector whose direction remains unchanged under matrix transformation $A$. The eigenvalue $\lambda$ is the scaling factor satisfying $A\mathbf{v} = \lambda \mathbf{v}$.
9. **Q:** Why do we solve $\det(A - \lambda I) = 0$ to find eigenvalues?
   * **A:** To guarantee a non-trivial solution ($\mathbf{v} \neq \mathbf{0}$) to $(A - \lambda I)\mathbf{v} = \mathbf{0}$, the matrix $(A - \lambda I)$ must be singular, which requires its determinant to equal zero.
10. **Q:** Why is the covariance matrix always symmetric?
    * **A:** Because $\text{Cov}(X_i, X_j) = \text{Cov}(X_j, X_i)$. The joint linear deviation between feature $i$ and feature $j$ is commutative.
11. **Q:** Why must we center data ($\bar{x} = 0$) before applying PCA?
    * **A:** PCA maximizes variance measured from the coordinate origin. Without mean-centering, the first principal component points toward the mean of the data cloud rather than along the axis of maximum variance.
12. **Q:** Why does PCA select the eigenvector with the largest eigenvalue?
    * **A:** Because the eigenvalue $\lambda_i$ equals the variance of the projected data along eigenvector $\mathbf{v}_i$. The largest eigenvalue corresponds to the direction retaining maximal variance.
13. **Q:** What is the difference between PCA and SVD?
    * **A:** PCA analyzes maximum variance via sample covariance $\Sigma = \frac{1}{n-1} X_c^T X_c$. SVD factorizes the data matrix directly as $X_c = U \Sigma V^T$. The right singular vectors in $V$ are mathematically identical to the principal component directions of $\Sigma$.
14. **Q:** Why do production libraries compute PCA using SVD rather than Eigendecomposition?
    * **A:** Forming $X_c^T X_c$ can introduce numerical squaring errors and precision loss for ill-conditioned matrices. SVD computes singular vectors directly on $X_c$, providing superior numerical stability.

### Calculus & Optimization
15. **Q:** What is the mathematical definition of a gradient?
    * **A:** A vector containing all first-order partial derivatives of a multivariable function: $\nabla f = [\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n}]^T$. It points in the direction of steepest ascent.
16. **Q:** Why do we subtract the gradient in Gradient Descent?
    * **A:** Because $\nabla f$ points uphill toward maximum loss. To minimize loss, parameters step in the opposite direction ($-\nabla f$) downhill.
17. **Q:** What happens if the learning rate $\alpha$ is too large or too small?
    * **A:** If $\alpha$ is too small, training converges very slowly and may get trapped in shallow local minima. If too large, parameter updates overshoot the valley, causing loss to oscillate or diverge to $\infty$.
18. **Q:** How does Backpropagation utilize the Chain Rule?
    * **A:** It computes the loss gradient with respect to hidden weights by multiplying local derivatives backward through the computational graph: $\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_2} \cdot \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial W_1}$.
19. **Q:** What is the Hessian matrix?
    * **A:** The square matrix of second-order partial derivatives ($H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$) describing the local curvature of the loss surface.

### Machine Learning Mathematics
20. **Q:** Why does Ordinary Least Squares minimize squared errors rather than absolute errors?
    * **A:** Squared error $(\mathbf{y} - X\mathbf{w})^2$ is smoothly differentiable everywhere, yielding the closed-form analytical solution $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$. Absolute error has a non-differentiable cusp at zero.
21. **Q:** What is the geometric interpretation of the Normal Equation?
    * **A:** The prediction vector $\hat{\mathbf{y}} = X\mathbf{w}$ is the orthogonal projection of target vector $\mathbf{y}$ onto the column space of feature matrix $X$.
22. **Q:** Why does L1 regularization produce sparse models while L2 does not?
    * **A:** The $L_1$ constraint region is a diamond with sharp corners on the coordinate axes, which intersect expanding loss contours at exact zero weights. The $L_2$ region is a smooth circle with no corners, shrinking weights smoothly without setting them to zero.
23. **Q:** Why is feature scaling mandatory before Ridge or Lasso regularization?
    * **A:** The penalty term $\alpha \sum w_j^2$ treats all coefficients equally. If features have different scales, unscaled larger features receive disproportionate penalties solely due to arbitrary units.
24. **Q:** Why does Logistic Regression use the Sigmoid function instead of a step function?
    * **A:** A step function has zero derivative almost everywhere, preventing gradient optimization. Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$ is smooth and differentiable with derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.
25. **Q:** What is the difference between Euclidean distance and Cosine similarity?
    * **A:** Euclidean distance measures the geometric distance between point coordinates, whereas Cosine similarity measures the angle between vectors, ignoring differences in magnitude.
26. **Q:** When would you choose Manhattan distance (L1) over Euclidean distance (L2)?
    * **A:** In high-dimensional spaces where the Curse of Dimensionality causes Euclidean distances to concentrate, or when distance represents movement along a grid.
27. **Q:** What is Shannon Entropy?
    * **A:** A measure of the average uncertainty or information content in a probability distribution: $H(X) = -\sum_{i=1}^{C} p_i \log_2 p_i$.
28. **Q:** Why do Decision Trees use Gini Impurity instead of Entropy by default?
    * **A:** Gini impurity $\text{Gini} = 1 - \sum p_i^2$ only requires basic arithmetic multiplication, whereas Entropy requires computing logarithms ($\log_2 p_i$), making Gini faster across millions of candidate split evaluations.
29. **Q:** What is the geometric meaning of Support Vectors in SVM?
    * **A:** They are the critical data points lying directly on the margin hyperplanes ($\mathbf{w}^T \mathbf{x} + b = \pm 1$) that uniquely define the decision boundary.
30. **Q:** Why is Bessel's correction ($n-1$) needed for sample variance?
    * **A:** Deviations from the sample mean $\bar{x}$ underestimate deviations from the true population mean $\mu$. Dividing by $n-1$ compensates for this bias, yielding an unbiased variance estimator.

### Advanced Linear Algebra & Decompositions
31. **Q:** What are the Four Fundamental Subspaces of a matrix $A \in \mathbb{R}^{m \times n}$?
    * **A:** Column Space $C(A) \subset \mathbb{R}^m$, Row Space $C(A^T) \subset \mathbb{R}^n$, Null Space $N(A) \subset \mathbb{R}^n$, and Left Null Space $N(A^T) \subset \mathbb{R}^m$. They form orthogonal complements: $C(A^T) \perp N(A)$ and $C(A) \perp N(A^T)$.
32. **Q:** What does the Rank-Nullity Theorem mean in practical ML terms?
    * **A:** $\text{rank}(A) + \text{nullity}(A) = n$. In a dataset with $n$ feature columns, rank represents the number of informative independent feature dimensions, while nullity represents the number of redundant dimensions mapping to zero.
33. **Q:** Why does the least-squares residual error vector lie in the Left Null Space $N(A^T)$?
    * **A:** Because least squares projects $\mathbf{y}$ orthogonally onto Column Space $C(A)$. The residual $\mathbf{e} = \mathbf{y} - A\hat{\mathbf{x}}$ is perpendicular to all columns of $A$, satisfying $A^T \mathbf{e} = \mathbf{0}$, which defines $N(A^T)$.
34. **Q:** What is the purpose of Gram-Schmidt Orthogonalization?
    * **A:** It converts linearly independent vectors into an equivalent set of orthonormal unit vectors spanning the exact same subspace, eliminating cross-talk between coordinate axes.
35. **Q:** Why do production ML libraries solve Linear Regression via QR Decomposition rather than normal equations?
    * **A:** Forming $X^T X$ squares the condition number ($\kappa(X^T X) = \kappa(X)^2$), amplifying numerical rounding errors. QR decomposition ($X = QR$) solves $R\mathbf{w} = Q^T \mathbf{y}$ via back-substitution without explicit matrix inversion.
36. **Q:** What is the difference between a Positive Definite (PD) and a Positive Semidefinite (PSD) matrix?
    * **A:** A symmetric matrix $A$ is PD ($A \succ 0$) if its quadratic form satisfies $\mathbf{x}^T A \mathbf{x} \gt 0$ for all $\mathbf{x} \neq \mathbf{0}$ (all $\lambda_i \gt 0$). It is PSD ($A \succeq 0$) if $\mathbf{x}^T A \mathbf{x} \ge 0$ (all $\lambda_i \ge 0$, allowing zero).
37. **Q:** Why is any sample covariance matrix $\Sigma$ guaranteed to be Positive Semidefinite?
    * **A:** Because for any vector $\mathbf{x}$, the quadratic form $\mathbf{x}^T \Sigma \mathbf{x} = \frac{1}{n-1} \|X_c \mathbf{x}\|_2^2 \ge 0$. The squared Euclidean norm is non-negative, so variance along any projection can never be negative.
38. **Q:** What is the Moore-Penrose Pseudoinverse $A^+$ and when is it used?
    * **A:** A generalized matrix inverse existing for any matrix. For overdetermined systems, $A^+ \mathbf{b}$ yields the Ordinary Least Squares solution. For underdetermined systems, it finds the unique solution with minimum $L_2$ norm.
39. **Q:** When is $A^+ = (A^T A)^{-1} A^T$ valid vs. the general SVD formulation $A^+ = V \Sigma^+ U^T$?
    * **A:** The formula $(A^T A)^{-1} A^T$ requires $A$ to have full column rank so $A^T A$ is invertible. The SVD formulation $A^+ = V \Sigma^+ U^T$ works universally for any matrix of arbitrary shape and rank.
40. **Q:** How does viewing a matrix as a linear transformation explain neural network dense layers?
    * **A:** A dense layer $\mathbf{z} = W\mathbf{x} + \mathbf{b}$ is an affine transformation: weight matrix $W$ linearly transforms (rotates, scales, shears) feature space, and bias vector $\mathbf{b}$ translates the coordinate origin.

---

# PART 22 — FOUR FUNDAMENTAL SUBSPACES

---

## 22.1 Column Space

```
                   COLUMN SPACE C(A) in R^m
        (All linear combinations of the columns of A)
                             y
                             │          Col(A) = span([1, 2]^T)
                             │         ╱
                           2 ┼────────● [1, 2]^T
                             │       ╱
                             │      ╱
                             └─────┴─────► x
                                   1
```

* **Definition:** The **Column Space** (or Range) of an $m \times n$ matrix $A$, denoted $C(A)$ or $\text{Col}(A)$, is the subspace of $\mathbb{R}^m$ spanned by the column vectors of $A$:

$$
C(A) = \left\lbrace A\mathbf{x} \mid \mathbf{x} \in \mathbb{R}^n \right\rbrace \subseteq \mathbb{R}^m
$$

* **Geometric Meaning:** $C(A)$ contains every output vector $\mathbf{b}$ that can be reached by multiplying $A$ by some input vector $\mathbf{x}$. The linear system $A\mathbf{x} = \mathbf{b}$ is solvable if and only if $\mathbf{b} \in C(A)$.
* **Dimension:** $\dim(C(A)) = r = \text{rank}(A)$.

### Hand Calculation Example
Let $A \in \mathbb{R}^{2 \times 2}$ be the matrix:

$$
A =
\begin{bmatrix}
1 & 2 \\
2 & 4
\end{bmatrix}
$$

* Column 1: $\mathbf{a}_1 = [1, 2]^T$, Column 2: $\mathbf{a}_2 = [2, 4]^T = 2\mathbf{a}_1$.
* Column 2 is a scalar multiple of Column 1 (linearly dependent).
* Therefore, the Column Space is the 1D line in $\mathbb{R}^2$ spanned by $[1, 2]^T$:

$$
C(A) = \text{span}\left( \begin{bmatrix} 1 \\ 2 \end{bmatrix} \right)
$$

* **Dimension:** $r = 1$.

---

## 22.2 Row Space

* **Definition:** The **Row Space** of $A$, denoted $C(A^T)$ or $\text{Row}(A)$, is the subspace of $\mathbb{R}^n$ spanned by the row vectors of $A$ (or the columns of $A^T$):

$$
C(A^T) = \left\lbrace A^T \mathbf{y} \mid \mathbf{y} \in \mathbb{R}^m \right\rbrace \subseteq \mathbb{R}^n
$$

* **Key Fundamental Theorem:** The Row Space and Column Space **always have the exact same dimension**, equal to the rank of the matrix:

$$
\dim(C(A^T)) = \dim(C(A)) = r = \text{rank}(A)
$$

### Hand Calculation Example
For the matrix $A$ defined above:
* Row 1: $[1, 2]$, Row 2: $[2, 4] = 2[1, 2]$.
* The Row Space is the 1D line in $\mathbb{R}^2$ spanned by $[1, 2]^T$:

$$
C(A^T) = \text{span}\left( \begin{bmatrix} 1 \\ 2 \end{bmatrix} \right)
$$

---

## 22.3 Null Space

* **Definition:** The **Null Space** (or Kernel) of $A$, denoted $N(A)$, is the set of all input vectors $\mathbf{x} \in \mathbb{R}^n$ that $A$ maps to the zero vector $\mathbf{0}$:

$$
N(A) = \left\lbrace \mathbf{x} \in \mathbb{R}^n \mid A\mathbf{x} = \mathbf{0} \right\rbrace \subseteq \mathbb{R}^n
$$

* **Geometric Meaning:** $N(A)$ represents all directions in the input space that are completely squashed/destroyed by the matrix transformation $A$.
* **Dimension:** $\dim(N(A)) = n - r$ (the **nullity**).

### Step-by-Step Hand Calculation Example
Find the Null Space basis for $A$:
1. Set up the homogeneous equation $A\mathbf{x} = \mathbf{0}$:

$$
\begin{bmatrix}
1 & 2 \\
2 & 4
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix} =
\begin{bmatrix}
0 \\
0
\end{bmatrix}
$$

2. Write the scalar equations:
   * Equation 1: $1x_1 + 2x_2 = 0 \implies x_1 = -2x_2$.
   * Equation 2: $2x_1 + 4x_2 = 0 \implies 2(-2x_2) + 4x_2 = 0$ (redundant $0=0$).
3. Identify the free variable:
   * $x_2$ is free. Let $x_2 = t$ where $t \in \mathbb{R}$.
4. Express the solution vector:

$$
\mathbf{x} =
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix} =
\begin{bmatrix}
-2t \\
t
\end{bmatrix} =
t
\begin{bmatrix}
-2 \\
1
\end{bmatrix}
$$

5. **Null Space Basis:**

$$
N(A) = \text{span}\left( \begin{bmatrix} -2 \\ 1 \end{bmatrix} \right)
$$

* **Dimension:** $\text{nullity}(A) = 1$.

---

## 22.4 Left Null Space

* **Definition:** The **Left Null Space** of $A$, denoted $N(A^T)$, is the null space of the transpose matrix $A^T$:

$$
N(A^T) = \left\lbrace \mathbf{y} \in \mathbb{R}^m \mid A^T \mathbf{y} = \mathbf{0} \right\rbrace \subseteq \mathbb{R}^m
$$

* *(Why "Left"? Because transposing gives $\mathbf{y}^T A = \mathbf{0}^T$, so $\mathbf{y}^T$ multiplies $A$ from the left).*
* **Dimension:** $\dim(N(A^T)) = m - r$.

### Step-by-Step Hand Calculation Example
For the matrix $A$ (where $A^T = A$ due to symmetry):
1. Set up $A^T \mathbf{y} = \mathbf{0}$:

$$
\begin{bmatrix}
1 & 2 \\
2 & 4
\end{bmatrix}
\begin{bmatrix}
y_1 \\
y_2
\end{bmatrix} =
\begin{bmatrix}
0 \\
0
\end{bmatrix}
\implies y_1 + 2y_2 = 0 \implies y_1 = -2y_2
$$

2. Let $y_2 = s$ (free variable):

$$
\mathbf{y} = s
\begin{bmatrix}
-2 \\
1
\end{bmatrix} \implies N(A^T) = \text{span}\left( \begin{bmatrix} -2 \\ 1 \end{bmatrix} \right)
$$

---

## 22.5 The Four Fundamental Subspaces (Big Picture)

```
        INPUT SPACE R^n (n dimensions)                     OUTPUT SPACE R^m (m dimensions)
   ┌─────────────────────────────────────┐            ┌─────────────────────────────────────┐
   │                                     │            │                                     │
   │           ROW SPACE                 │            │          COLUMN SPACE               │
   │           C(A^T)                    │   ──A──►   │          C(A)                       │
   │         Dimension = r               │            │        Dimension = r                │
   │                                     │            │                                     │
   │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │            │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
   │           NULL SPACE                │            │       LEFT NULL SPACE               │
   │           N(A)                      │   ──A──►   │          N(A^T)                     │
   │        Dimension = n - r            │   maps to  │        Dimension = m - r            │
   │                                     │   vector 0 │                                     │
   └─────────────────────────────────────┘            └─────────────────────────────────────┘
        Orthogonal Complements:                            Orthogonal Complements:
           C(A^T) ⊥ N(A)                                      C(A) ⊥ N(A^T)
```

| Subspace | Notation | Ambient Space | Dimension | Orthogonal Complement |
| :--- | :--- | :--- | :--- | :--- |
| **Column Space** | $C(A)$ | $\mathbb{R}^m$ | $r$ | $N(A^T)$ |
| **Row Space** | $C(A^T)$ | $\mathbb{R}^n$ | $r$ | $N(A)$ |
| **Null Space** | $N(A)$ | $\mathbb{R}^n$ | $n - r$ | $C(A^T)$ |
| **Left Null Space** | $N(A^T)$ | $\mathbb{R}^m$ | $m - r$ | $C(A)$ |

### Orthogonality Verification
Notice that any vector in the Row Space $C(A^T)$ is perpendicular to any vector in the Null Space $N(A)$:
* Row vector: $\mathbf{r} = [1, 2]^T \in C(A^T)$.
* Null vector: $\mathbf{n} = [-2, 1]^T \in N(A)$.
* Dot product: $\mathbf{r} \cdot \mathbf{n} = (1)(-2) + (2)(1) = -2 + 2 = 0 \implies \mathbf{r} \perp \mathbf{n} \quad \checkmark$.

---

## 22.6 Rank-Nullity Theorem

For any matrix $A \in \mathbb{R}^{m \times n}$ with $n$ columns:

$$
\text{rank}(A) + \text{nullity}(A) = n
$$

* **Intuition:** The total number of columns ($n$) is divided between:
  1. Directions that produce meaningful outputs ($\text{rank} = r$).
  2. Directions that are squashed into zero ($\text{nullity} = n - r$).
* *Concrete Check:* For our $2 \times 2$ matrix $A$, $\text{rank}(A) = 1$, $\text{nullity}(A) = 1$, and $n = 2$:
  $1 + 1 = 2 \quad \checkmark$.

---

## 22.7 ML Connection

* **Redundant Features & Multicollinearity:** If a dataset has $n = 100$ features but $\text{rank}(X) = 80$, then $\text{nullity}(X) = 20$. Exactly 20 feature combinations lie in the Null Space $N(X)$, meaning they have zero predictive power and make $(X^T X)$ singular.
* **Least Squares Projection** ($\mathbf{b} = \mathbf{p} + \mathbf{e}$): In linear regression, the target $\mathbf{y} \in \mathbb{R}^m$ rarely lies in the Column Space $C(X)$. The model decomposes $\mathbf{y}$ into:
  * Prediction $\hat{\mathbf{y}} = \mathbf{p} \in C(X)$ (orthogonal projection onto column space).
  * Residual error $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} \in N(X^T)$ (perpendicular error living in the Left Null Space!).

> [!TIP]
> **Common Interview Question:** *"Where does the residual error vector $\mathbf{e} = \mathbf{y} - X\mathbf{w}$ live in the four fundamental subspaces?"*
> **Answer:** It lives in the **Left Null Space** $N(X^T)$, because $\mathbf{e}$ is perpendicular to every column of $X$, satisfying $X^T \mathbf{e} = \mathbf{0}$.

> [!WARNING]
> **Common Mistake:** Confusing the ambient spaces: the Column Space $C(A)$ lives in $\mathbb{R}^m$ (output space), while the Row Space $C(A^T)$ and Null Space $N(A)$ live in $\mathbb{R}^n$ (input space).

---

# PART 23 — GRAM-SCHMIDT ORTHOGONALIZATION & QR DECOMPOSITION

---

## 23.1 Why We Need Orthogonal Bases

When basis vectors are mutually orthogonal ($\mathbf{q}_i \cdot \mathbf{q}_j = 0 \text{ for } i \neq j$) and normalized ($\|\mathbf{q}_i\|_2 = 1$):
1. **No Cross-Talk:** Projecting a vector onto coordinate axis $i$ is completely independent of coordinate axis $j$:

$$
c_i = \mathbf{x} \cdot \mathbf{q}_i
$$

2. **Trivial Matrix Inversion:** Any orthogonal matrix satisfies:

$$
Q^T Q = I \implies Q^{-1} = Q^T
$$

3. **Numerical Precision:** Orthogonal transformations preserve $L_2$ lengths and do not amplify floating-point rounding errors.

---

## 23.2 Gram-Schmidt Derivation

The **Gram-Schmidt Process** converts a set of linearly independent vectors $\mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_k$ into an orthonormal set $\mathbf{q}_1, \mathbf{q}_2, \dots, \mathbf{q}_k$.

```
        VECTOR 1: Set u1 = a1                VECTOR 2: Subtract projection onto u1
                  y                                           y
                  │     a1 = u1                               │     a2
                  │    ╱                                      │    ╱│
                1 ┼───●                                     1 ┼───● │  u2 = a2 - proj_u1(a2)
                  │  ╱                                        │   │ └──► (Orthogonal to u1!)
                  └──┴─────► x                                └───┴────► x
                     1                                            1
```

### Complete Step-by-Step Hand Calculation Example
Orthonormalize the two vectors:

$$
\mathbf{a}_1 =
\begin{bmatrix}
1 \\
1
\end{bmatrix},
\quad
\mathbf{a}_2 =
\begin{bmatrix}
1 \\
0
\end{bmatrix}
$$

---

### Step 1: Set the First Orthogonal Vector $\mathbf{u}_1$

$$
\mathbf{u}_1 = \mathbf{a}_1 =
\begin{bmatrix}
1 \\
1
\end{bmatrix}
$$

---

### Step 2: Compute the Second Orthogonal Vector $\mathbf{u}_2$
Subtract the projection of $\mathbf{a}_2$ onto $\mathbf{u}_1$:

$$
\mathbf{u}_2 = \mathbf{a}_2 - \text{proj}_{\mathbf{u}_1}(\mathbf{a}_2) = \mathbf{a}_2 - \left(\frac{\mathbf{a}_2 \cdot \mathbf{u}_1}{\mathbf{u}_1 \cdot \mathbf{u}_1}\right) \mathbf{u}_1
$$

1. Dot products:
   * $\mathbf{a}_2 \cdot \mathbf{u}_1 = (1)(1) + (0)(1) = 1$.
   * $\mathbf{u}_1 \cdot \mathbf{u}_1 = 1^2 + 1^2 = 2$.
2. Projection fraction: $\frac{\mathbf{a}_2 \cdot \mathbf{u}_1}{\mathbf{u}_1 \cdot \mathbf{u}_1} = \frac{1}{2}$.
3. Subtract:

$$
\mathbf{u}_2 =
\begin{bmatrix}
1 \\
0
\end{bmatrix} - \frac{1}{2}
\begin{bmatrix}
1 \\
1
\end{bmatrix} =
\begin{bmatrix}
1 - 1/2 \\
0 - 1/2
\end{bmatrix} =
\begin{bmatrix}
1/2 \\
-1/2
\end{bmatrix}
$$

4. **Orthogonality Check:**
   $\mathbf{u}_1 \cdot \mathbf{u}_2 = (1)(1/2) + (1)(-1/2) = 1/2 - 1/2 = 0 \quad \checkmark$.

---

### Step 3: Normalize Both Vectors to Unit Length ($L_2 = 1$)
1. Normalize $\mathbf{u}_1$:
   * $\|\mathbf{u}_1\|_2 = \sqrt{1^2 + 1^2} = \sqrt{2}$.

$$
\mathbf{q}_1 = \frac{\mathbf{u}_1}{\|\mathbf{u}_1\|_2} =
\begin{bmatrix}
1/\sqrt{2} \\
1/\sqrt{2}
\end{bmatrix}
$$

2. Normalize $\mathbf{u}_2$:
   * $\|\mathbf{u}_2\|_2 = \sqrt{(1/2)^2 + (-1/2)^2} = \sqrt{1/4 + 1/4} = \sqrt{2/4} = \frac{1}{\sqrt{2}}$.

$$
\mathbf{q}_2 = \frac{\mathbf{u}_2}{\|\mathbf{u}_2\|_2} = \sqrt{2}
\begin{bmatrix}
1/2 \\
-1/2
\end{bmatrix} =
\begin{bmatrix}
1/\sqrt{2} \\
-1/\sqrt{2}
\end{bmatrix}
$$

---

## 23.3 Build Q and R ($A = QR$)

Any matrix $A$ with linearly independent columns can be factored into:

$$
A = QR
$$

Where:
* $Q$: An **orthogonal matrix** ($Q^T Q = I$) containing the orthonormal basis vectors as columns.
* $R$: An **upper-triangular matrix** ($R_{ij} = \mathbf{q}_i^T \mathbf{a}_j$, with $R_{ij} = 0$ for $i \gt j$) storing the projection coefficients.

### Constructing $Q$ and $R$ from our Hand Example:
1. Form $Q$:

$$
Q =
\begin{bmatrix}
\mathbf{q}_1 & \mathbf{q}_2
\end{bmatrix} =
\begin{bmatrix}
1/\sqrt{2} & 1/\sqrt{2} \\
1/\sqrt{2} & -1/\sqrt{2}
\end{bmatrix}
$$

2. Compute entries of $R = Q^T A$:
   * $R_{11} = \mathbf{q}_1 \cdot \mathbf{a}_1 = (1/\sqrt{2})(1) + (1/\sqrt{2})(1) = 2/\sqrt{2} = \sqrt{2}$.
   * $R_{12} = \mathbf{q}_1 \cdot \mathbf{a}_2 = (1/\sqrt{2})(1) + (1/\sqrt{2})(0) = 1/\sqrt{2}$.
   * $R_{21} = \mathbf{q}_2 \cdot \mathbf{a}_1 = (1/\sqrt{2})(1) + (-1/\sqrt{2})(1) = 0$.
   * $R_{22} = \mathbf{q}_2 \cdot \mathbf{a}_2 = (1/\sqrt{2})(1) + (-1/\sqrt{2})(0) = 1/\sqrt{2}$.

$$
R =
\begin{bmatrix}
\sqrt{2} & 1/\sqrt{2} \\
0 & 1/\sqrt{2}
\end{bmatrix}
$$

3. **Verification** ($A = QR$):

$$
QR =
\begin{bmatrix}
1/\sqrt{2} & 1/\sqrt{2} \\
1/\sqrt{2} & -1/\sqrt{2}
\end{bmatrix}
\begin{bmatrix}
\sqrt{2} & 1/\sqrt{2} \\
0 & 1/\sqrt{2}
\end{bmatrix} =
\begin{bmatrix}
1 + 0 & 1/2 + 1/2 \\
1 + 0 & 1/2 - 1/2
\end{bmatrix} =
\begin{bmatrix}
1 & 1 \\
1 & 0
\end{bmatrix} = A \quad \checkmark
$$

---

## 23.4 Why QR Matters in ML

In Ordinary Least Squares, the Normal Equation is $X^T X \mathbf{w} = X^T \mathbf{y}$.
1. Substitute $X = QR$:

$$
(QR)^T (QR) \mathbf{w} = (QR)^T \mathbf{y} \implies R^T (Q^T Q) R \mathbf{w} = R^T Q^T \mathbf{y}
$$

2. Since $Q^T Q = I$:

$$
R^T R \mathbf{w} = R^T Q^T \mathbf{y} \implies R \mathbf{w} = Q^T \mathbf{y}
$$

3. **Why this is superior to** $(X^T X)^{-1}$:
   * Because $R$ is **upper-triangular**, $R\mathbf{w} = Q^T \mathbf{y}$ is solved instantly using **back-substitution** without inverting any matrices!
   * Forming $X^T X$ squares the condition number: $\kappa(X^T X) = \kappa(X)^2$. If $\kappa(X) = 10^4$, then $\kappa(X^T X) = 10^8$, ruining numerical accuracy. QR solves least squares with condition number $\kappa(X)$, providing maximum numerical stability.

> [!TIP]
> **Common Interview Question:** *"Why do production machine learning libraries use QR decomposition to solve Ordinary Least Squares rather than inverting $X^T X$ directly?"*
> **Answer:** Explicitly forming $X^T X$ squares the condition number ($\kappa(X^T X) = \kappa(X)^2$), which magnifies floating-point roundoff errors and can cause numerical singularity. QR decomposition solves $R\mathbf{w} = Q^T \mathbf{y}$ via simple back-substitution with condition number $\kappa(X)$, providing maximum numerical stability without computing any matrix inverse.

> [!WARNING]
> **Common Mistake:** Forgetting that Gram-Schmidt produces **orthogonal** vectors ($\mathbf{u}_k$) that must be individually normalized by dividing by their $L_2$ norm ($\mathbf{q}_k = \mathbf{u}_k / \|\mathbf{u}_k\|_2$) to form an **orthonormal** matrix $Q$ where $Q^T Q = I$.

---

# PART 24 — POSITIVE DEFINITE & POSITIVE SEMIDEFINITE MATRICES

---

## 24.1 Quadratic Form

A **Quadratic Form** is a scalar polynomial where every term has degree 2, generated by a symmetric matrix $A \in \mathbb{R}^{n \times n}$:

$$
f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} = \sum_{i=1}^{n} \sum_{j=1}^{n} A_{ij} x_i x_j
$$

### Hand Calculation Example
Let $\mathbf{x} = [x_1, x_2]^T$ and $A$ be the symmetric matrix:

$$
A =
\begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}
$$

1. Compute $A\mathbf{x}$:

$$
A\mathbf{x} =
\begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix} =
\begin{bmatrix}
2x_1 + x_2 \\
x_1 + 2x_2
\end{bmatrix}
$$

2. Multiply by $\mathbf{x}^T$:

$$
\mathbf{x}^T (A\mathbf{x}) =
\begin{bmatrix}
x_1 & x_2
\end{bmatrix}
\begin{bmatrix}
2x_1 + x_2 \\
x_1 + 2x_2
\end{bmatrix} = x_1(2x_1 + x_2) + x_2(x_1 + 2x_2) = 2x_1^2 + 2x_1 x_2 + 2x_2^2
$$

---

## 24.2 Positive Definite ($A \succ 0$)

* **Definition:** A symmetric matrix $A$ is **Positive Definite** if its quadratic form is strictly positive for every non-zero vector $\mathbf{x} \neq \mathbf{0}$:

$$
\mathbf{x}^T A \mathbf{x} \gt 0 \quad \forall \mathbf{x} \neq \mathbf{0} \quad \iff \quad \text{All eigenvalues } \lambda_i \gt 0
$$

### Hand Proof for Positive Definiteness of Matrix $A$:
Complete the square on the quadratic form:

$$
2x_1^2 + 2x_1 x_2 + 2x_2^2 = 2\left(x_1^2 + x_1 x_2 + \frac{1}{4}x_2^2\right) + \frac{3}{2}x_2^2 = 2\left(x_1 + \frac{1}{2}x_2\right)^2 + \frac{3}{2}x_2^2
$$

* Since both terms are squares multiplied by positive numbers, the sum is strictly $\gt 0$ for any $(x_1, x_2) \neq (0, 0)$.
* Eigenvalue check: $\lambda_1 = 3 \gt 0, \lambda_2 = 1 \gt 0 \implies A \succ 0$.

---

## 24.3 Positive Semidefinite ($A \succeq 0$)

* **Definition:** A symmetric matrix $A$ is **Positive Semidefinite** if its quadratic form is non-negative for all vectors:

$$
\mathbf{x}^T A \mathbf{x} \ge 0 \quad \forall \mathbf{x} \quad \iff \quad \text{All eigenvalues } \lambda_i \ge 0 \text{ (zeros allowed)}
$$

* *Example:* For the symmetric matrix:

$$
B =
\begin{bmatrix}
1 & 1 \\
1 & 1
\end{bmatrix}
$$

The quadratic form is $\mathbf{x}^T B \mathbf{x} = x_1^2 + 2x_1 x_2 + x_2^2 = (x_1 + x_2)^2 \ge 0$.
* For $\mathbf{x} = [1, -1]^T \neq \mathbf{0}$, $\mathbf{x}^T B \mathbf{x} = (1 - 1)^2 = 0$.
* Eigenvalues: $\lambda_1 = 2, \lambda_2 = 0 \implies B \succeq 0$.

---

## 24.4 Negative Definite and Indefinite

| Matrix Class | Condition on $\mathbf{x}^T A \mathbf{x}$ | Condition on Eigenvalues | Geometric Shape of Surface |
| :--- | :--- | :--- | :--- |
| **Positive Definite** ($A \succ 0$) | $\mathbf{x}^T A \mathbf{x} \gt 0$ ($\mathbf{x} \neq \mathbf{0}$) | All $\lambda_i \gt 0$ | Upward bowl (strict minimum) |
| **Positive Semidefinite** ($A \succeq 0$) | $\mathbf{x}^T A \mathbf{x} \ge 0$ | All $\lambda_i \ge 0$ | Flat-bottomed trough/valley |
| **Negative Definite** ($A \prec 0$) | $\mathbf{x}^T A \mathbf{x} \lt 0$ ($\mathbf{x} \neq \mathbf{0}$) | All $\lambda_i \lt 0$ | Downward dome (strict maximum) |
| **Indefinite** | Positive for some $\mathbf{x}$, negative for others | Mixed positive & negative $\lambda_i$ | **Saddle point** (mountain pass) |

---

## 24.5 Geometric Intuition

```
     POSITIVE DEFINITE (Bowl)               INDEFINITE (Saddle Point)
               z                                       z
               │      (Min at origin)                  │        (Up in x, Down in y)
               │    )                          ╭───────┼───────╮
               │   (                           │       │       │
               └───┴──────► x                  └───●───┴───────┴──► x
                  y                                (Saddle)
```

---

## 24.6 ML Connections

1. **Why Covariance Matrices are Always PSD:**
   For any mean-centered data matrix $X_c$ and any vector $\mathbf{u}$:

$$
\mathbf{u}^T \Sigma \mathbf{u} = \mathbf{u}^T \left(\frac{1}{n-1} X_c^T X_c\right) \mathbf{u} = \frac{1}{n-1} (X_c \mathbf{u})^T (X_c \mathbf{u}) = \frac{1}{n-1} \|X_c \mathbf{u}\|_2^2 \ge 0
$$

   *Because the squared $L_2$ norm of any vector is non-negative, the variance of projected data can NEVER be negative!*

2. **Hessian Matrix & Convex Optimization:**
   * If the Hessian matrix $H = \nabla^2 \mathcal{L}(\mathbf{w})$ is Positive Definite ($H \succ 0$), the loss surface is strictly **convex** (a bowl), guaranteeing that any stationary point ($\nabla \mathcal{L} = \mathbf{0}$) is a **unique global minimum**.
   * If $H$ is Indefinite, the optimizer is stuck at a **saddle point**.

> [!TIP]
> **Common Interview Question:** *"Why is every sample covariance matrix $\Sigma = \frac{1}{n-1} X_c^T X_c$ guaranteed to be Positive Semidefinite?"*
> **Answer:** For any test vector $\mathbf{x} \in \mathbb{R}^p$, the quadratic form evaluates to $\mathbf{x}^T \Sigma \mathbf{x} = \frac{1}{n-1} \|X_c \mathbf{x}\|_2^2$. Because the squared Euclidean norm of any real vector is strictly non-negative ($\ge 0$), the variance of the projected data along $\mathbf{x}$ can never be negative.

> [!WARNING]
> **Common Mistake:** Confusing **Positive Definite** ($A \succ 0$, all $\lambda_i \gt 0$, strictly positive quadratic form $\mathbf{x}^T A \mathbf{x} \gt 0$) with **Positive Semidefinite** ($A \succeq 0$, all $\lambda_i \ge 0$, allows zero eigenvalues). If a covariance matrix has a zero eigenvalue ($\lambda = 0$), it is PSD but NOT PD, meaning at least one redundant feature direction has zero variance.

---

# PART 25 — MOORE-PENROSE PSEUDOINVERSE

---

## 25.1 Why Ordinary Inverse Fails

The standard matrix inverse $A^{-1}$ only exists if $A$ is **square** ($n \times n$) and **full rank** ($\det(A) \neq 0$). It fails when:
1. Matrix is **rectangular** ($m \times n$, e.g., $1000$ samples $\times 10$ features).
2. Matrix is **square but singular** ($\det(A) = 0$, redundant features).

---

## 25.2 What the Pseudoinverse Does

The **Moore-Penrose Pseudoinverse** $A^+$ is the unique matrix satisfying the 4 Moore-Penrose conditions:
1. $A A^+ A = A$
2. $A^+ A A^+ = A^+$
3. $(A A^+)^T = A A^+$
4. $(A^+ A)^T = A^+ A$

* **Intuition:** $A^+$ provides the "best possible inverse":
  * For **overdetermined systems** ($m \gt n$, more equations than unknowns), $A^+ \mathbf{b}$ gives the least-squares solution minimizing $\|\mathbf{y} - X\mathbf{w}\|_2^2$.
  * For **underdetermined systems** ($m \lt n$, infinitely many solutions), $A^+ \mathbf{b}$ picks the unique solution with the minimum $L_2$ norm ($\|\mathbf{w}\|_2$).

---

## 25.3 Least Squares Connection

$$
\mathbf{w}_{\text{LS}} = A^+ \mathbf{b}
$$

* When $A$ has full column rank ($\text{rank}(A) = n$), the pseudoinverse formula is:

$$
A^+ = (A^T A)^{-1} A^T
$$

* When $A$ has full row rank ($\text{rank}(A) = m$), the right-inverse formula is:

$$
A^+ = A^T (A A^T)^{-1}
$$

* **Universal SVD Formulation:** For ANY matrix of any rank, SVD $A = U \Sigma V^T$ gives:

$$
A^+ = V \Sigma^+ U^T
$$

  where $\Sigma^+$ transposes $\Sigma$ and inverts all non-zero singular values ($\sigma_i \to 1/\sigma_i$).

---

## 25.4 Complete Step-by-Step Hand Calculation

Find the pseudoinverse of the rectangular column vector $A = [1, 2]^T \in \mathbb{R}^{2 \times 1}$:
1. Since $A$ has full column rank ($r = 1 = n$), use $A^+ = (A^T A)^{-1} A^T$:
2. Compute $A^T A$:

$$
A^T A =
\begin{bmatrix}
1 & 2
\end{bmatrix}
\begin{bmatrix}
1 \\
2
\end{bmatrix} = 1^2 + 2^2 = 5
$$

3. Invert scalar $(A^T A)^{-1} = \frac{1}{5} = 0.2$.
4. Multiply by $A^T$:

$$
A^+ = \frac{1}{5}
\begin{bmatrix}
1 & 2
\end{bmatrix} =
\begin{bmatrix}
0.2 & 0.4
\end{bmatrix}
$$

5. **Verify Moore-Penrose Property** ($A^+ A = I_1$):

$$
A^+ A =
\begin{bmatrix}
0.2 & 0.4
\end{bmatrix}
\begin{bmatrix}
1 \\
2
\end{bmatrix} = 0.2(1) + 0.4(2) = 0.2 + 0.8 = 1.0 = I_1 \quad \checkmark
$$

6. **Solve Least Squares Problem:**
   Let target vector $\mathbf{b} = [3, 1]^T$. Find best scalar $x$:

$$
\hat{x} = A^+ \mathbf{b} =
\begin{bmatrix}
0.2 & 0.4
\end{bmatrix}
\begin{bmatrix}
3 \\
1
\end{bmatrix} = 0.2(3) + 0.4(1) = 0.6 + 0.4 = \mathbf{1.0}
$$

   * Fitted point: $A\hat{x} = [1, 2]^T(1) = [1, 2]^T$.
   * Residual error: $\mathbf{e} = \mathbf{b} - A\hat{x} = [3-1, 1-2]^T = [2, -1]^T$.
   * Orthogonality check: $\mathbf{e} \cdot A = 2(1) + (-1)(2) = 0 \quad \checkmark$.

---

## 25.5 ML Applications

* **Linear Regression:** Directly computes $\mathbf{w} = X^+ \mathbf{y}$ even if $X$ contains collinear columns.
* **Ridge Regularization Connection:** As regularization $\alpha \to 0$, Ridge solution $(X^T X + \alpha I)^{-1} X^T \mathbf{y}$ converges smoothly to the minimum-norm pseudoinverse solution $X^+ \mathbf{y}$.

> [!TIP]
> **Common Interview Question:** *"When is the formula $A^+ = (A^T A)^{-1} A^T$ valid, and what should you use if $A$ does not have full column rank?"*
> **Answer:** The normal-equation formula $(A^T A)^{-1} A^T$ is valid only when $A$ has full column rank ($A^T A$ is non-singular and invertible). When $A$ is rank-deficient or has collinear columns, you must use the universal SVD formulation $A^+ = V \Sigma^+ U^T$, which inverts only the non-zero singular values ($1/\sigma_i$) and sets zero singular values to zero.

> [!WARNING]
> **Common Mistake:** Assuming $A^+ A = I$ is always the full identity matrix. For a rectangular matrix with $m \gt n$, $A^+ A = I_n$ (an $n \times n$ identity), but $A A^+ \neq I_m$ — instead, $A A^+$ is an orthogonal projection matrix onto the Column Space $C(A)$.

---

# PART 26 — LINEAR TRANSFORMATIONS

---

## 26.1 Definition

A **Transformation** $T: \mathbb{R}^n \to \mathbb{R}^m$ is a rule that takes an input vector $\mathbf{x} \in \mathbb{R}^n$ and produces an output vector $T(\mathbf{x}) \in \mathbb{R}^m$.

A transformation is **Linear** if it can be represented entirely as a matrix multiplication:

$$
T(\mathbf{x}) = A\mathbf{x}
$$

```
                   LINEAR TRANSFORMATION AS MAPPING
        Input Space R^2                             Output Space R^2
               y                                           y
               │     x = [1, 2]^T                          │           A @ x = [2, 6]^T
             2 ┼────●                                    6 ┼──────────●
               │    │                                      │          │
               └───┴────► x                                └──────────┴────► x
                   1                                                  2
```

---

## 26.2 The Two Defining Properties

A transformation $T$ is linear if and only if it preserves vector addition and scalar multiplication:
1. **Additivity:** $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
2. **Homogeneity (Scalar Scaling):** $T(c\mathbf{u}) = cT(\mathbf{u})$

### Numerical Verification Example
Let $\mathbf{u} = [1, 2]^T$, $\mathbf{v} = [3, 1]^T$, $c = 4$, and:

$$
A =
\begin{bmatrix}
2 & 0 \\
0 & 3
\end{bmatrix}
$$

* $T(\mathbf{u}) = A\mathbf{u} = [2, 6]^T$, $T(\mathbf{v}) = A\mathbf{v} = [6, 3]^T$.
* $\mathbf{u} + \mathbf{v} = [4, 3]^T \implies T(\mathbf{u} + \mathbf{v}) = A[4, 3]^T = [8, 9]^T$.
* Check Additivity: $T(\mathbf{u}) + T(\mathbf{v}) = [2+6, 6+3]^T = [8, 9]^T \quad \checkmark$.
* Check Homogeneity: $T(4\mathbf{u}) = A[4, 8]^T = [8, 24]^T = 4[2, 6]^T = 4T(\mathbf{u}) \quad \checkmark$.

---

## 26.3 Matrices as Geometric Transformations

Every 2D linear transformation is completely determined by where it sends the standard unit basis vectors $\mathbf{i} = [1, 0]^T$ and $\mathbf{j} = [0, 1]^T$:

1. **Non-Uniform Scaling (Stretch/Shrink):**

$$
A_{\text{scale}} =
\begin{bmatrix}
s_x & 0 \\
0 & s_y
\end{bmatrix}
\implies
\begin{bmatrix}
2 & 0 \\
0 & 0.5
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix} =
\begin{bmatrix}
2x \\
0.5y
\end{bmatrix}
$$

2. **Rotation by Angle** $\theta$ **(Counter-Clockwise):**

$$
R_\theta =
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
\implies R_{90^\circ} =
\begin{bmatrix}
0 & -1 \\
1 & 0
\end{bmatrix}
$$

   * Sends $\mathbf{i} = [1, 0]^T \to [0, 1]^T$ and $\mathbf{j} = [0, 1]^T \to [-1, 0]^T$.

3. **Reflection over the x-axis:**

$$
A_{\text{reflect}} =
\begin{bmatrix}
1 & 0 \\
0 & -1
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix} =
\begin{bmatrix}
x \\
-y
\end{bmatrix}
$$

4. **Horizontal Shear (Sliding Layers):**

$$
A_{\text{shear}} =
\begin{bmatrix}
1 & k \\
0 & 1
\end{bmatrix}
\implies
\begin{bmatrix}
1 & 1.5 \\
0 & 1
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix} =
\begin{bmatrix}
x + 1.5y \\
y
\end{bmatrix}
$$

---

## 26.4 Connection to Eigenvectors

* **Geometric Insight:** Under most linear transformations, vectors change both length and direction.
* **The Eigenvector Exception:** An **eigenvector** is a direction where the linear transformation $A$ acts as a pure scalar stretch without rotating the vector at all:

$$
T(\mathbf{v}) = A\mathbf{v} = \lambda \mathbf{v}
$$

---

## 26.5 Connection to SVD

Singular Value Decomposition factorizes ANY linear transformation $A = U \Sigma V^T$ into three pure geometric actions:

$$
\mathbf{x} \quad \xrightarrow{\quad V^T \quad} \quad \mathbf{x}' \quad \xrightarrow{\quad \Sigma \quad} \quad \mathbf{x}'' \quad \xrightarrow{\quad U \quad} \quad A\mathbf{x}
$$

1. **Rotate / Reflect** ($V^T$): Aligns the input coordinate system with the principal axes.
2. **Scale** ($\Sigma$): Stretches or compresses along each principal axis by singular value factors $\sigma_i$.
3. **Second Rotation** ($U$): Rotates the resulting ellipse into the target coordinate space.

---

## 26.6 ML Connection

* **Neural Network Dense Layers:** A fully connected neural network layer $\mathbf{z} = W\mathbf{x} + \mathbf{b}$ is an **affine transformation**: the weight matrix $W$ performs a linear transformation on the input features, and the bias vector $\mathbf{b}$ translates the origin. Non-linear activation functions (ReLU, GELU) then warp the space to learn non-linear decision boundaries.
* **Embeddings & Latent Representations:** Learned weight matrices project raw inputs into lower-dimensional latent spaces where geometric relationships (like cosine angle and Euclidean distance) encode semantic meaning.

> [!TIP]
> **Common Interview Question:** *"How does Singular Value Decomposition decompose any arbitrary linear transformation geometrically?"*
> **Answer:** SVD factorizes $A = U \Sigma V^T$ into three successive geometric operations: (1) an initial rotation/reflection in the domain by orthogonal matrix $V^T$, (2) an axis-aligned stretching/compression along coordinate axes by diagonal matrix $\Sigma$, and (3) a second rotation/reflection in the codomain by orthogonal matrix $U$.

> [!WARNING]
> **Common Mistake:** Confusing a **linear transformation** ($T(\mathbf{x}) = W\mathbf{x}$) with an **affine transformation** ($T(\mathbf{x}) = W\mathbf{x} + \mathbf{b}$). A linear transformation must map the origin to the origin ($T(\mathbf{0}) = \mathbf{0}$). A neural network dense layer is an affine transformation because the bias vector $\mathbf{b}$ shifts the origin.

---

# FINAL SECTION — ONE-PAGE MASTER FORMULA CHEAT SHEET

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               MASTER FORMULA REFERENCE SHEET                                     │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│ 1. DOT PRODUCT & ANGLE:                                                                          │
│    a · b = a^T b = sum(a_i * b_i) = ||a||_2 * ||b||_2 * cos(theta)                               │
│                                                                                                  │
│ 2. VECTOR NORMS & DISTANCES:                                                                     │
│    ||x||_2 = sqrt(sum(x_i^2))                     ||x||_1 = sum(|x_i|)                           │
│    d_Euclidean(p, q) = sqrt(sum((p_i - q_i)^2))    d_Manhattan(p, q) = sum(|p_i - q_i|)           │
│    Cosine_Similarity(u, v) = (u · v) / (||u||_2 * ||v||_2)                                       │
│                                                                                                  │
│ 3. 2x2 DETERMINANT & INVERSE:                                                                    │
│    det([a b; c d]) = ad - bc                                                                     │
│    [a b; c d]^-1 = (1 / (ad - bc)) * [d -b; -c a]       (Exists iff det != 0)                     │
│                                                                                                  │
│ 4. MATRIX TRANSPOSE RULE:                                                                        │
│    (A * B)^T = B^T * A^T                          (A * B)^-1 = B^-1 * A^-1                       │
│                                                                                                  │
│ 5. EIGENVALUES & EIGENVECTORS:                                                                   │
│    A * v = lambda * v   ===>   det(A - lambda * I) = 0                                           │
│                                                                                                  │
│ 6. SAMPLE COVARIANCE MATRIX:                                                                     │
│    Cov(X, Y) = (1 / (n - 1)) * sum((x_i - mean_x) * (y_i - mean_y))                              │
│    Sigma = (1 / (n - 1)) * X_centered^T * X_centered                                             │
│                                                                                                  │
│ 7. VECTOR PROJECTION:                                                                            │
│    proj_b(a) = ((a · b) / (b · b)) * b            P = X * (X^T * X)^-1 * X^T                      │
│                                                                                                  │
│ 8. LINEAR REGRESSION (OLS):                                                                      │
│    Loss = ||y - X*w||_2^2   ===>   Normal Eq: w = (X^T * X)^-1 * X^T * y                         │
│                                                                                                  │
│ 9. GRADIENT DESCENT:                                                                             │
│    w_(t+1) = w_t - alpha * grad_L(w_t)                                                           │
│                                                                                                  │
│ 10. MULTIVARIABLE CHAIN RULE:                                                                    │
│     dL / dw = (dL / da) * (da / dz) * (dz / dw)                                                  │
│                                                                                                  │
│ 11. REGULARIZED OBJECTIVES:                                                                      │
│     Ridge (L2): Loss = MSE + alpha * sum(w_j^2)    ==>  w = (X^T*X + alpha*I)^-1 * X^T*y         │
│     Lasso (L1): Loss = MSE + alpha * sum(|w_j|)    ==>  Promotes exact zeros (sparsity)          │
│                                                                                                  │
│ 12. SHANNON ENTROPY & GINI IMPURITY:                                                             │
│     H(S) = - sum(p_i * log2(p_i))                 Gini(S) = 1 - sum(p_i^2)                       │
│                                                                                                  │
│ 13. SINGULAR VALUE DECOMPOSITION (SVD):                                                          │
│     A = U * Sigma * V^T                           X_c^T * X_c = V * Sigma^2 * V^T                │
│                                                                                                  │
│ 14. RANK-NULLITY THEOREM:                                                                        │
│     rank(A) + nullity(A) = n (number of columns)                                                 │
│                                                                                                  │
│ 15. GRAM-SCHMIDT & QR DECOMPOSITION:                                                             │
│     A = Q * R                                     (Q is orthonormal, R is upper-triangular)      │
│                                                                                                  │
│ 16. POSITIVE DEFINITE TEST:                                                                      │
│     x^T * A * x > 0 for all x != 0               <===> all eigenvalues lambda_i > 0              │
│                                                                                                  │
│ 17. MOORE-PENROSE PSEUDOINVERSE:                                                                 │
│     A^+ = (A^T * A)^-1 * A^T (full col rank)     A^+ = V * Sigma^+ * U^T (universal SVD)         │
│                                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---