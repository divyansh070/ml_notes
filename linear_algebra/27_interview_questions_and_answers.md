> 📖 **Navigation:** [← Previous: Part 26: Linear Transformations & Change of Basis](./26_linear_transformations_and_change_of_basis.md) | [🏠 Index](./README.md) | [Next: Part 28: One-Page Master Formula Cheat Sheet →](./28_master_formula_cheat_sheet.md)

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

> 📖 **Navigation:** [← Previous: Part 26: Linear Transformations & Change of Basis](./26_linear_transformations_and_change_of_basis.md) | [🏠 Index](./README.md) | [Next: Part 28: One-Page Master Formula Cheat Sheet →](./28_master_formula_cheat_sheet.md)
