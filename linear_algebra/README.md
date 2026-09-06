# Linear Algebra & Mathematics for Machine Learning & Data Science
### Master Revision Guide • Visual Intuition • Hand Calculations • Step-by-Step Derivations

> **Welcome to the Modular Linear Algebra Study Guide!**
> This repository is broken down into **28 focused, standalone topic modules** designed for rapid loading, seamless GitHub KaTeX math rendering, and targeted interview preparation.
> * Every topic includes **geometric intuition**, **rigorous mathematical definitions**, and **complete hand calculations** on small matrices.
> * Grounded in premier curricula: **Prof. Gilbert Strang's MIT 18.06 / 18.065**, **Stanford CS229 Linear Algebra Review**, and Imperial College's **Mathematics for Machine Learning**.
> * No library abstractions—everything is derived step-by-step from first principles.

---

## 🗺️ Master Table of Contents & Learning Tracks

### 🟢 Track 1: Linear Algebra Core Fundamentals
Foundational vectors, matrices, spaces, and elementary linear operations.

| Part | Topic / Module | Key Concepts Covered |
| :---: | :--- | :--- |
| **01** | [Vectors & Vector Spaces](./01_vectors_and_vector_spaces.md) | Vectors, Scalars, Addition, Dot Product, $L_1/L_2/L_p/L_\infty$ Norms, $L_0$ Pseudo-Norm, Cauchy-Schwarz Inequality & Proof, Minkowski & Hölder Inequalities, Cosine Similarity |
| **02** | [Matrices & Matrix Operations](./02_matrices_and_operations.md) | Data Tables vs. Operators, Matrix Multiplication Deep Trace, Transpose, Identity & Diagonal, Matrix Trace $\text{Tr}(A)$ & Cyclic Properties, Orthogonal/Idempotent/Involutory/Nilpotent Matrices, Frobenius/Spectral/Nuclear Norms, Hadamard & Kronecker Products |
| **03** | [Matrix Inverses, Gauss-Jordan & Adjugate](./03_matrix_inverses_and_gauss_jordan.md) | 2x2 shortcut, 3x3 Gauss-Jordan on $[A \mid I]$, NxN Adjugate/Cofactor, Invertible Matrix Theorem, Sherman-Morrison Rank-1 Update, Woodbury Matrix Identity, Block Matrix Inversion & Schur Complement, Left vs. Right Inverses |
| **04** | [Determinants & Geometric Scaling](./04_determinants.md) | 2x2 Determinant formula, Area/Volume scaling, Strang's 10 Fundamental Determinant Rules, Eigenvalue Product $\det(A) = \prod \lambda_i$, Jacobian Determinant in Normalizing Flows, Cramer's Rule |
| **05** | [Systems of Linear Equations ($Ax = b$)](./05_systems_of_linear_equations.md) | Simultaneous equations, Matrix inversion solution, Elimination, 3 Solution Scenarios (Unique, Infinite, Inconsistent) |
| **06** | [Linear Independence & Matrix Rank](./06_linear_independence_and_rank.md) | Linear combinations, Span, Independence hand check, Column/Row Rank, Rank-Nullity Theorem, Multicollinearity |

---

### 🔵 Track 2: Spectral Theory, PCA & Dimensionality Reduction
Eigenvalues, eigenvectors, covariance structures, SVD, and orthogonal projections.

| Part | Topic / Module | Key Concepts Covered |
| :---: | :--- | :--- |
| **07** | [Eigenvalues & Eigenvectors](./07_eigenvalues_and_eigenvectors.md) | $A\mathbf{v} = \lambda \mathbf{v}$, Complete 13-Step Hand Derivation on 2x2, Spectral Theorem & Orthogonal Diagonalization ($A = Q \Lambda Q^T$), Spectral Radius $\rho(A)$ & RNN Stability, Power Iteration Algorithm, Gershgorin Circle Theorem |
| **08** | [Eigenvalues in PCA](./08_eigenvalues_in_pca.md) | The Chain of Logic (Data $\to$ Covariance $\to$ Eigenvalues $\to$ Principal Components), Rayleigh quotient maximization |
| **09** | [Covariance Matrix Complete Calculation](./09_covariance_matrix.md) | Variance vs. Covariance, Step-by-Step Hand Calculation of $\Sigma$, Bessel's $n-1$ correction, Mahalanobis Distance $D_M(\mathbf{x}, \boldsymbol{\mu})$, Total Variance $\text{Tr}(\Sigma)$ vs. Generalized Variance $\det(\Sigma)$ |
| **10** | [Complete PCA Walkthrough from Scratch](./10_pca_scratch_walkthrough.md) | The 6-Step PCA Algorithm, 2D to 1D numerical hand projection, Explained variance ratio |
| **11** | [Singular Value Decomposition (SVD)](./11_singular_value_decomposition.md) | $A = U \Sigma V^T$, Geometric rotation/scaling/rotation view, PCA connection, Four Fundamental Subspaces via SVD, Eckart-Young-Mirsky Theorem (Optimal Rank-$k$ & LoRA), Condition Number $\kappa(A)$ & Loss Geometry |
| **12** | [Orthogonality & Orthonormal Bases](./12_orthogonality_and_bases.md) | Orthogonal vectors ($\mathbf{u} \cdot \mathbf{v} = 0$), Orthonormal bases, Orthogonal matrices ($Q^T Q = I$, Isometry) |
| **13** | [Vector Projections](./13_vector_projections.md) | Scalar/vector projection derivations, Numerical hand calculation, Projection matrices ($P^T = P, P^2 = P$), Orthogonal complement projector $(I - P)$ |

---

### 🟡 Track 3: Machine Learning Mathematics & Optimization
Linear regression, multivariate calculus, backpropagation, and loss formulations.

| Part | Topic / Module | Key Concepts Covered |
| :---: | :--- | :--- |
| **14** | [Linear Regression Matrix Mathematics](./14_linear_regression_matrix_math.md) | $y = X\mathbf{w} + \epsilon$, Least squares derivation, Normal Equation $(X^T X)\mathbf{w} = X^T \mathbf{y}$, Hat Matrix $H = X(X^T X)^{-1} X^T$ & Leverage Scores, Residual Orthogonality ($X^T \mathbf{e} = \mathbf{0}$), Gauss-Markov Theorem (BLUE) |
| **15** | [Gradients & Derivatives for Optimization](./15_gradients_and_derivatives.md) | 1D Derivatives, Tangent slopes, Partial derivatives, Gradient vector, 3-Step Gradient Descent trace, 7 Master Matrix Calculus Identities, Hessian Matrix & Second-Order Taylor Expansion, Newton-Raphson Optimization |
| **16** | [The Chain Rule & Backpropagation](./16_chain_rule_and_backprop.md) | Single/multivariable chain rule, Hand trace on nested functions, Neural network computational graphs |
| **17** | [Distance & Similarity Metrics](./17_distance_and_similarity_metrics.md) | Euclidean ($L_2$) vs. Manhattan ($L_1$), Cosine Distance vs. Similarity, ML Selection Matrix |
| **18** | [Regularization Mathematics ($L_1$ vs. $L_2$)](./18_regularization_mathematics.md) | Constrained optimization, Geometric diamond boundary proof for $L_1$ sparsity, Circular $L_2$ decay, SVD / Spectral Shrinkage in Ridge, Lasso Subgradient & Soft-Thresholding Operator, ElasticNet |
| **19** | [Entropy & Information Gain Mathematics](./19_entropy_and_information_gain.md) | Shannon entropy $H(X)$, 50/50 vs. Pure split, Information Gain in Decision Trees, Gini Impurity |
| **20** | [ML Mathematics Roadmap Table](./20_ml_mathematics_roadmap.md) | Comprehensive reference linking all ML algorithms to their foundational math concepts |
| **21** | [Paper-and-Pencil Exam Checklist](./21_paper_and_pencil_checklist.md) | 20 Core self-assessment derivation challenges to test before coding exams and technical interviews |

---

### 🟣 Track 4: Advanced Linear Algebra & Matrix Decompositions
Strang's four fundamental subspaces, Gram-Schmidt QR, definiteness, pseudoinverses, and change of basis.

| Part | Topic / Module | Key Concepts Covered |
| :---: | :--- | :--- |
| **22** | [Four Fundamental Subspaces](./22_four_fundamental_subspaces.md) | $C(A), N(A), C(A^T), N(A^T)$, Dimensions, Orthogonal complements, Fundamental Theorem of Linear Algebra |
| **23** | [Gram-Schmidt & QR Decomposition](./23_gram_schmidt_and_qr_decomposition.md) | Gram-Schmidt projection algorithm, Complete hand calculation, $A = QR$ factorization, Stable solver |
| **24** | [Positive Definite & Semidefinite Matrices](./24_positive_definite_matrices.md) | $\mathbf{x}^T A \mathbf{x} > 0$, Strang's 5 Equivalent tests, Rayleigh Quotient & PCA derivation, Mercer's Theorem & Kernel Gram Matrices in SVMs, Cholesky Factorization ($A = L L^T$) & Multivariate Gaussian Sampling (VAE Reparameterization) |
| **25** | [Moore-Penrose Pseudoinverse ($A^+$)](./25_moore_penrose_pseudoinverse.md) | 4 Moore-Penrose conditions, SVD derivation $A^+ = V \Sigma^+ U^T$, Minimum-norm least squares, 2x2 trace |
| **26** | [Linear Transformations & Change of Basis](./26_linear_transformations_and_change_of_basis.md) | Linear maps, Standard matrix $[T]$, Change of basis matrix $P$, Matrix similarity $A' = P^{-1} A P$, Diagonalization |

---

### 🔴 Track 5: Interview Preparation & Master Cheat Sheet
Complete interview preparation package with 40 questions and one-page master reference.

| Part | Topic / Module | Key Concepts Covered |
| :---: | :--- | :--- |
| **27** | [40 Essential Technical Interview Questions & Answers](./27_interview_questions_and_answers.md) | 40 In-depth technical interview questions across all linear algebra and math domains with complete worked solutions |
| **28** | [One-Page Master Formula Cheat Sheet](./28_master_formula_cheat_sheet.md) | Ultra-condensed single-page reference sheet with all formulas, properties, identities, matrix calculus rules, and SVD theorems |

---

## 💡 How to Navigate
Each topic module contains breadcrumb navigation at the top and bottom of the file:
* `[← Previous Topic]` moves to the preceding module in logical curriculum order.
* `[🏠 Index]` returns directly to this master `README.md`.
* `[Next Topic →]` advances to the next conceptual module.
