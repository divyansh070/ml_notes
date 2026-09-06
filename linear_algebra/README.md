# Linear Algebra & Mathematics for Machine Learning & Data Science
### Master Revision Guide • Visual Intuition • Hand Calculations • Step-by-Step Derivations

> **Welcome to the Modular Linear Algebra Study Guide!**
> This repository is broken down into **28 focused, standalone topic modules** designed for rapid loading, seamless GitHub KaTeX math rendering, and targeted interview preparation.
> * Every topic includes **geometric intuition**, **rigorous mathematical definitions**, and **complete hand calculations** on small matrices.
> * Grounded in premier curricula: **Prof. Gilbert Strang's MIT 18.06 / 18.065**, **Stanford CS229 Linear Algebra Review**, and Imperial College's **Mathematics for Machine Learning**.
> * Two-track design: Master the **Core Fundamentals Track** first, then explore the **Advanced / Optional Track** sections at the bottom of each module.

---

## 🎯 The Core Learning Philosophy

1. **Intuition First, Rigor Second:** Understand *what* a concept does geometrically and *why* we need it before memorizing abstract formulas.
2. **Paper-and-Pencil Mastery:** Every major formula is verified by hand on $2 \times 2$ or $3 \times 3$ matrices with real numbers.
3. **Direct Machine Learning Connections:** Every mathematical concept is directly linked to real ML models (PCA, SVD, Linear Regression, Ridge, Lasso, Neural Networks, Transformers).

---

## 🗺️ Master Table of Contents & Learning Tracks

### 🟢 Track 1: Linear Algebra Core Fundamentals
Foundational vectors, matrices, spaces, systems, and rank.

| Part | Topic / Module | Key Concepts Covered |
| :---: | :--- | :--- |
| **01** | [Vectors & Vector Spaces](./01_vectors_and_vector_spaces.md) | 3 Vector views, Addition, Dot Product, $L_1/L_2/L_p/L_\infty$ Norms, Cosine Similarity, Subspaces, Affine vs. Linear |
| **02** | [Matrices & Matrix Operations](./02_matrices_and_operations.md) | Data Tables vs. Operators, 3 Multiplication views, 5 Geometric Transformations, Transpose, Trace $\text{Tr}(A)$, Frobenius Norm |
| **03** | [Matrix Inverses & Gauss-Jordan](./03_matrix_inverses_and_gauss_jordan.md) | $2\times 2$ formula, Gauss-Jordan on $[A \mid I]$, Invertible Matrix Theorem, Why we don't invert in ML |
| **04** | [Determinants & Geometric Scaling](./04_determinants.md) | $2\times 2$ Determinant formula, Area/Volume scaling, Strang's 10 Rules, $\det(A) = \prod \lambda_i$, Dimensional collapse |
| **05** | [Systems of Linear Equations ($Ax = b$)](./05_systems_of_linear_equations.md) | Row picture vs. Column picture, 3 Solution Scenarios (Unique, Infinite, Inconsistent), Rouché-Capelli rank conditions |
| **06** | [Linear Independence & Matrix Rank](./06_linear_independence_and_rank.md) | Linear combinations, Span, Basis, Dimension, Column/Row Rank, Rank-Nullity Theorem, Multicollinearity |

---

### 🔵 Track 2: Spectral Theory, PCA & Dimensionality Reduction
Eigenvalues, eigenvectors, covariance structures, SVD, and orthogonal projections.

| Part | Topic / Module | Key Concepts Covered |
| :---: | :--- | :--- |
| **07** | [Eigenvalues & Eigenvectors](./07_eigenvalues_and_eigenvectors.md) | $A\mathbf{v} = \lambda \mathbf{v}$, Invariant directions, Complete 13-Step Hand Derivation on 2x2, Spectral Theorem ($A = Q \Lambda Q^T$) |
| **08** | [Eigenvalues in PCA](./08_eigenvalues_in_pca.md) | The Chain of Logic (Data $\to$ Covariance $\to$ Eigenvalues $\to$ Principal Components), Lagrangian variance maximization |
| **09** | [Covariance Matrix Complete Calculation](./09_covariance_matrix.md) | Variance vs. Covariance, Sum of Outer Products derivation, Bessel's $n-1$ correction, Proof of PSD property |
| **10** | [Complete PCA Walkthrough from Scratch](./10_pca_scratch_walkthrough.md) | The 6-Step PCA Algorithm, 2D to 1D numerical hand projection, 2D reconstruction, Explained variance ratio |
| **11** | [Singular Value Decomposition (SVD)](./11_singular_value_decomposition.md) | $A = U \Sigma V^T$, Geometric rotation/scaling/rotation view, PCA connection, Full/Compact/Truncated SVD |
| **12** | [Orthogonality & Orthonormal Bases](./12_orthogonality_and_bases.md) | Orthogonal vectors ($\mathbf{u} \cdot \mathbf{v} = 0$), Proof that orthogonal vectors are independent, Orthogonal matrices ($Q^T Q = I$) |
| **13** | [Vector Projections](./13_vector_projections.md) | Scalar/vector projection derivations, Projection matrices ($P = X(X^T X)^{-1} X^T$), Orthogonal complement projector $(I - P)$ |

---

### 🟡 Track 3: Machine Learning Mathematics & Optimization
Linear regression, multivariate calculus, backpropagation, and loss formulations.

| Part | Topic / Module | Key Concepts Covered |
| :---: | :--- | :--- |
| **14** | [Linear Regression Matrix Mathematics](./14_linear_regression_matrix_math.md) | $\mathbf{y} = X\mathbf{w} + \boldsymbol{\epsilon}$, Least squares derivation, Normal Equation $(X^T X)\mathbf{w} = X^T \mathbf{y}$, Residual Orthogonality |
| **15** | [Gradients & Derivatives for Optimization](./15_gradients_and_derivatives.md) | Partial derivatives, Gradient vector, Gradient Descent trace, Master Matrix Calculus Identities, Hessian & Curvature |
| **16** | [The Chain Rule & Backpropagation](./16_chain_rule_and_backprop.md) | Single/multivariable chain rule, Computational graphs, Hand trace on 2-layer neural network |
| **17** | [Distance & Similarity Metrics](./17_distance_and_similarity_metrics.md) | Euclidean ($L_2$) vs. Manhattan ($L_1$), Cosine Distance vs. Similarity, ML Selection Matrix |
| **18** | [Regularization Mathematics ($L_1$ vs. $L_2$)](./18_regularization_mathematics.md) | Geometric diamond boundary proof for $L_1$ sparsity, Circular $L_2$ decay, SVD / Spectral Shrinkage in Ridge, ElasticNet |
| **19** | [Entropy & Information Gain Mathematics](./19_entropy_and_information_gain.md) | Shannon entropy $H(X)$, 50/50 vs. Pure split, Information Gain in Decision Trees, Gini Impurity |
| **20** | [ML Mathematics Roadmap Table](./20_ml_mathematics_roadmap.md) | Comprehensive reference linking all ML algorithms to their foundational math concepts |
| **21** | [Paper-and-Pencil Exam Checklist](./21_paper_and_pencil_checklist.md) | 30 Core self-assessment challenges across 3 progressive tiers (Level A Calculate, Level B Explain, Level C Derive) |

---

### 🟣 Track 4: Advanced Linear Algebra & Matrix Decompositions
Strang's four fundamental subspaces, Gram-Schmidt QR, definiteness, pseudoinverses, and change of basis.

| Part | Topic / Module | Key Concepts Covered |
| :---: | :--- | :--- |
| **22** | [Four Fundamental Subspaces](./22_four_fundamental_subspaces.md) | $C(A), N(A), C(A^T), N(A^T)$, Dimensions, Orthogonal complements, $2\times 3$ rectangular example, Regression residuals |
| **23** | [Gram-Schmidt & QR Decomposition](./23_gram_schmidt_and_qr_decomposition.md) | Gram-Schmidt projection algorithm, Complete hand calculation, $A = QR$ factorization, Stable least squares solver |
| **24** | [Positive Definite & Semidefinite Matrices](./24_positive_definite_matrices.md) | $\mathbf{x}^T A \mathbf{x} > 0$, Strang's 5 Equivalent tests, Covariance PSD proof, Rayleigh Quotient, Cholesky & VAEs |
| **25** | [Moore-Penrose Pseudoinverse ($A^+$)](./25_moore_penrose_pseudoinverse.md) | Overdetermined & underdetermined systems, Left/Right inverses, SVD formulation $A^+ = V \Sigma^+ U^T$, Ridge limit |
| **26** | [Linear Transformations & Change of Basis](./26_linear_transformations_and_change_of_basis.md) | Linear maps, 5 Elementary 2D transformations, Change of basis matrix $P$, Matrix similarity $B = P^{-1} A P$, Diagonalization |

---

### 🔴 Track 5: Interview Preparation & Master Cheat Sheet
Complete interview preparation package with 40 questions and one-page master reference.

| Part | Topic / Module | Key Concepts Covered |
| :---: | :--- | :--- |
| **27** | [40 Essential Technical Interview Questions & Answers](./27_interview_questions_and_answers.md) | 40 In-depth technical interview questions across 3 tiers with Core Answer, Why It Matters, Geometric Picture, and ML Connection |
| **28** | [One-Page Master Formula Cheat Sheet](./28_master_formula_cheat_sheet.md) | Ultra-condensed single-page reference sheet with all formulas, properties, identities, matrix calculus rules, and SVD theorems |

---

## 💡 How to Navigate
Each topic module contains breadcrumb navigation at the top and bottom of the file:
* `[← Previous Topic]` moves to the preceding module in logical curriculum order.
* `[🏠 Index]` returns directly to this master `README.md`.
* `[Next Topic →]` advances to the next conceptual module.
