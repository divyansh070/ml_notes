# Linear Algebra & Mathematics for Machine Learning & Data Science
### Master Revision Guide • Visual Intuition • Hand Calculations • Step-by-Step Derivations

> **Welcome to the Modular Linear Algebra Study Guide!**
> This repository provides a complete, readable, mathematically rigorous foundation in Linear Algebra tailored specifically for Machine Learning and Data Science.
> 
> * **Clear Progression:** Concepts build logically from vector operations to systems of equations, vector spaces, projections, spectral theory, and SVD.
> * **Balanced Depth:** Prioritizes concise, high-quality explanations and small numerical examples over abstract textbook proofs.
> * **Three-Tier Organization:**
>   * 🟢 **CORE (Parts 01–21):** Essential foundations that every ML practitioner must understand.
>   * 🔵 **ADVANCED (Parts 22–25):** Algorithmic deep dives and companion topics (useful, but can be explored after mastering the Core).
>   * 🟡 **REFERENCE (Parts 26–28):** Self-tests, interview preparation, and master cheat sheets.

---

## 🗺️ Master Curriculum Overview

```
[01 Vectors] ──► [02 Matrices] ──► [03 Systems Ax=b] ──► [04 Elimination & RREF] ──► [05 Inverses]
                                                                                            │
┌───────────────────────────────────────────────────────────────────────────────────────────┘
▼
[06 Determinants] ──► [07 Rank] ──► [08 Independence & Basis] ──► [09 Subspaces & Null Space]
                                                                            │
┌───────────────────────────────────────────────────────────────────────────┘
▼
[10 Four Subspaces] ──► [11 Transforms] ──► [12 Orthogonality] ──► [13 Projections]
                                                                           │
┌──────────────────────────────────────────────────────────────────────────┘
▼
[14 Least Squares] ──► [15 Eigenvalues] ──► [16 Diagonalization] ──► [17 Spectral Theorem]
                                                                            │
┌───────────────────────────────────────────────────────────────────────────┘
▼
[18 Positive Definite] ──► [19 SVD] ──► [20 Pseudoinverse] ──► [21 ML Synthesis]
```

---

## 🟢 CORE TRACK (Must Understand for ML: Parts 01–21)

| Part | Module | Key Topics Covered |
| :---: | :--- | :--- |
| **01** | [Vectors and Vector Spaces](./01_vectors_and_vector_spaces.md) | Vector operations, dot products, geometric cosine angle proof, $L_1, L_2, L_\infty, L_p$ norms, unit vectors, Euclidean distance |
| **02** | [Matrices and Operations](./02_matrices_and_operations.md) | Matrices as operators, row/column views of $A\mathbf{x}$, non-commutativity ($AB \neq BA$), transpose rules, trace $\operatorname{Tr}(A)$, Frobenius norm |
| **03** | [Systems of Linear Equations](./03_systems_of_linear_equations.md) | $A\mathbf{x} = \mathbf{b}$, row vs column picture, 3 solution cases (unique, infinite with free variables, inconsistent contradiction) |
| **04** | [Gaussian and Gauss-Jordan Elimination](./04_gaussian_and_gauss_jordan_elimination.md) | Forward elimination, Row Echelon Form (REF) vs Reduced REF (RREF), back-substitution, $[A \mid I] \to [I \mid A^{-1}]$ $3\times 3$ trace |
| **05** | [Matrix Inverses](./05_matrix_inverses.md) | Two-sided, left, right, and pseudoinverse distinctions; full $2\times 2$ derivation; complete $3\times 3$ adjugate/cofactor method; $A \operatorname{adj}(A) = \det(A)I$ |
| **06** | [Determinants](./06_determinants.md) | Volume/area scaling, 2D/3D determinants, Laplace expansion, triangular matrices, 6 fundamental rules, $\det(A) = \prod \lambda_i$ |
| **07** | [Matrix Rank](./07_matrix_rank.md) | Row rank = column rank, pivot counting in REF, full rank vs rank deficiency, dimension of span, solvability connections |
| **08** | [Linear Independence, Span, and Basis](./08_linear_independence_span_basis.md) | Linear combinations, span geometry, linear independence test, basis minimal spanning set, dimension, span vs basis contrast |
| **09** | [Vector Spaces and Subspaces](./09_vector_spaces_and_subspaces.md) | 8 Space axioms, 3 subspace closure requirements, linear subspaces vs affine hyperplanes, null space definition and basis calculation |
| **10** | [Four Fundamental Subspaces](./10_four_fundamental_subspaces.md) | Strang's Big Picture, $C(A), N(A), C(A^T), N(A^T)$, dimensions, Rank-Nullity theorem, fundamental orthogonality proof, $2\times 3$ trace |
| **11** | [Linear Transformations](./11_linear_transformations.md) | Definition $T(c\mathbf{u}+\mathbf{v})=cT(\mathbf{u})+T(\mathbf{v})$, 5 elementary 2D transforms, standard matrix $[T]$, linear vs affine maps, change of basis |
| **12** | [Orthogonality and Bases](./12_orthogonality_and_bases.md) | Orthogonal/orthonormal vectors, proof that orthogonal vectors are independent, coordinate dot products, orthogonal matrices $Q^T Q = I$ |
| **13** | [Vector Projections](./13_vector_projections.md) | 1D vector projection, closest-point geometry, projection matrix $P = X(X^T X)^{-1}X^T$, idempotency ($P^2 = P$) and symmetry ($P^T = P$), $I-P$ |
| **14** | [Least Squares and Linear Regression](./14_least_squares_and_linear_regression.md) | Geometric derivation via residual orthogonality ($X^T \mathbf{e} = \mathbf{0}$), normal equation $(X^T X)\mathbf{w}^* = X^T \mathbf{y}$, calculus derivation, Ridge fix |
| **15** | [Eigenvalues and Eigenvectors](./15_eigenvalues_and_eigenvectors.md) | Invariant axis intuition ($A\mathbf{v} = \lambda \mathbf{v}$), characteristic equation $\det(A - \lambda I)=0$, eigenspaces, trace/determinant identities, worked example |
| **16** | [Diagonalization](./16_diagonalization.md) | $A = P D P^{-1}$, conditions for diagonalizability, non-diagonalizable/defective matrices, matrix powers $A^k = P D^k P^{-1}$, worked $A^4$ example |
| **17** | [Symmetric Matrices and the Spectral Theorem](./17_symmetric_matrices_and_spectral_theorem.md) | 3 Spectral Theorem guarantees, $A = Q \Lambda Q^T$, rank-1 outer product spectral decomposition, worked $2\times 2$ example |
| **18** | [Positive Definite Matrices](./18_positive_definite_matrices.md) | Quadratic forms $\mathbf{x}^T A \mathbf{x}$, 4 equivalent tests (eigenvalues, minors, pivots, Cholesky), covariance PSD proof, Hessian curvature |
| **19** | [Singular Value Decomposition (SVD)](./19_singular_value_decomposition.md) | $A = U \Sigma V^T$, Rotate $\to$ Scale $\to$ Rotate geometry, Full vs Compact vs Truncated SVD, Eckart-Young Theorem, hand-computable example, PCA connection |
| **20** | [Moore-Penrose Pseudoinverse](./20_moore_penrose_pseudoinverse.md) | Universal definition via SVD ($A^+ = V \Sigma^+ U^T$), 4 Penrose conditions, left inverse shortcut, right inverse shortcut, minimum-norm solutions |
| **21** | [Linear Algebra for ML Synthesis](./21_linear_algebra_for_ml_synthesis.md) | The Three Pillars of Linear Algebra in ML, Algorithm-to-Mathematics mapping, Rosetta Stone conversion table |

---

## 🔵 ADVANCED TRACK (Useful Deep Dives: Parts 22–25)

| Part | Module | Key Topics Covered |
| :---: | :--- | :--- |
| **22** | [PCA and Covariance Walkthrough](./22_pca_and_covariance_walkthrough.md) | Sample covariance derivation, Lagrangian variance maximization, complete 9-step numerical PCA walkthrough from scratch |
| **23** | [Gram-Schmidt and QR Decomposition](./23_gram_schmidt_and_qr_decomposition.md) | Gram-Schmidt orthogonalization process, $A = QR$ factorization, numerically stable least-squares solver without squaring condition number |
| **24** | [Optimization and Derivatives](./24_optimization_and_derivatives.md) | Gradients, Hessians, matrix calculus lookup table, multivariable chain rule, backpropagation trace on 2-layer neural network |
| **25** | [Information Theory for ML](./25_information_theory_for_ml.md) | Shannon entropy, Gini impurity, decision tree information gain hand calculation, KL divergence, Cross-Entropy loss connection |

---

## 🟡 REFERENCE TRACK (Self-Tests, Interview Prep & Cheat Sheets: Parts 26–28)

| Part | Module | Key Topics Covered |
| :---: | :--- | :--- |
| **26** | [Paper & Pencil Self-Test Checklist](./26_paper_and_pencil_checklist.md) | 30 Core technical problems divided into Level A (Calculations), Level B (Geometric Intuition), and Level C (First-Principles Derivations) |
| **27** | [Interview Questions & Answers](./27_interview_questions_and_answers.md) | 20 High-impact technical interview questions with Direct Answer, Mathematical Mechanism, Geometric Picture, and ML Connection |
| **28** | [Master Formula Cheat Sheet](./28_master_formula_cheat_sheet.md) | Clean KaTeX tables and identities covering norms, inverses, determinants, subspaces, spectral theory, SVD, and matrix calculus |

---

## Minimum Skills Checklist

After completing the **Core Track (Parts 01–21)**, you should be able to perform every calculation and explain every concept listed below on paper without assistance, along with its direct Machine Learning connection:

### 1. Concrete Calculations
* [ ] **Matrix/Vector Multiplication:** Compute matrix-vector ($A\mathbf{x}$) and matrix-matrix ($AB$) products by rows, by columns, and as sums of rank-1 outer products. Verify $(AB)^T = B^T A^T$.
  * *ML Connection:* Feedforward layers ($\mathbf{z} = W\mathbf{x} + \mathbf{b}$), attention projections ($Q = X W_Q$).
* [ ] **Solve $A\mathbf{x} = \mathbf{b}$:** Set up augmented matrix $[A \mid \mathbf{b}]$ and determine whether a solution exists, is unique, or has infinite solutions.
  * *ML Connection:* Parameter estimation, equilibrium states in Markov chains.
* [ ] **Gaussian Elimination:** Perform row operations to convert a matrix into upper-triangular Row Echelon Form (REF) and use back-substitution to solve for unknown variables.
  * *ML Connection:* Foundation of LU solvers used in scientific computing.
* [ ] **Reduced Row Echelon Form (RREF):** Execute full Gauss-Jordan elimination to reach leading 1s with zeros above and below each pivot.
  * *ML Connection:* Identifying pivot variables vs. free variables; determining exact linear dependencies among features.
* [ ] **Matrix Rank:** Determine the rank of a matrix by counting non-zero pivot rows in its REF.
  * *ML Connection:* Detecting feature redundancy and multicollinearity; low-rank model architectures.
* [ ] **Determinants:** Compute $2 \times 2$ determinants ($ad - bc$) and $3 \times 3$ determinants using cofactor expansion or row reduction.
  * *ML Connection:* Invertibility check; volume scaling factor in Normalizing Flow generative models.
* [ ] **Inverse by $2 \times 2$ Formula:** Compute $A^{-1} = \frac{1}{ad-bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$ and verify $A A^{-1} = I$.
  * *ML Connection:* Analytic solution for 2D Gaussian distributions and bivariate regression.
* [ ] **Inverse by Gauss-Jordan:** Augment $[A \mid I]$ and apply row operations to row-reduce $A$ to $I$, producing $[I \mid A^{-1}]$.
  * *ML Connection:* Exact numerical inversion algorithm implemented in linear algebra libraries.
* [ ] **Inverse by Adjugate / Cofactors:** Calculate all minors $M_{ij}$, cofactors $C_{ij} = (-1)^{i+j}M_{ij}$, transpose cofactor matrix $C$ to form $\operatorname{adj}(A) = C^T$, and compute $A^{-1} = \frac{1}{\det(A)}\operatorname{adj}(A)$ on a $3 \times 3$ matrix.
  * *ML Connection:* Closed-form analytic inverse in camera calibration, robotics, and small-dimensional spatial transforms.
* [ ] **Linear Independence, Span, and Basis:** Test whether a set of vectors is linearly independent ($c_1 \mathbf{v}_1 + \cdots + c_k \mathbf{v}_k = \mathbf{0} \implies c_i = 0$); find a minimal basis spanning a subspace.
  * *ML Connection:* Identifying non-redundant feature subsets; latent coordinate representation.
* [ ] **Null Space:** Solve $A\mathbf{x} = \mathbf{0}$ by expressing pivot variables in terms of free variables to find a basis for $N(A)$.
  * *ML Connection:* Uninformative feature combinations that produce zero model response; adversarial perturbations.
* [ ] **Four Fundamental Subspaces:** Compute the dimensions and basis vectors for $C(A)$, $N(A)$, $C(A^T)$, and $N(A^T)$ for a rectangular matrix.
  * *ML Connection:* Understanding how a data matrix partitions sample space and feature space.
* [ ] **Projections:** Compute the projection of a vector $\mathbf{b}$ onto vector $\mathbf{a}$ ($\mathbf{p} = \frac{\mathbf{a}^T\mathbf{b}}{\mathbf{a}^T\mathbf{a}}\mathbf{a}$) and subspace projection matrix $P = X(X^T X)^{-1}X^T$.
  * *ML Connection:* Orthogonal projection onto feature columns in regression; subspace clustering.
* [ ] **Least Squares:** Solve the Normal Equation $(X^T X)\mathbf{w}^* = X^T \mathbf{y}$ for a small dataset by hand.
  * *ML Connection:* Ordinary Least Squares (OLS) linear regression; optimal linear predictor.
* [ ] **Eigenvalues & Eigenvectors:** Solve $\det(A - \lambda I) = 0$ for eigenvalues and $(A - \lambda I)\mathbf{v} = \mathbf{0}$ for eigenvectors of a $2 \times 2$ matrix.
  * *ML Connection:* Finding principal directions of maximum variance in PCA; spectral graph analysis.
* [ ] **Diagonalization:** Decompose a diagonalizable matrix into $A = P D P^{-1}$ and compute matrix powers $A^k = P D^k P^{-1}$.
  * *ML Connection:* Fast computation of transition matrix powers in Markov chains and PageRank.
* [ ] **Symmetric Matrices:** Verify that eigenvectors of a symmetric matrix corresponding to distinct eigenvalues are mutually orthogonal; write its spectral decomposition $A = Q \Lambda Q^T$.
  * *ML Connection:* Guarantees that covariance matrices and graph Laplacians yield orthogonal feature representations.
* [ ] **Positive Definite Matrices:** Test whether a symmetric matrix is positive definite via eigenvalues, leading principal minors, or Cholesky factorability.
  * *ML Connection:* Validating covariance matrices; verifying strict convexity of loss functions for guaranteed unique global minima.
* [ ] **Singular Value Decomposition (SVD):** For a small rectangular or rank-deficient matrix, compute $A^T A$, its eigenvalues $\sigma_i^2$, singular values $\sigma_i$, right singular vectors $\mathbf{v}_i$, and left singular vectors $\mathbf{u}_i = \frac{1}{\sigma_i}A\mathbf{v}_i$. Verify $A = \sum \sigma_i \mathbf{u}_i \mathbf{v}_i^T$.
  * *ML Connection:* Low-Rank Adaptation (LoRA), Latent Semantic Analysis (LSA), collaborative filtering / matrix completion.
* [ ] **Pseudoinverse:** Compute $A^+$ using SVD ($V \Sigma^+ U^T$) or the left inverse shortcut $(A^T A)^{-1}A^T$.
  * *ML Connection:* Closed-form least squares solution for overdetermined systems and minimum-norm solution for underdetermined deep models.

---

### 2. Conceptual & Geometric Explanations
* [ ] **Row Picture vs. Column Picture:** Explain $A\mathbf{x} = \mathbf{b}$ both as the intersection of hyperplanes and as a linear combination of feature columns landing on target $\mathbf{b}$.
* [ ] **Why $A \operatorname{adj}(A) = \det(A) I$:** Explain why diagonal entries yield Laplace cofactor expansions while off-diagonal entries expand matrices with repeated rows (yielding zero).
* [ ] **Two-Sided vs. Left vs. Right vs. Pseudoinverse:** Explain why tall matrices ($m > n$, full column rank) have left inverses, wide matrices ($m < n$, full row rank) have right inverses, and only square full-rank matrices have two-sided inverses.
* [ ] **Geometric Derivation of Least Squares:** Explain why the residual error $\mathbf{e} = \mathbf{y} - X\mathbf{w}^*$ must be perpendicular to every column of $X$, immediately yielding $X^T \mathbf{e} = \mathbf{0} \implies (X^T X)\mathbf{w}^* = X^T \mathbf{y}$.
* [ ] **Full vs. Compact vs. Truncated SVD:** Explain the geometric action of SVD (Rotate $\to$ Scale $\to$ Rotate) and how truncated SVD provides the optimal rank-$k$ approximation under the Eckart-Young Theorem.
* [ ] **Why Covariance Matrices are Positive Semi-Definite:** Explain why sample variance along any direction $\mathbf{u}^T \Sigma \mathbf{u} = \frac{1}{n-1}\|X_c \mathbf{u}\|_2^2$ can never be negative.
* [ ] **The Four Fundamental Subspaces Geometry:** Explain Strang's Big Picture: how $C(A^T) \perp N(A)$ in $\mathbb{R}^n$ and $C(A) \perp N(A^T)$ in $\mathbb{R}^m$.
* [ ] **Explain the ML Connection of Each:** Confidently explain how every single item on this checklist directly enables or explains a practical machine learning algorithm.

---

> 📖 **Next Step:** Begin your journey with **[Part 01: Vectors and Vector Spaces](./01_vectors_and_vector_spaces.md)** or test your current readiness with the **[Part 26: Paper & Pencil Self-Test Checklist](./26_paper_and_pencil_checklist.md)**.
