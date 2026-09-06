> 📖 **Navigation:** [← Previous: Part 27: 40 Essential Technical Interview Questions & Answers](./27_interview_questions_and_answers.md) | [🏠 Index](./README.md) | [Return to Index →](./README.md)

---

# PART 28 — MASTER FORMULA CHEAT SHEET & QUICK REFERENCE

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             MASTER LINEAR ALGEBRA & ML CHEAT SHEET                               │
├──────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                  │
│ 1. VECTOR OPERATIONS & INNER PRODUCT SPACES:                                                     │
│    • Dot Product & Angle:     a · b = a^T b = sum(a_i * b_i) = ||a||_2 ||b||_2 cos(θ)             │
│    • Cauchy-Schwarz Bound:    |u · v| ≤ ||u||_2 ||v||_2  (Guarantees -1 ≤ Cosine Sim ≤ 1)       │
│    • General Lp Norm:         ||x||_p = (sum |x_i|^p)^(1/p)                                      │
│    • L-infinity (Max Norm):   ||x||_∞ = max |x_i|                                                │
│    • L0 "Pseudo-Norm":        ||x||_0 = count(x_i ≠ 0)  (Fails absolute scalability!)            │
│    • Minkowski Inequality:    ||u + v||_p ≤ ||u||_p + ||v||_p                                    │
│    • Hölder's Inequality:     |x^T y| ≤ ||x||_p ||y||_q   where 1/p + 1/q = 1                    │
│                                                                                                  │
│ 2. MATRIX OPERATIONS & ESSENTIAL PROPERTIES:                                                     │
│    • Transpose Product:       (A B)^T = B^T A^T               (A B)^-1 = B^-1 A^-1               │
│    • Matrix Trace:            Tr(A) = sum(A_ii) = sum(λ_i)                                       │
│    • Cyclic Trace Property:   Tr(A B C) = Tr(B C A) = Tr(C A B)                                  │
│    • Frobenius Norm:          ||A||_F = sqrt(sum A_ij^2) = sqrt(Tr(A^T A)) = sqrt(sum σ_i^2)     │
│    • Spectral Norm:           ||A||_2 = σ_max(A) = sqrt(λ_max(A^T A))                            │
│    • Nuclear / Trace Norm:    ||A||_* = sum(σ_i)  (Convex relaxation of rank)                    │
│    • Hadamard vs. Kronecker:  A ⊙ B (Element-wise) vs. A ⊗ B (Block outer product)               │
│                                                                                                  │
│ 3. INVERSES & LOW-RANK UPDATES:                                                                  │
│    • 2x2 Inverse Formula:     [a b; c d]^-1 = (1 / (ad - bc)) * [d -b; -c a]                     │
│    • Sherman-Morrison (Rank-1):(A + u v^T)^-1 = A^-1 - (A^-1 u v^T A^-1) / (1 + v^T A^-1 u)       │
│    • Woodbury Identity:       (A + U C V)^-1 = A^-1 - A^-1 U (C^-1 + V A^-1 U)^-1 V A^-1         │
│    • Schur Complement:        S_D = A - B D^-1 C  ==>  Cov(x_1 | x_2) = Σ_11 - Σ_12 Σ_22^-1 Σ_21 │
│    • Left Inverse (Tall M>N): A_left^-1 = (A^T A)^-1 A^T          ==> A_left^-1 A = I_N          │
│    • Right Inverse (Wide M<N):A_right^-1 = A^T (A A^T)^-1         ==> A A_right^-1 = I_M         │
│                                                                                                  │
│ 4. DETERMINANTS (STRANG'S CORE RULES):                                                           │
│    • Product Rule:            det(A B) = det(A) det(B)        det(A^-1) = 1 / det(A)             │
│    • Scalar Scaling:          det(c A) = c^n det(A)  (For n x n matrix!)                         │
│    • Transpose Invariance:    det(A^T) = det(A)                                                  │
│    • Eigenvalues Product:     det(A) = prod(λ_i) = λ_1 * λ_2 * ... * λ_n                         │
│    • Cramer's Rule:           x_i = det(A_i) / det(A)                                            │
│    • Jacobian Change of Vars: p_Y(y) = p_X(x) * |det(∂f/∂x)|^-1  (Normalizing Flows)             │
│                                                                                                  │
│ 5. SPECTRAL THEORY & EIGENVALUES:                                                                │
│    • Characteristic Equation: det(A - λ I) = 0                                                   │
│    • Spectral Theorem:        A = A^T ==> A = Q Λ Q^T = sum(λ_i q_i q_i^T) (All λ real, q orthog)│
│    • Spectral Radius & Stab:  ρ(A) = max |λ_i|  (lim A^k = 0 <==> ρ(A) < 1.0)                    │
│    • Power Iteration:         x_(k+1) = (A x_k) / ||A x_k||_2  ==> Converges to dominant q_1     │
│    • Matrix Operations:       Eigenvalues of A^k are λ_i^k; Eigenvalues of A^-1 are 1/λ_i        │
│                                                                                                  │
│ 6. SINGULAR VALUE DECOMPOSITION (SVD):                                                           │
│    • Full SVD Equation:       A = U Σ V^T    (U: Left Singular, Σ: Stretch, V: Right Singular)   │
│    • Covariance Connection:   X_c = U Σ V^T  ==>  X_c^T X_c = V Σ^2 V^T  ==> λ_i = σ_i^2 / (n-1) │
│    • Eckart-Young Theorem:    A_k = sum_{i=1}^k σ_i u_i v_i^T is optimal rank-k approximation    │
│    • Approximation Errors:    ||A - A_k||_F = sqrt(sum_{i=k+1}^r σ_i^2),  ||A - A_k||_2 = σ_{k+1}│
│    • Condition Number:        κ(A) = σ_max / σ_min ≥ 1.0  (Controls gradient descent convergence)│
│                                                                                                  │
│ 7. POSITIVE DEFINITE MATRICES & OPTIMIZATION:                                                    │
│    • 5 Equivalent Tests:      1. x^T A x > 0 (x≠0)   2. All λ_i > 0   3. All leading minors > 0  │
│                               4. All pivots > 0      5. Cholesky Factorization A = L L^T         │
│    • Rayleigh-Ritz Theorem:   λ_min ≤ (x^T A x) / (x^T x) ≤ λ_max  (Derives PCA via Lagrange)    │
│    • Mercer's Theorem (SVM):  Kernel Gram Matrix K_ij = k(x_i, x_j) is always PSD (K ⪰ 0)        │
│    • Gaussian Sampling (VAE): x = μ + L z   where z ~ N(0, I) and Σ = L L^T (Reparameterization) │
│                                                                                                  │
│ 8. PROJECTIONS & LINEAR REGRESSION (OLS):                                                        │
│    • Projection Matrix:       P = X (X^T X)^-1 X^T  (Symmetric P^T = P, Idempotent P^2 = P)      │
│    • Residual Projector:      (I - P) projects onto orthogonal complement Col(X)^⊥               │
│    • Normal Equation:         w = (X^T X)^-1 X^T y                                               │
│    • Residual Orthogonality:  X^T e = X^T (y - X w) = 0 ==> sum(e_i) = 0 and ŷ^T e = 0           │
│    • Gauss-Markov Theorem:    OLS is BLUE: E[ŵ] = w,  Cov(ŵ) = σ^2 (X^T X)^-1                   │
│                                                                                                  │
│ 9. MATRIX CALCULUS IDENTITIES:                                                                   │
│    • ∇_x (a^T x) = a                          ∇_x (x^T A x) = (A + A^T) x = 2 A x (if A=A^T)     │
│    • ∇_x ||A x - b||_2^2 = 2 A^T (A x - b)    ∇_X Tr(A X) = A^T                                  │
│    • ∇_X log det(X) = X^-T                    ∇_X Tr(X^T A X) = (A + A^T) X                      │
│    • Multivariable Taylor:    f(x) ≈ f(x_0) + ∇f(x_0)^T (x - x_0) + (1/2)(x - x_0)^T H (x - x_0) │
│    • Newton's Method Step:    x_(k+1) = x_k - [H(x_k)]^-1 ∇f(x_k)                                │
│                                                                                                  │
│ 10. REGULARIZATION & SPARSITY:                                                                   │
│     • Ridge (L2) SVD Filter:  w_ridge = sum (σ_i / (σ_i^2 + λ)) (u_i^T y) v_i                    │
│     • Lasso (L1) Soft-Thresh: w_lasso = sign(w_ols) * max(|w_ols| - λ, 0)                        │
│     • ElasticNet:             Loss = MSE + λ_1 ||w||_1 + (λ_2 / 2) ||w||_2^2 (Groups correlated) │
│                                                                                                  │
│ 11. COVARIANCE & MAHALANOBIS DISTANCE:                                                           │
│     • Sample Covariance:      Σ = (1 / (n - 1)) X_centered^T X_centered                          │
│     • Mahalanobis Distance:   D_M(x, μ) = sqrt((x - μ)^T Σ^-1 (x - μ))                           │
│     • Total Variance:         Tr(Σ) = sum(s_i^2) = sum(λ_i)                                      │
│     • Generalized Variance:   det(Σ) = prod(λ_i)                                                 │
│                                                                                                  │
│ 12. MOORE-PENROSE PSEUDOINVERSE (A^+):                                                           │
│     • Full Column Rank (M>N): A^+ = (A^T A)^-1 A^T  (Left Inverse)                               │
│     • Full Row Rank (M<N):    A^+ = A^T (A A^T)^-1  (Right Inverse)                              │
│     • Universal SVD:          A^+ = V Σ^+ U^T       (Inverts non-zero σ_i to 1/σ_i)              │
│                                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

> 📖 **Navigation:** [← Previous: Part 27: 40 Essential Technical Interview Questions & Answers](./27_interview_questions_and_answers.md) | [🏠 Index](./README.md) | [Return to Index →](./README.md)
