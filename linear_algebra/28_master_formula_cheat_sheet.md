> 📖 **Navigation:** [← Previous: Part 27: 40 Essential Technical Interview Questions & Answers](./27_interview_questions_and_answers.md) | [🏠 Index](./README.md) | [Return to Index →](./README.md)

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

> 📖 **Navigation:** [← Previous: Part 27: 40 Essential Technical Interview Questions & Answers](./27_interview_questions_and_answers.md) | [🏠 Index](./README.md) | [Return to Index →](./README.md)
