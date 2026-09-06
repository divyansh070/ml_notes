> 📖 **Navigation:** [← Previous: Part 06: Linear Independence, Basis & Rank](./06_linear_independence_and_rank.md) | [🏠 Index](./README.md) | [Next: Part 08: Eigenvalues in Principal Component Analysis (PCA) →](./08_eigenvalues_in_pca.md)

---

# PART 7 — EIGENVALUES, EIGENVECTORS & DIAGONALIZATION

When a matrix $A$ acts on most vectors $\mathbf{x}$, it both **rotates** and **stretches** them. 

However, for any square matrix, there exist special invariant directions $\mathbf{v}$ where the matrix **ONLY STRETCHES or COMPRESSES the vector without rotating it at all**:

$$
A \mathbf{v} = \lambda \mathbf{v}
$$

```
        TYPICAL VECTOR (Rotated & Stretched)            EIGENVECTOR (Only Stretched by Lambda!)
                 y                                                y
                 │        A @ x                                   │            A @ v = lambda * v
                 │       ╱                                        │           ╱
                 │      ╱                                         │          ╱
                 │     ● ◄── Rotated                              │         ●
                 │    ╱                                           │        ╱
                 │   ● x                                          │       ● v (Same invariant line!)
                 └───┴────────► x                                 └───────┴────────► x
```

* $\mathbf{v} \neq \mathbf{0}$ is the **Eigenvector** (the invariant axis).
* $\lambda \in \mathbb{R}$ is the **Eigenvalue** (the scalar stretch factor along that axis).

---

## 7.1 Complete 13-Step Hand Derivation on a 2x2 Matrix

We will solve for all eigenvalues and eigenvectors of:
$$
A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}
$$

### Step 1: Fundamental Equation
$$
A \mathbf{v} = \lambda \mathbf{v}
$$

### Step 2: Move Terms to One Side
$$
(A - \lambda I) \mathbf{v} = \mathbf{0}
$$

### Step 3: Why $\det(A - \lambda I) = 0$ is Required
If $(A - \lambda I)$ were invertible, multiplying by its inverse would yield only the trivial solution $\mathbf{v} = \mathbf{0}$. To find non-zero eigenvectors $\mathbf{v} \neq \mathbf{0}$, the matrix $(A - \lambda I)$ **must be singular (non-invertible)**:
$$
\det(A - \lambda I) = 0
$$

### Step 4: Construct the Matrix $(A - \lambda I)$
$$
A - \lambda I = \begin{bmatrix} 2 - \lambda & 1 \\ 1 & 2 - \lambda \end{bmatrix}
$$

### Step 5: Compute the Characteristic Polynomial
$$
\det(A - \lambda I) = (2 - \lambda)^2 - (1 \times 1) = \lambda^2 - 4\lambda + 3 = 0
$$

### Step 6: Solve for Eigenvalues
$$
(\lambda - 3)(\lambda - 1) = 0 \implies \lambda_1 = 3, \quad \lambda_2 = 1
$$

### Step 7: Find Eigenvector 1 for $\lambda_1 = 3$
Substitute $\lambda_1 = 3$ into $(A - 3I)\mathbf{v} = \mathbf{0}$:
$$
\begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \implies -v_1 + v_2 = 0 \implies v_1 = v_2
$$
Choose unit length:

$$
\mathbf{v}_1 = \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix}
$$


### Step 8: Find Eigenvector 2 for $\lambda_2 = 1$
Substitute $\lambda_2 = 1$ into $(A - 1I)\mathbf{v} = \mathbf{0}$:
$$
\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \implies v_1 + v_2 = 0 \implies v_2 = -v_1
$$
Choose unit length:

$$
\mathbf{v}_2 = \begin{bmatrix} 1/\sqrt{2} \\ -1/\sqrt{2} \end{bmatrix}
$$


### Step 9: Verification
$$
A \mathbf{v}_1 = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 3 \\ 3 \end{bmatrix} = 3 \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \lambda_1 \mathbf{v}_1 \quad \checkmark
$$
$$
A \mathbf{v}_2 = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 1 \\ -1 \end{bmatrix} = 1 \begin{bmatrix} 1 \\ -1 \end{bmatrix} = \lambda_2 \mathbf{v}_2 \quad \checkmark
$$

---

## 7.2 The Spectral Theorem & Orthogonal Diagonalization

For any real symmetric matrix $A = A^T$ (such as sample covariance matrices $\Sigma = \frac{1}{n-1} X_c^T X_c$):

> [!IMPORTANT]
> **The Spectral Theorem:**
> 1. All eigenvalues $\lambda_1, \dots, \lambda_n$ are strictly **real numbers**.
> 2. Eigenvectors corresponding to distinct eigenvalues are **mutually orthogonal** ($\mathbf{v}_1 \cdot \mathbf{v}_2 = 0$).
> 3. Matrix $A$ can be **orthogonally diagonalized**:
>    $$
>    A = Q \Lambda Q^T = \sum_{i=1}^{n} \lambda_i \mathbf{q}_i \mathbf{q}_i^T
>    $$
>    where $Q$ is an orthogonal matrix of eigenvectors ($Q^T Q = I$) and $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_n)$.

* **Geometric Meaning of Diagonalization:** $Q^T$ rotates space to align the eigenvectors with coordinate axes; $\Lambda$ stretches along those axes; $Q$ rotates space back.

---

## 7.3 Power Iteration: Finding the Dominant Eigenvector

**Power Iteration** is the foundational algorithm used to find the principal eigenvector (used in Google's PageRank):
1. Start with a random unit vector $\mathbf{x}_0$.
2. Repeatedly multiply and normalize:
   $$
   \mathbf{x}_{k+1} = \frac{A \mathbf{x}_k}{\|A \mathbf{x}_k\|_2}
   $$
3. As $k \to \infty$, $\mathbf{x}_k$ converges to the dominant eigenvector $\mathbf{q}_1$ at rate $\left|\frac{\lambda_2}{\lambda_1}\right|^k$.

---

## Advanced / Optional — Do not study until Core track is complete

### A.1 Spectral Radius ($\rho(A)$) & System Stability in ML
The Spectral Radius $\rho(A) = \max_i |\lambda_i|$ governs the behavior of matrix powers $A^k$:
* If $\rho(A) < 1.0 \implies \lim_{k \to \infty} A^k = 0$ (Stable dynamics / vanishing gradients in RNNs).
* If $\rho(A) > 1.0 \implies \lim_{k \to \infty} \|A^k\| = \infty$ (Exploding dynamics / exploding gradients in RNNs).

### A.2 The Gershgorin Circle Theorem
Every eigenvalue of $A \in \mathbb{C}^{n \times n}$ lies within at least one Gershgorin disk in the complex plane centered at diagonal entry $A_{ii}$ with radius $R_i = \sum_{j \neq i} |A_{ij}|$.

---

> 📖 **Navigation:** [← Previous: Part 06: Linear Independence, Basis & Rank](./06_linear_independence_and_rank.md) | [🏠 Index](./README.md) | [Next: Part 08: Eigenvalues in Principal Component Analysis (PCA) →](./08_eigenvalues_in_pca.md)
