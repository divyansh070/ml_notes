> 📖 **Navigation:** [← Previous: Part 14: Least Squares & Linear Regression](./14_least_squares_and_linear_regression.md) | [🏠 Index](./README.md) | [Next: Part 16: Diagonalization →](./16_diagonalization.md)

---

# PART 15 — EIGENVALUES & EIGENVECTORS

When a square matrix $A$ multiplies most vectors $\mathbf{x}$, it simultaneously **rotates** and **stretches** them. 

However, for any square matrix, there exist special invariant directions $\mathbf{v}$ where the matrix **ONLY STRETCHES or COMPRESSES the vector without rotating it at all**:

$$
A\mathbf{v} = \lambda\mathbf{v}
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
* $\lambda \in \mathbb{R}$ (or $\mathbb{C}$) is the **Eigenvalue** (the scalar stretch factor along axis $\mathbf{v}$).
  * $\lambda > 1$: Stretches vector.
  * $0 < \lambda < 1$: Compresses vector.
  * $\lambda < 0$: Reverses vector direction along the invariant line.
  * $\lambda = 0$: Collapses vector to $\mathbf{0}$ (meaning $\mathbf{v} \in N(A)$).

---

## 15.1 The Characteristic Equation: Deriving $\det(A - \lambda I) = 0$

Starting from the fundamental definition:

$$
A\mathbf{v} = \lambda\mathbf{v}
$$

Subtract $\lambda\mathbf{v} = \lambda I\mathbf{v}$ to move all terms to the left:

$$
(A - \lambda I)\mathbf{v} = \mathbf{0}
$$

This is a homogeneous linear system.
* If $(A - \lambda I)$ were invertible, multiplying by its inverse would force the trivial solution $\mathbf{v} = (A - \lambda I)^{-1}\mathbf{0} = \mathbf{0}$.
* But by definition, an **eigenvector must be non-zero** ($\mathbf{v} \neq \mathbf{0}$).
* Therefore, the matrix $(A - \lambda I)$ **MUST BE SINGULAR (non-invertible)**!

A matrix is singular if and only if its determinant is zero. This yields the **Characteristic Equation**:

$$
\det(A - \lambda I) = 0
$$

The polynomial $p(\lambda) = \det(A - \lambda I)$ is the **Characteristic Polynomial**. Its roots are the eigenvalues of $A$.

---

## 15.2 Eigenspaces

For each eigenvalue $\lambda_i$, the set of all vectors satisfying $(A - \lambda_i I)\mathbf{v} = \mathbf{0}$ forms a vector subspace called the **Eigenspace** of $\lambda_i$:

$$
E_{\lambda_i} = N(A - \lambda_i I) = \left\lbrace \mathbf{v} \in \mathbb{R}^n \;\middle|\; (A - \lambda_i I)\mathbf{v} = \mathbf{0} \right\rbrace
$$

Every non-zero vector in $E_{\lambda_i}$ is a valid eigenvector. Scaling an eigenvector by any scalar $c \neq 0$ produces another valid eigenvector.

---

## 15.3 The Two Master Identities: Trace and Determinant

For any $n \times n$ matrix $A$ with eigenvalues $\lambda_1, \dots, \lambda_n$:

1. **The Sum of Eigenvalues is the Trace:**

$$
\text{Tr}(A) = \sum_{i=1}^{n} \lambda_i = \lambda_1 + \lambda_2 + \dots + \lambda_n
$$

2. **The Product of Eigenvalues is the Determinant:**

$$
\det(A) = \prod_{i=1}^{n} \lambda_i = \lambda_1 \times \lambda_2 \times \dots \times \lambda_n
$$

*(These identities allow instantaneous verification of hand-calculated eigenvalues!).*

---

## 15.4 Complete Worked Numerical Example ($2 \times 2$)

Find all eigenvalues and eigenvectors for:

$$
A = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}
$$

### Step 1: Set Up the Characteristic Equation

$$
A - \lambda I = \begin{bmatrix} 4 - \lambda & 2 \\ 1 & 3 - \lambda \end{bmatrix}
$$

$$
\det(A - \lambda I) = (4 - \lambda)(3 - \lambda) - (2)(1) = 0
$$

### Step 2: Expand and Solve the Quadratic Polynomial

$$
\lambda^2 - 7\lambda + 12 - 2 = \lambda^2 - 7\lambda + 10 = 0
$$

Factor the quadratic:

$$
(\lambda - 5)(\lambda - 2) = 0 \implies \lambda_1 = 5, \quad \lambda_2 = 2
$$

#### Sanity Check via Trace and Determinant:
* $\text{Tr}(A) = 4 + 3 = 7$; Sum of eigenvalues: $5 + 2 = 7$ ✓
* $\det(A) = (4)(3) - (2)(1) = 10$; Product of eigenvalues: $5 \times 2 = 10$ ✓

---

### Step 3: Find Eigenvector 1 for $\lambda_1 = 5$
Substitute $\lambda_1 = 5$ into $(A - 5I)\mathbf{v}_1 = \mathbf{0}$:

$$
\begin{bmatrix} 4 - 5 & 2 \\ 1 & 3 - 5 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} =
\begin{bmatrix} -1 & 2 \\ 1 & -2 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

Both rows yield the identical constraint:

$$
-v_1 + 2v_2 = 0 \implies v_1 = 2v_2
$$

Choose $v_2 = 1 \implies v_1 = 2$:

$$
\mathbf{v}_1 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}
$$

Normalize to unit length:

$$
\mathbf{u}_1 = \frac{1}{\sqrt{2^2 + 1^2}} \begin{bmatrix} 2 \\ 1 \end{bmatrix} = \frac{1}{\sqrt{5}} \begin{bmatrix} 2 \\ 1 \end{bmatrix}
$$

---

### Step 4: Find Eigenvector 2 for $\lambda_2 = 2$
Substitute $\lambda_2 = 2$ into $(A - 2I)\mathbf{v}_2 = \mathbf{0}$:

$$
\begin{bmatrix} 4 - 2 & 2 \\ 1 & 3 - 2 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} =
\begin{bmatrix} 2 & 2 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

Row constraint:

$$
v_1 + v_2 = 0 \implies v_1 = -v_2
$$

Choose $v_2 = 1 \implies v_1 = -1$:

$$
\mathbf{v}_2 = \begin{bmatrix} -1 \\ 1 \end{bmatrix}
$$

Normalize to unit length:

$$
\mathbf{u}_2 = \frac{1}{\sqrt{(-1)^2 + 1^2}} \begin{bmatrix} -1 \\ 1 \end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix} -1 \\ 1 \end{bmatrix}
$$

---

### Step 5: Verification ($A\mathbf{v} = \lambda\mathbf{v}$)
* For $\lambda_1 = 5$:

$$
A\mathbf{v}_1 = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} 2 \\ 1 \end{bmatrix} =
\begin{bmatrix} 8 + 2 \\ 2 + 3 \end{bmatrix} = \begin{bmatrix} 10 \\ 5 \end{bmatrix} = 5 \begin{bmatrix} 2 \\ 1 \end{bmatrix} = \lambda_1 \mathbf{v}_1
$$

✓ **Verified**.

* For $\lambda_2 = 2$:

$$
A\mathbf{v}_2 = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix} \begin{bmatrix} -1 \\ 1 \end{bmatrix} =
\begin{bmatrix} -4 + 2 \\ -1 + 3 \end{bmatrix} = \begin{bmatrix} -2 \\ 2 \end{bmatrix} = 2 \begin{bmatrix} -1 \\ 1 \end{bmatrix} = \lambda_2 \mathbf{v}_2
$$

✓ **Verified**.

---

## 15.5 Why this matters in ML

1. **Principal Component Analysis (PCA):** The directions of maximum variance in a dataset are the eigenvectors of the sample covariance matrix $\Sigma = \frac{1}{n-1} X_c^T X_c$, and the variance along each direction is its eigenvalue $\lambda_i$.
2. **Spectral Graph Theory & Clustering:** The eigenvectors of the Graph Laplacian matrix $L = D - A$ reveal optimal cluster cuts and partitions in complex network datasets.
3. **Google PageRank & Markov Chains:** The long-term stationary distribution of a random walk across web pages is the dominant eigenvector ($\lambda = 1$) of the hyperlink transition matrix.

---

## 15.6 Common mistakes

* **Including the Zero Vector as an Eigenvector:** By definition, $\mathbf{v} \neq \mathbf{0}$. While $\lambda = 0$ is a completely valid eigenvalue (meaning $A$ is singular), $\mathbf{v}$ can never be the zero vector.
* **Confusing Eigenvectors with Ordinary Vectors:** Multiplying a matrix by a general vector changes both its length and direction. Only along eigenvectors is the transformation a pure scalar scaling.
* **Assuming Eigenvectors Are Unique:** Any non-zero multiple $c\mathbf{v}$ is equally valid as an eigenvector. We normalize them to unit length $\|\mathbf{v}\|_2 = 1$ to establish a standardized basis.

---

> 📖 **Navigation:** [← Previous: Part 14: Least Squares & Linear Regression](./14_least_squares_and_linear_regression.md) | [🏠 Index](./README.md) | [Next: Part 16: Diagonalization →](./16_diagonalization.md)
