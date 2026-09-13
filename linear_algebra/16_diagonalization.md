> 📖 **Navigation:** [← Previous: Part 15: Eigenvalues & Eigenvectors](./15_eigenvalues_and_eigenvectors.md) | [🏠 Index](./README.md) | [Next: Part 17: Symmetric Matrices & Spectral Theorem →](./17_symmetric_matrices_and_spectral_theorem.md)

---

# PART 16 — DIAGONALIZATION ($A = P D P^{-1}$)

Diagonal matrices are the simplest possible matrices to work with: multiplying by a diagonal matrix simply scales each coordinate axis independently. **Diagonalization** rotates and rescales our coordinate system so that a general matrix acts as a pure diagonal scaling matrix.

---

## 16.1 The Diagonalization Equation: $A = P D P^{-1}$

Let $A \in \mathbb{R}^{n \times n}$ possess $n$ linearly independent eigenvectors $\mathbf{v}_1, \dots, \mathbf{v}_n$ with corresponding eigenvalues $\lambda_1, \dots, \lambda_n$.

Package the eigenvectors into the columns of matrix $P$:

$$
P = \begin{bmatrix} \mid & \mid & & \mid \\ \mathbf{v}_1 & \mathbf{v}_2 & \dots & \mathbf{v}_n \\ \mid & \mid & & \mid \end{bmatrix}
$$

Package the eigenvalues into the diagonal matrix $D$:

$$
D = \begin{bmatrix}
\lambda_1 & 0 & \dots & 0 \\
0 & \lambda_2 & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & \lambda_n
\end{bmatrix}
$$

### The Algebraic Derivation:
Multiply $A$ by eigenvector matrix $P$:

$$
AP = A \begin{bmatrix} \mathbf{v}_1 & \dots & \mathbf{v}_n \end{bmatrix} = \begin{bmatrix} A\mathbf{v}_1 & \dots & A\mathbf{v}_n \end{bmatrix}
$$

Because $A\mathbf{v}_i = \lambda_i \mathbf{v}_i$:

$$
AP = \begin{bmatrix} \lambda_1 \mathbf{v}_1 & \dots & \lambda_n \mathbf{v}_n \end{bmatrix} = \begin{bmatrix} \mathbf{v}_1 & \dots & \mathbf{v}_n \end{bmatrix} \begin{bmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{bmatrix} = P D
$$

Since the $n$ eigenvectors are linearly independent, $P$ is invertible. Multiplying by $P^{-1}$ on the right yields:

$$
A = P D P^{-1}
$$

Equivalently, pre-multiplying by $P^{-1}$:

$$
D = P^{-1} A P
$$

---

## 16.2 Matrix Powers via Diagonalization ($A^k = P D^k P^{-1}$)

Computing a high power of a matrix $A^k = A \times A \times \dots \times A$ directly is computationally expensive ($\mathcal{O}(k n^3)$).

Using diagonalization, all interior terms collapse:

$$
A^k = (P D P^{-1}) (P D P^{-1}) \dots (P D P^{-1}) = P D (P^{-1} P) D \dots D P^{-1} = P D^k P^{-1}
$$

Because $D$ is diagonal, raising it to the power $k$ simply raises each diagonal eigenvalue to the power $k$:

$$
D^k = \begin{bmatrix}
\lambda_1^k & 0 & \dots & 0 \\
0 & \lambda_2^k & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & \lambda_n^k
\end{bmatrix}
$$

> [!IMPORTANT]
> **Long-Term System Stability in Machine Learning:**
> As $k \to \infty$, the behavior of $A^k$ is governed entirely by the magnitudes of its eigenvalues $|\lambda_i|$:
> * If all $|\lambda_i| < 1 \implies \lim_{k \to \infty} A^k = 0$ (System decays to zero / **Vanishing Gradients** in recurrent neural networks).
> * If any $|\lambda_i| > 1 \implies \lim_{k \to \infty} \|A^k\| = \infty$ (System blows up / **Exploding Gradients**).
> * If $\max |\lambda_i| = 1 \implies$ System converges to a stable steady state (Markov Chain steady-state distribution).

---

## 16.3 When is a Matrix Diagonalizable?

An $n \times n$ matrix $A$ is diagonalizable **if and only if it has $n$ linearly independent eigenvectors**.

1. **Distinct Eigenvalues (Sufficient Condition):** If an $n \times n$ matrix has $n$ strictly distinct eigenvalues ($\lambda_i \neq \lambda_j$), it is **guaranteed to be diagonalizable**.
2. **Repeated Eigenvalues (Defective Matrices):**
   * **Algebraic Multiplicity ($AM$):** The number of times $\lambda_i$ appears as a root of $\det(A - \lambda I) = 0$.
   * **Geometric Multiplicity ($GM$):** The number of independent eigenvectors for $\lambda_i$ ($\dim(N(A - \lambda_i I))$).
   * A matrix is diagonalizable if and only if **$GM = AM$ for every eigenvalue**.
   * If $GM < AM$, the matrix is **defective** (lacks sufficient independent eigenvectors) and cannot be diagonalized. Example: shear matrix $\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ has $\lambda = 1$ with $AM = 2$ but $GM = 1$.

---

## 16.4 Complete Worked Numerical Example

Diagonalize $A = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix}$ and compute $A^4$.

### Step 1: Eigenvalues and Eigenvectors (from Part 15)
* $\lambda_1 = 5 \implies \mathbf{v}_1 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$
* $\lambda_2 = 2 \implies \mathbf{v}_2 = \begin{bmatrix} -1 \\ 1 \end{bmatrix}$

### Step 2: Assemble $P$ and $D$

$$
P = \begin{bmatrix} 2 & -1 \\ 1 & 1 \end{bmatrix}, \quad D = \begin{bmatrix} 5 & 0 \\ 0 & 2 \end{bmatrix}
$$

### Step 3: Invert $P$ using $2 \times 2$ formula
$\det(P) = (2)(1) - (-1)(1) = 2 + 1 = 3$.

$$
P^{-1} = \frac{1}{3} \begin{bmatrix} 1 & 1 \\ -1 & 2 \end{bmatrix}
$$

### Step 4: Verify $A = P D P^{-1}$

$$
P D = \begin{bmatrix} 2 & -1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 5 & 0 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 10 & -2 \\ 5 & 2 \end{bmatrix}
$$

$$
(P D) P^{-1} = \frac{1}{3} \begin{bmatrix} 10 & -2 \\ 5 & 2 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ -1 & 2 \end{bmatrix} =
\frac{1}{3} \begin{bmatrix} 10(1) - 2(-1) & 10(1) - 2(2) \\ 5(1) + 2(-1) & 5(1) + 2(2) \end{bmatrix} =
\frac{1}{3} \begin{bmatrix} 12 & 6 \\ 3 & 9 \end{bmatrix} = \begin{bmatrix} 4 & 2 \\ 1 & 3 \end{bmatrix} = A
$$

✓ **Verified**.

---

### Step 5: Compute $A^4$ via $P D^4 P^{-1}$
* $D^4 = \begin{bmatrix} 5^4 & 0 \\ 0 & 2^4 \end{bmatrix} = \begin{bmatrix} 625 & 0 \\ 0 & 16 \end{bmatrix}$
* $P D^4 = \begin{bmatrix} 2 & -1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 625 & 0 \\ 0 & 16 \end{bmatrix} = \begin{bmatrix} 1250 & -16 \\ 625 & 16 \end{bmatrix}$
* $A^4 = (P D^4) P^{-1} = \frac{1}{3} \begin{bmatrix} 1250 & -16 \\ 625 & 16 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ -1 & 2 \end{bmatrix}$:

$$
A^4 = \frac{1}{3} \begin{bmatrix} 1250(1) - 16(-1) & 1250(1) - 16(2) \\ 625(1) + 16(-1) & 625(1) + 16(2) \end{bmatrix} =
\frac{1}{3} \begin{bmatrix} 1266 & 1218 \\ 609 & 657 \end{bmatrix} = \begin{bmatrix} 422 & 406 \\ 203 & 219 \end{bmatrix}
$$

---

## 16.5 Why this matters in ML

1. **Recurrent Neural Networks (RNNs):** In unrolling an RNN across $T$ timesteps, hidden states multiply recurrent weight matrix $W_h$ repeatedly ($W_h^T$). If the spectral radius $\rho(W_h) = \max |\lambda_i| > 1$, hidden activations explode; if $< 1$, long-term memory vanishes.
2. **Decoupling Dynamical Systems:** Diagonalization transforms coupled differential equations or multi-variable gradient descent dynamics into independent 1D exponential decay rates.

---

## 16.6 Common mistakes

* **Mismatched Column Ordering:** If column 1 of $P$ is eigenvector $\mathbf{v}_1$, then diagonal entry $(1, 1)$ of $D$ **MUST be eigenvalue $\lambda_1$**. Swapping the order in one matrix but not the other breaks the identity $A = P D P^{-1}$.
* **Assuming All Matrices Are Diagonalizable:** Non-square matrices cannot be diagonalized (they require SVD). Defective square matrices with repeated eigenvalues and insufficient independent eigenvectors cannot be diagonalized.

---

> 📖 **Navigation:** [← Previous: Part 15: Eigenvalues & Eigenvectors](./15_eigenvalues_and_eigenvectors.md) | [🏠 Index](./README.md) | [Next: Part 17: Symmetric Matrices & Spectral Theorem →](./17_symmetric_matrices_and_spectral_theorem.md)
