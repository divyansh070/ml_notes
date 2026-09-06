> 📖 **Navigation:** [← Previous: Part 21: Paper-and-Pencil Exam Checklist](./21_paper_and_pencil_checklist.md) | [🏠 Index](./README.md) | [Next: Part 23: Gram-Schmidt Orthogonalization & QR Decomposition →](./23_gram_schmidt_and_qr_decomposition.md)

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

> 📖 **Navigation:** [← Previous: Part 21: Paper-and-Pencil Exam Checklist](./21_paper_and_pencil_checklist.md) | [🏠 Index](./README.md) | [Next: Part 23: Gram-Schmidt Orthogonalization & QR Decomposition →](./23_gram_schmidt_and_qr_decomposition.md)
