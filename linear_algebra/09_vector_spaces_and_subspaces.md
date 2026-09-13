> 📖 **Navigation:** [← Previous: Part 08: Linear Independence, Span, Basis & Dimension](./08_linear_independence_span_basis.md) | [🏠 Index](./README.md) | [Next: Part 10: Four Fundamental Subspaces →](./10_four_fundamental_subspaces.md)

---

# PART 9 — VECTOR SPACES & SUBSPACES

Linear algebra operates inside geometric playgrounds called **Vector Spaces**. In machine learning, understanding subspaces is what allows us to project high-dimensional data onto lower-dimensional manifolds and understand what information neural networks preserve versus discard.

---

## 9.1 What is a Vector Space?

A **Vector Space** $V$ is a collection of mathematical objects (vectors) that can be added together and scaled by real numbers, such that the resulting vectors remain inside $V$.

### The Core Axioms (Intuition):
For all vectors $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ and scalars $c, d \in \mathbb{R}$:
1. **Closure under Addition:** $\mathbf{u} + \mathbf{v} \in V$.
2. **Closure under Scalar Multiplication:** $c\mathbf{u} \in V$.
3. **Zero Vector Existence:** There exists a unique $\mathbf{0} \in V$ such that $\mathbf{v} + \mathbf{0} = \mathbf{v}$.
4. **Additive Inverses:** For every $\mathbf{v}$, there exists $(-\mathbf{v}) \in V$ such that $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$.
5. **Associativity & Commutativity:** $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$ and $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$.

In ML, our primary vector space is Euclidean $d$-dimensional space $\mathbb{R}^d$.

---

## 9.2 Vector Subspaces & The 3-Step Verification Test

A subset $S \subseteq \mathbb{R}^d$ is a **Vector Subspace** if $S$ is itself a complete vector space under standard vector addition and scalar multiplication.

> [!IMPORTANT]
> **The 3-Step Subspace Verification Test:**
> A subset $S \subseteq \mathbb{R}^d$ is a subspace if and only if:
> 1. **Contains Zero:** The origin vector $\mathbf{0} \in S$.
> 2. **Closed under Addition:** If $\mathbf{u}, \mathbf{v} \in S$, then $\mathbf{u} + \mathbf{v} \in S$.
> 3. **Closed under Scaling:** If $\mathbf{u} \in S$ and $c \in \mathbb{R}$, then $c\mathbf{u} \in S$.

### The Golden Subspace Rule:
Any line, plane, or hyperplane that **does NOT pass through the origin $(0, 0, \dots, 0)$ is NOT a subspace** (it fails Axiom 1).

```
          LINEAR SUBSPACE (Passes through Origin)         AFFINE HYPERPLANE (Shifted by Bias b)
                     y                                               y
                     │      / y = Ax                                 │      / y = Ax + b
                     │     / (Subspace!)                             │     / (NOT a Subspace!)
                     │    /                                        b ┼────●
                     └───●──────► x                                  │   /
                      Origin (0, 0)                                  └──┴────────► x
```

---

## 9.3 Linear Subspaces vs. Affine Spaces in Neural Networks

* **Linear Subspace ($A\mathbf{x}$):** Always contains the origin ($A\mathbf{0} = \mathbf{0}$). A purely linear classifier ($\mathbf{w}^T \mathbf{x} = 0$) is forced to pass directly through the origin $(0, 0, \dots, 0)$.
* **Affine Space ($A\mathbf{x} + \mathbf{b}$):** A linear subspace translated by a fixed displacement vector $\mathbf{b}$.
* **Why Neural Networks Require Bias:** Without the bias vector $\mathbf{b}$ in $\mathbf{z} = W\mathbf{x} + \mathbf{b}$, every layer's decision boundary would be trapped at the origin, severely crippling its ability to separate non-centered data.

---

## 9.4 The Null Space ($N(A)$): Solving $A\mathbf{x} = \mathbf{0}$

The **Null Space** (or Kernel) of an $m \times n$ matrix $A$, denoted $N(A)$, is the set of all input vectors $\mathbf{x} \in \mathbb{R}^n$ that $A$ squashes into the zero vector:

$$
N(A) = \left\lbrace \mathbf{x} \in \mathbb{R}^n \;\middle|\; A\mathbf{x} = \mathbf{0} \right\rbrace
$$

### Proof: Why the Null Space is ALWAYS a Subspace of $\mathbb{R}^n$
1. **Contains $\mathbf{0}$:** $A\mathbf{0} = \mathbf{0} \implies \mathbf{0} \in N(A)$.
2. **Closed under Addition:** If $\mathbf{u}, \mathbf{v} \in N(A)$, then $A\mathbf{u} = \mathbf{0}$ and $A\mathbf{v} = \mathbf{0}$. Therefore:

$$
A(\mathbf{u} + \mathbf{v}) = A\mathbf{u} + A\mathbf{v} = \mathbf{0} + \mathbf{0} = \mathbf{0} \implies (\mathbf{u} + \mathbf{v}) \in N(A)
$$

3. **Closed under Scaling:** If $\mathbf{u} \in N(A)$ and $c \in \mathbb{R}$, then:

$$
A(c\mathbf{u}) = c(A\mathbf{u}) = c\mathbf{0} = \mathbf{0} \implies c\mathbf{u} \in N(A)
$$

■

---

## 9.5 Complete Worked Example: Finding the Null Space & Basis

Find the null space and a basis for $N(A)$ for:

$$
A = \begin{bmatrix}
1 & 3 & 1 & 4 \\
2 & 6 & 3 & 9
\end{bmatrix} \in \mathbb{R}^{2 \times 4}
$$

### Step 1: Set Up and Row Reduce $A\mathbf{x} = \mathbf{0}$

$$
\left[\begin{array}{cccc|c}
1 & 3 & 1 & 4 & 0 \\
2 & 6 & 3 & 9 & 0
\end{array}\right]
$$

Eliminate Column 1 in Row 2 ($R_2 \leftarrow R_2 - 2R_1$):

$$
[2, 6, 3, 9 \mid 0] - 2[1, 3, 1, 4 \mid 0] = [0, 0, 1, 1 \mid 0]
$$

Matrix becomes:

$$
\left[\begin{array}{cccc|c}
1 & 3 & 1 & 4 & 0 \\
0 & 0 & 1 & 1 & 0
\end{array}\right]
$$

Eliminate Column 3 in Row 1 ($R_1 \leftarrow R_1 - R_2$):

$$
[1, 3, 1, 4 \mid 0] - [0, 0, 1, 1 \mid 0] = [1, 3, 0, 3 \mid 0]
$$

$$
\left[\begin{array}{cccc|c}
1 & 3 & 0 & 3 & 0 \\
0 & 0 & 1 & 1 & 0
\end{array}\right] \quad (\text{RREF})
$$

### Step 2: Identify Pivot and Free Variables
* **Pivot Columns:** Column 1 ($x_1$) and Column 3 ($x_3$).
* **Free Columns:** Column 2 ($x_2 = s$) and Column 4 ($x_4 = t$).

### Step 3: Express Solution in Parametric Vector Form
* Row 1: $x_1 + 3s + 3t = 0 \implies x_1 = -3s - 3t$
* Row 2: $x_3 + t = 0 \implies x_3 = -t$

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{bmatrix} =
\begin{bmatrix} -3s - 3t \\ s \\ -t \\ t \end{bmatrix} =
s \begin{bmatrix} -3 \\ 1 \\ 0 \\ 0 \end{bmatrix} +
t \begin{bmatrix} -3 \\ 0 \\ -1 \\ 1 \end{bmatrix}
$$

### Step 4: Extract the Null Space Basis
The two direction vectors form the **basis for $N(A)$**:

$$
\mathcal{B}_{N(A)} = \left\lbrace \begin{bmatrix} -3 \\ 1 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix} -3 \\ 0 \\ -1 \\ 1 \end{bmatrix} \right\rbrace
$$

* **Nullity:** $\dim(N(A)) = 2$ (There are 2 free variables).
* **Geometric Meaning:** Any linear combination of these two basis vectors is sent directly to $\mathbf{0}$ by matrix $A$.

---

## 9.6 Why this matters in ML

1. **Information Loss:** The null space $N(W)$ of a neural network weight matrix $W$ contains the exact directions of input data that are **blindly ignored** by that layer ($W\mathbf{x}_{\text{null}} = \mathbf{0}$).
2. **Adversarial Perturbations:** If an adversarial noise vector $\boldsymbol{\delta}$ lies inside the null space of a feature extractor $W$, then $W(\mathbf{x} + \boldsymbol{\delta}) = W\mathbf{x}$. The feature representation remains unchanged.
3. **Exact Multicollinearity:** If design matrix $X$ has a non-trivial null space ($N(X) \neq \{\mathbf{0}\}$), there exist weight changes $\Delta \mathbf{w} \in N(X)$ that produce zero change in model predictions ($X(\mathbf{w} + \Delta \mathbf{w}) = X\mathbf{w}$). Parameter estimation is underdetermined.

---

## 9.7 Common mistakes

* **Thinking Any Flat Surface is a Subspace:** A plane $x_1 + x_2 + x_3 = 5$ is NOT a subspace because $[0, 0, 0]^T$ does not satisfy the equation ($0 + 0 + 0 \neq 5$). Only planes with zero intercept ($x_1 + x_2 + x_3 = 0$) are subspaces.
* **Confusing the Null Space with the Zero Vector:** $N(A)$ is not just the single vector $\mathbf{0}$. It is an entire continuous subspace of all vectors satisfying $A\mathbf{x} = \mathbf{0}$. If $A$ is rank-deficient, $N(A)$ contains infinitely many points.

---

> 📖 **Navigation:** [← Previous: Part 08: Linear Independence, Span, Basis & Dimension](./08_linear_independence_span_basis.md) | [🏠 Index](./README.md) | [Next: Part 10: Four Fundamental Subspaces →](./10_four_fundamental_subspaces.md)
