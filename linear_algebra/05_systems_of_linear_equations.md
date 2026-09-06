> 📖 **Navigation:** [← Previous: Part 04: Determinants & Geometric Scaling](./04_determinants.md) | [🏠 Index](./README.md) | [Next: Part 06: Linear Independence, Basis & Rank →](./06_linear_independence_and_rank.md)

---

# PART 5 — SYSTEMS OF LINEAR EQUATIONS ($A\mathbf{x} = \mathbf{b}$)

The fundamental problem of linear algebra is solving a system of simultaneous linear equations:

$$
A\mathbf{x} = \mathbf{b}
$$

where $A \in \mathbb{R}^{m \times n}$ is the coefficient matrix, $\mathbf{x} \in \mathbb{R}^n$ is the unknown vector, and $\mathbf{b} \in \mathbb{R}^m$ is the target output vector.

---

## 5.1 What Does $A\mathbf{x} = \mathbf{b}$ Mean Geometrically?

There are two completely different, equally profound geometric ways to view the equation $A\mathbf{x} = \mathbf{b}$:

```
     1. THE ROW PICTURE (Intersection of Planes)      2. THE COLUMN PICTURE (Combining Vectors)
                     y                                                y
                     │      / Line 1 (Row 1)                          │         b = x1*a1 + x2*a2
                     │     /                                          │            ●
                     │    ● (x1, x2) Intersection                     │           / ╱
                     │   / \                                          │   x1*a1  / ╱ x2*a2
                     └──┴───\────────► x                              │         ● ╱
                             \ Line 2 (Row 2)                         └─────────┴────────► x
```

### 1. The Row Picture (Intersection of Hyperplanes)
Each row represents a linear constraint (a line in 2D, a plane in 3D, or an $(n-1)$-dimensional hyperplane in $\mathbb{R}^n$).
* Solving $A\mathbf{x} = \mathbf{b}$ means finding the **geometric point where all $m$ hyperplanes intersect simultaneously**.

### 2. The Column Picture (Linear Combination of Feature Vectors)
Write $A$ as a collection of column vectors $A = [\mathbf{a}_1, \mathbf{a}_2, \dots, \mathbf{a}_n]$:
$$
x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \dots + x_n \mathbf{a}_n = \mathbf{b}
$$
* Solving $A\mathbf{x} = \mathbf{b}$ means finding the **exact scalar weights $(x_1, \dots, x_n)$ needed to combine the column vectors to reach the target point $\mathbf{b}$**.
* **Crucial Rule:** $A\mathbf{x} = \mathbf{b}$ is solvable if and only if **$\mathbf{b}$ lies within the Column Space $\text{Col}(A)$** (the span of the columns of $A$).

---

## 5.2 The Three Solution Scenarios (Geometric & Algebraic)

For any linear system $A\mathbf{x} = \mathbf{b}$, exactly **one** of three mathematical scenarios must occur:

```
  1. UNIQUE SOLUTION (Consistent)      2. INFINITE SOLUTIONS (Underdetermined)    3. NO SOLUTION (Inconsistent)
           y                                       y                                      y
           │      / Line 1                         │                                      │      / Line 1
           │     ● Intersection                    │    ════════ Line 1 & Line 2          │     /
           │    / \ Line 2                         │     (Coincident / Overlapping)       │    /   / Line 2 (Parallel)
           └───┴───\────────► x                    └───┴──────────────────────► x         └───┴───┴──────► x
```

| Scenario | Geometric Row Picture | Geometric Column Picture | Rank Condition (Rouché-Capelli) |
| :--- | :--- | :--- | :--- |
| **1. Unique Solution** | Hyperplanes intersect at a single unique point. | Columns of $A$ are independent; $\mathbf{b} \in \text{Col}(A)$. | $\text{rank}(A) = \text{rank}([A \mid \mathbf{b}]) = n$ |
| **2. Infinite Solutions** | Hyperplanes overlap along a line, plane, or higher subspace. | Columns are dependent (redundant features); $\mathbf{b} \in \text{Col}(A)$. | $\text{rank}(A) = \text{rank}([A \mid \mathbf{b}]) < n$ |
| **3. No Solution** | Hyperplanes are parallel and never meet. | Target $\mathbf{b}$ lies outside the span of columns ($\mathbf{b} \notin \text{Col}(A)$). | $\text{rank}(A) < \text{rank}([A \mid \mathbf{b}])$ |

---

## 5.3 Rank Conditions for Solvability (Rouché-Capelli Theorem)

To check solvability, form the **Augmented Matrix** $[A \mid \mathbf{b}]$ and compare ranks:

1. **Consistent (Solutions Exist):** 
   $$
   \text{rank}(A) = \text{rank}([A \mid \mathbf{b}])
   $$
   *(Adding column $\mathbf{b}$ does not increase the rank because $\mathbf{b}$ is already in the span of $A$'s columns).*
   * If $\text{rank}(A) = n$ (number of variables) $\implies$ **Unique Solution**.
   * If $\text{rank}(A) = r < n \implies$ **Infinitely Many Solutions** with $(n - r)$ free variables.

2. **Inconsistent (No Solution Exists):**
   $$
   \text{rank}(A) < \text{rank}([A \mid \mathbf{b}])
   $$
   *(Adding $\mathbf{b}$ increases the rank because $\mathbf{b} \notin \text{Col}(A)$).*
   * **The ML Connection (Least Squares):** In machine learning, datasets typically have far more equations (samples $N$) than features ($d$). The system $X\mathbf{w} = \mathbf{y}$ is almost always inconsistent ($\mathbf{y} \notin \text{Col}(X)$). **Ordinary Least Squares** finds the best approximate solution by orthogonally projecting $\mathbf{y}$ onto $\text{Col}(X)$!

---

## 5.4 Homogeneous Systems ($A\mathbf{x} = \mathbf{0}$) & The Null Space

A system where the target vector is zero is called **Homogeneous**:

$$
A\mathbf{x} = \mathbf{0}
$$

### Core Properties of Homogeneous Systems:
1. **Always Consistent:** It **always** has at least the **trivial solution** $\mathbf{x} = [0, 0, \dots, 0]^T$.
2. **When do Non-Trivial Solutions ($\mathbf{x} \neq \mathbf{0}$) exist?**
   * Non-trivial solutions exist if and only if matrix $A$ is **rank-deficient** ($\text{rank}(A) < n$, or $\det(A) = 0$ for square matrices).
3. **The Null Space ($\text{Null}(A)$):**
   * The set of ALL solutions to $A\mathbf{x} = \mathbf{0}$ forms a vector subspace called the **Null Space** (or Kernel) of $A$.
   * Every non-trivial solution $\mathbf{x}_{\text{null}}$ represents a direction in feature space that matrix $A$ completely squashes to zero.

---

## 5.5 Free Variables vs. Pivot Variables: Complete Hand Calculation

Let us solve the underdetermined system:
$$
\begin{aligned}
x_1 + 2x_2 + 3x_3 &= 6 \\
2x_1 + 4x_2 + 7x_3 &= 14
\end{aligned}
$$

### Step 1: Set Up Augmented Matrix $[A \mid \mathbf{b}]$
$$
\left[\begin{array}{ccc|c}
1 & 2 & 3 & 6 \\
2 & 4 & 7 & 14
\end{array}\right]
$$

### Step 2: Eliminate Column 1 below Pivot ($R_2 \leftarrow R_2 - 2R_1$)
$$
\left[\begin{array}{ccc|c}
1 & 2 & 3 & 6 \\
0 & 0 & 1 & 2
\end{array}\right]
$$

### Step 3: Eliminate Column 3 above Pivot ($R_1 \leftarrow R_1 - 3R_2$)
$$
\left[\begin{array}{ccc|c}
1 & 2 & 0 & 0 \\
0 & 0 & 1 & 2
\end{array}\right]
$$

### Step 4: Identify Pivot vs. Free Variables
* **Pivot Columns:** Column 1 (pivot in row 1) and Column 3 (pivot in row 2) $\implies x_1, x_3$ are **Pivot Variables**.
* **Free Column:** Column 2 has no pivot $\implies x_2$ is a **Free Variable** ($x_2 = t$, any real number).

### Step 5: Express the General Solution in Vector Form
From Row 1: $x_1 + 2x_2 = 0 \implies x_1 = -2t$.
From Row 2: $x_3 = 2$.

$$
\mathbf{x} =
\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}
=
\begin{bmatrix} -2t \\ t \\ 2 \end{bmatrix}
=
\underbrace{\begin{bmatrix} 0 \\ 0 \\ 2 \end{bmatrix}}_{\mathbf{x}_p \text{ (Particular Solution)}}
+
t \underbrace{\begin{bmatrix} -2 \\ 1 \\ 0 \end{bmatrix}}_{\mathbf{x}_h \text{ (Homogeneous Null Space Basis)}}
$$

* **Geometric Meaning:** The solution set is a 1-dimensional line in 3D space, passing through the particular point $(0, 0, 2)$ and running parallel to the null space direction $[-2, 1, 0]^T$.

---

> 📖 **Navigation:** [← Previous: Part 04: Determinants & Geometric Scaling](./04_determinants.md) | [🏠 Index](./README.md) | [Next: Part 06: Linear Independence, Basis & Rank →](./06_linear_independence_and_rank.md)
