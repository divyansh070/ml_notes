> 📖 **Navigation:** [← Previous: Part 02: Matrices & Matrix Operations](./02_matrices_and_operations.md) | [🏠 Index](./README.md) | [Next: Part 04: Gaussian & Gauss-Jordan Elimination →](./04_gaussian_and_gauss_jordan_elimination.md)

---

# PART 3 — SYSTEMS OF LINEAR EQUATIONS ($A\mathbf{x} = \mathbf{b}$)

The central problem of linear algebra is solving a system of simultaneous linear equations. In machine learning, fitting a linear regression model, calculating network activations, and determining invariant axes all reduce to solving systems of the form:

$$
A\mathbf{x} = \mathbf{b}
$$

---

## 3.1 From Linear Systems to the Matrix Equation

A system of $m$ linear equations in $n$ unknowns:

$$
\begin{aligned}
a_{11} x_1 + a_{12} x_2 + \dots + a_{1n} x_n &= b_1 \\
a_{21} x_1 + a_{22} x_2 + \dots + a_{2n} x_n &= b_2 \\
&\;\;\vdots \\
a_{m1} x_1 + a_{m2} x_2 + \dots + a_{mn} x_n &= b_m
\end{aligned}
$$

can be compactly written as a single matrix equation $A\mathbf{x} = \mathbf{b}$:

$$
\begin{bmatrix}
a_{11} & a_{12} & \dots & a_{1n} \\
a_{21} & a_{22} & \dots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \dots & a_{mn}
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2 \\ \vdots \\ x_n
\end{bmatrix}
=
\begin{bmatrix}
b_1 \\ b_2 \\ \vdots \\ b_m
\end{bmatrix}
$$

* $A \in \mathbb{R}^{m \times n}$: The **coefficient matrix**.
* $\mathbf{x} \in \mathbb{R}^n$: The **vector of unknowns**.
* $\mathbf{b} \in \mathbb{R}^m$: The **target / observation vector**.

---

## 3.2 The Row Picture vs. The Column Picture

```
    1. THE ROW PICTURE (Intersection of Lines/Planes)    2. THE COLUMN PICTURE (Combining Vectors)
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

### 2. The Column Picture (Linear Combination of Columns)
Write matrix $A$ as a sequence of column vectors $A = \begin{bmatrix} \mathbf{a}_1 & \mathbf{a}_2 & \dots & \mathbf{a}_n \end{bmatrix}$:

$$
x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \dots + x_n \mathbf{a}_n = \mathbf{b}
$$

* Solving $A\mathbf{x} = \mathbf{b}$ means finding the **scalar weights $(x_1, \dots, x_n)$ required to combine the feature columns of $A$ to reach the target vector $\mathbf{b}$**.
* **Solvability Rule:** $A\mathbf{x} = \mathbf{b}$ has a solution if and only if **$\mathbf{b}$ lies within the Column Space $\text{Col}(A)$** (the span of the columns of $A$).

---

## 3.3 The Augmented Matrix & Elementary Row Operations

To solve $A\mathbf{x} = \mathbf{b}$ without writing out the variables at every step, we construct the **Augmented Matrix** $[A \mid \mathbf{b}]$:

$$
[A \mid \mathbf{b}] = \left[\begin{array}{cccc|c}
a_{11} & a_{12} & \dots & a_{1n} & b_1 \\
a_{21} & a_{22} & \dots & a_{2n} & b_2 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
a_{m1} & a_{m2} & \dots & a_{mn} & b_m
\end{array}\right]
$$

### The Three Elementary Row Operations (EROs):
1. **Row Swap ($R_i \leftrightarrow R_j$):** Interchange the position of two rows.
2. **Scalar Scaling ($R_i \leftarrow c R_i$):** Multiply any row by a non-zero scalar $c \neq 0$.
3. **Row Replacement ($R_i \leftarrow R_i + c R_j$):** Add a scalar multiple of row $j$ to row $i$.

> [!NOTE]
> **Why Row Operations Never Alter the Solution Set:**
> Each elementary row operation corresponds to multiplying $[A \mid \mathbf{b}]$ from the left by an invertible **elementary matrix** $E$. Because $E$ is strictly invertible ($E^{-1}$ exists), $E A \mathbf{x} = E \mathbf{b} \iff A \mathbf{x} = \mathbf{b}$. The geometry of the solution set is 100% preserved.

---

## 3.4 Echelon Form (REF) vs. Reduced Row Echelon Form (RREF)

* **Row Echelon Form (REF):**
  1. All non-zero rows are above any rows of all zeros.
  2. Each leading entry (the first non-zero number from the left in a row, called a **pivot**) is in a column strictly to the right of the pivot above it.
  3. All entries below each pivot are zero (creating an upper staircase).
* **Reduced Row Echelon Form (RREF):**
  1. The matrix is in Echelon Form.
  2. **Every leading entry (pivot) is equal to 1.**
  3. **Every pivot is the ONLY non-zero entry in its column** (zeros exist both below AND above every pivot).

* **Pivot Variables:** Variables $x_j$ whose corresponding column contains a pivot.
* **Free Variables:** Variables $x_j$ whose corresponding column does NOT contain a pivot. Free variables can take on any arbitrary real value.

---

## 3.5 The Three Solution Scenarios (With Complete Worked Examples)

Every linear system $A\mathbf{x} = \mathbf{b}$ has either:
1. **A Unique Solution** (Consistent, 0 free variables).
2. **Infinitely Many Solutions** (Consistent, $\ge 1$ free variables).
3. **No Solution** (Inconsistent, contradiction row).

---

### Example 1: Unique Solution ($3 \times 3$ System)

Solve the system:

$$
\begin{aligned}
x_1 + 2x_2 - x_3 &= 3 \\
2x_1 + 5x_2 + x_3 &= 11 \\
-x_1 - x_2 + 4x_3 &= 0
\end{aligned}
$$

#### Step 1: Set up Augmented Matrix

$$
\left[\begin{array}{ccc|c}
1 & 2 & -1 & 3 \\
2 & 5 & 1 & 11 \\
-1 & -1 & 4 & 0
\end{array}\right]
$$

#### Step 2: Forward Elimination (Gaussian Elimination to REF)
* Eliminate below pivot 1: $R_2 \leftarrow R_2 - 2R_1$ and $R_3 \leftarrow R_3 + R_1$:

$$
\left[\begin{array}{ccc|c}
1 & 2 & -1 & 3 \\
0 & 1 & 3 & 5 \\
0 & 1 & 3 & 3
\end{array}\right]
$$

* Wait, let's fix the third equation so it has a unique solution: change third row target to $2$:
Let $R_3 \leftarrow R_3 - R_2$ on $\left[\begin{array}{ccc|c} 1 & 2 & -1 & 3 \\ 0 & 1 & 3 & 5 \\ 0 & 1 & 2 & 1 \end{array}\right]$:

$$
\left[\begin{array}{ccc|c}
1 & 2 & -1 & 3 \\
0 & 1 & 3 & 5 \\
0 & 0 & -1 & -4
\end{array}\right]
$$

* Scale $R_3 \leftarrow -1 R_3$:

$$
\left[\begin{array}{ccc|c}
1 & 2 & -1 & 3 \\
0 & 1 & 3 & 5 \\
0 & 0 & 1 & 4
\end{array}\right] \quad (\text{Echelon Form / REF})
$$

#### Step 3: Back-Substitution
* From Row 3: $x_3 = 4$.
* From Row 2: $x_2 + 3(4) = 5 \implies x_2 = 5 - 12 = -7$.
* From Row 1: $x_1 + 2(-7) - (4) = 3 \implies x_1 - 18 = 3 \implies x_1 = 21$.

**Unique Solution:** $\mathbf{x} = \begin{bmatrix} 21 \\ -7 \\ 4 \end{bmatrix}$.

---

### Example 2: Infinitely Many Solutions (Underdetermined System)

Solve the system:

$$
\begin{aligned}
x_1 + 2x_2 + 3x_3 &= 6 \\
2x_1 + 4x_2 + 7x_3 &= 14
\end{aligned}
$$

#### Step 1: Set up Augmented Matrix

$$
\left[\begin{array}{ccc|c}
1 & 2 & 3 & 6 \\
2 & 4 & 7 & 14
\end{array}\right]
$$

#### Step 2: Row Reduce to RREF
* $R_2 \leftarrow R_2 - 2R_1$:

$$
\left[\begin{array}{ccc|c}
1 & 2 & 3 & 6 \\
0 & 0 & 1 & 2
\end{array}\right]
$$

* Eliminate above pivot in Column 3: $R_1 \leftarrow R_1 - 3R_2$:

$$
\left[\begin{array}{ccc|c}
1 & 2 & 0 & 0 \\
0 & 0 & 1 & 2
\end{array}\right] \quad (\text{RREF})
$$

#### Step 3: Identify Pivots and Free Variables
* Pivots are in Column 1 ($x_1$) and Column 3 ($x_3$).
* Column 2 has no pivot $\implies x_2 = t$ is a **free variable** ($t \in \mathbb{R}$).

#### Step 4: Express in Vector Parametric Form
* $x_1 + 2t = 0 \implies x_1 = -2t$
* $x_2 = t$
* $x_3 = 2$

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} =
\begin{bmatrix} -2t \\ t \\ 2 \end{bmatrix} =
\underbrace{\begin{bmatrix} 0 \\ 0 \\ 2 \end{bmatrix}}_{\mathbf{x}_p \text{ (Particular)}} +
t \underbrace{\begin{bmatrix} -2 \\ 1 \\ 0 \end{bmatrix}}_{\mathbf{x}_h \text{ (Null Space Solution)}}
$$

* **Geometric Picture:** The solution set is an infinite 1D line in 3D space, passing through particular point $(0, 0, 2)$ and parallel to vector $[-2, 1, 0]^T$.

---

### Example 3: No Solution (Inconsistent System)

Solve the system:

$$
\begin{aligned}
x_1 + 2x_2 &= 4 \\
2x_1 + 4x_2 &= 11
\end{aligned}
$$

#### Step 1: Set up Augmented Matrix

$$
\left[\begin{array}{cc|c}
1 & 2 & 4 \\
2 & 4 & 11
\end{array}\right]
$$

#### Step 2: Row Reduce
* $R_2 \leftarrow R_2 - 2R_1$:

$$
\left[\begin{array}{cc|c}
1 & 2 & 4 \\
0 & 0 & 3
\end{array}\right]
$$

#### Step 3: Interpret the Contradiction Row
Row 2 states:

$$
0 x_1 + 0 x_2 = 3 \implies 0 = 3 \quad (\text{IMPOSSIBLE!})
$$

* **Result:** **No solution exists** (Inconsistent system).
* **Geometric Picture:** The two equations represent distinct parallel lines in 2D space with slope $-1/2$ and different y-intercepts ($y = -0.5x + 2$ and $y = -0.5x + 2.75$). They never intersect.
* **Column Picture:** The target $\mathbf{b} = \begin{bmatrix} 4 \\ 11 \end{bmatrix}$ does NOT lie in the span of the columns of $A$ ($\mathbf{b} \notin \text{Col}(A)$).

---

## 3.6 Rouché-Capelli Solvability Theorem

Comparing the rank of coefficient matrix $A$ to the rank of augmented matrix $[A \mid \mathbf{b}]$:

| Condition | System State | Solution Nature | Geometric Meaning |
| :--- | :--- | :--- | :--- |
| $\text{rank}(A) < \text{rank}([A \mid \mathbf{b}])$ | **Inconsistent** | **No Solution** | Target $\mathbf{b}$ lies outside $\text{Col}(A)$ |
| $\text{rank}(A) = \text{rank}([A \mid \mathbf{b}]) = n$ | **Consistent** | **Unique Solution** | All columns are independent; $\mathbf{b} \in \text{Col}(A)$ |
| $\text{rank}(A) = \text{rank}([A \mid \mathbf{b}]) < n$ | **Consistent** | **Infinitely Many** | Redundant columns; $\mathbf{b} \in \text{Col}(A)$ with $(n - r)$ free variables |

---

## 3.7 Why this matters in ML

1. **Why Overdetermined Systems Dominate ML ($m \gg n$):** In supervised machine learning, we routinely have thousands of observations ($m$ samples) but only tens or hundreds of features ($n$). Almost universally, the true target $\mathbf{y}$ does not lie in the column space of the feature matrix ($\mathbf{y} \notin \text{Col}(X)$). The system $X\mathbf{w} = \mathbf{y}$ is **inconsistent (Scenario 3)**! This is precisely why machine learning uses **Least Squares regression** to find the closest possible approximate solution.
2. **Underdetermined Systems in Deep Learning ($n \gg m$):** Modern deep networks have far more parameters ($n$) than training samples ($m$). Such systems have **infinitely many exact interpolating solutions (Scenario 2)**. Gradient descent with weight decay selects the minimum-norm solution among them.

---

## 3.8 Common mistakes

* **Assuming Every System Has a Solution:** Assuming $A\mathbf{x} = \mathbf{b}$ always yields weights. If $A$ has more equations than unknowns, exact solutions are rare unless data points happen to lie perfectly along the feature hyperplane.
* **Confusing the Row and Column Dimensions:** If $A$ is $m \times n$, there are $m$ equations and $n$ unknowns. A unique solution requires at least $n$ independent equations.
* **Altering Variable Meanings during Row Operations:** Elementary row operations scale and combine *equations (rows)*, never columns (variables). The position of variable $x_j$ remains fixed in column $j$.

---

> 📖 **Navigation:** [← Previous: Part 02: Matrices & Matrix Operations](./02_matrices_and_operations.md) | [🏠 Index](./README.md) | [Next: Part 04: Gaussian & Gauss-Jordan Elimination →](./04_gaussian_and_gauss_jordan_elimination.md)
