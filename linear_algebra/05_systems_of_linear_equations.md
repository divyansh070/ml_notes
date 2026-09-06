> 📖 **Navigation:** [← Previous: Part 04: Determinants & Geometric Scaling](./04_determinants.md) | [🏠 Index](./README.md) | [Next: Part 06: Linear Independence & Matrix Rank →](./06_linear_independence_and_rank.md)

---

# PART 4 — SYSTEMS OF LINEAR EQUATIONS ($Ax = b$)

---

## 4.1 Simultaneous Equations to Matrix Form

Consider a system of 2 linear equations with 2 unknowns:

$$
\begin{aligned}
2x + y &= 5 \\
x - y &= 1
\end{aligned}
$$

### Converting to Matrix-Vector Form:

$$
\begin{bmatrix}
2 & 1 \\
1 & -1
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix} =
\begin{bmatrix}
5 \\
1
\end{bmatrix}
\quad \iff \quad A \mathbf{x} = \mathbf{b}
$$

Where:
* The **Coefficient Matrix** $A$:

$$
A =
\begin{bmatrix}
2 & 1 \\
1 & -1
\end{bmatrix}
$$

* $\mathbf{x} = [x, y]^T$ is the **Vector of Unknowns**.
* $\mathbf{b} = [5, 1]^T$ is the **Output / Target Vector**.

---

## 4.2 Solving by Elimination & Matrix Inversion

### Method 1: Algebraic Elimination (Fast on Paper)
1. Add equation (1) and equation (2):

$$
(2x + y) + (x - y) = 5 + 1 \implies 3x = 6 \implies x = 2
$$

2. Substitute $x = 2$ back into equation (2):

$$
2 - y = 1 \implies y = 1
$$

3. Solution: $\mathbf{x} = [2, 1]^T$.

### Method 2: Matrix Inversion ($\mathbf{x} = A^{-1} \mathbf{b}$)
1. Compute $\det(A)$:

$$
\det(A) = (2 \times -1) - (1 \times 1) = -2 - 1 = -3
$$

2. Compute $A^{-1}$:

$$
A^{-1} = \frac{1}{-3}
\begin{bmatrix}
-1 & -1 \\
-1 & 2
\end{bmatrix} =
\begin{bmatrix}
1/3 & 1/3 \\
1/3 & -2/3
\end{bmatrix}
$$

3. Multiply $A^{-1} \mathbf{b}$:

$$
\mathbf{x} =
\begin{bmatrix}
1/3 & 1/3 \\
1/3 & -2/3
\end{bmatrix}
\begin{bmatrix}
5 \\
1
\end{bmatrix} =
\begin{bmatrix}
(5/3 + 1/3) \\
(5/3 - 2/3)
\end{bmatrix} =
\begin{bmatrix}
6/3 \\
3/3
\end{bmatrix} =
\begin{bmatrix}
2 \\
1
\end{bmatrix}
$$

---

## 4.3 The Three Solution Scenarios (Unique, None, Infinite)

```
      UNIQUE SOLUTION (det != 0)           NO SOLUTION (Parallel)         INFINITE SOLUTIONS (Same Line)
               y                                     y                                    y
               │      ╲   ╱                          │    ╱   ╱                           │    ╱ (Lines
               │       ╲ ╱                           │   ╱   ╱                            │   ╱   overlap
               │        ● Intersection               │  ╱   ╱  No intersection            │  ╱    completely)
               └────────┼─────► x                    └─┼───┼────────► x                   └─┼────────► x
```

1. **Unique Solution ($\det(A) \neq 0$):** The two lines cross at exactly one coordinate point. Matrix $A$ has full rank.
2. **No Solution ($\det(A) = 0$, Parallel Lines):** E.g., $x + y = 2$ and $x + y = 5$. The lines never intersect; the equations contradict each other.
3. **Infinitely Many Solutions ($\det(A) = 0$, Dependent Lines):** E.g., $x + y = 2$ and $2x + 2y = 4$. The two equations describe the exact same line.

---

> 📖 **Navigation:** [← Previous: Part 04: Determinants & Geometric Scaling](./04_determinants.md) | [🏠 Index](./README.md) | [Next: Part 06: Linear Independence & Matrix Rank →](./06_linear_independence_and_rank.md)
