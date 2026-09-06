> 📖 **Navigation:** [← Previous: Part 18: Regularization Mathematics (L1 vs. L2)](./18_regularization_mathematics.md) | [🏠 Index](./README.md) | [Next: Part 20: ML Mathematics Roadmap Table →](./20_ml_mathematics_roadmap.md)

---

# PART 18 — ENTROPY & INFORMATION GAIN MATHEMATICS

---

## 18.1 Shannon Entropy Formula: $H(X) = -\sum p_i \log_2 p_i$

**Entropy** measures the degree of uncertainty, disorder, or impurity in a probability distribution:

$$
H(S) = - \sum_{i=1}^{C} p_i \log_2(p_i)
$$

* Units: **Bits** (when using $\log_2$).

---

## 18.2 Hand Calculation: 50/50 Split vs. Pure Split

### Case 1: Maximum Disorder (50% Cat, 50% Dog)
* $p(\text{cat}) = 0.5$, $p(\text{dog}) = 0.5$

$$
H = - [0.5 \log_2(0.5) + 0.5 \log_2(0.5)] = - [0.5(-1) + 0.5(-1)] = - [-0.5 - 0.5] = \mathbf{1.0 \text{ Bit}}
$$

*(Maximum possible entropy for binary classification; completely unpredictable).*

### Case 2: Complete Certainty / Pure Node (100% Cat, 0% Dog)
* $p(\text{cat}) = 1.0$, $p(\text{dog}) = 0.0$ ($0 \log_2 0 \equiv 0$ by limit):

$$
H = - [1.0 \log_2(1.0) + 0] = - [1.0(0) + 0] = \mathbf{0.0 \text{ Bits}}
$$

*(Zero uncertainty; completely pure).*

---

## 18.3 Information Gain & Gini Impurity Hand Trace

### 1. Information Gain (Decision Tree Split Criterion)

$$
\text{Information Gain} = H(\text{Parent}) - \sum_{v \in \text{Children}} \frac{N_v}{N} H(v)
$$

* **Goal:** A Decision Tree chooses the feature split that **maximizes Information Gain** (creates children with the lowest combined entropy).

### 2. Gini Impurity (Faster Alternative)

$$
\text{Gini}(S) = 1 - \sum_{i=1}^{C} p_i^2
$$

* Pure Node: $\text{Gini} = 1 - (1.0)^2 = \mathbf{0.0}$.
* 50/50 Split: $\text{Gini} = 1 - (0.5^2 + 0.5^2) = 1 - (0.25 + 0.25) = \mathbf{0.50}$.
* *Why Scikit-Learn defaults to Gini:* Computing squared sums ($p_i^2$) is much faster for CPUs than computing logarithms ($\log_2 p_i$).

---

> 📖 **Navigation:** [← Previous: Part 18: Regularization Mathematics (L1 vs. L2)](./18_regularization_mathematics.md) | [🏠 Index](./README.md) | [Next: Part 20: ML Mathematics Roadmap Table →](./20_ml_mathematics_roadmap.md)
