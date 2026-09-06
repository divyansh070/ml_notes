> 📖 **Navigation:** [← Previous: Part 18: Regularization Mathematics](./18_regularization_mathematics.md) | [🏠 Index](./README.md) | [Next: Part 20: ML Mathematics Roadmap Table →](./20_ml_mathematics_roadmap.md)

---

# PART 19 — ENTROPY & INFORMATION GAIN MATHEMATICS

Information theory provides the mathematical criterion used by Decision Trees (ID3, C4.5, CART) and Classification loss functions (Cross-Entropy).

---

## 19.1 Shannon Entropy ($H(S)$)

**Entropy** measures the degree of uncertainty or impurity in a probability distribution:

$$
H(S) = - \sum_{i=1}^{C} p_i \log_2(p_i)
$$

* **Pure Node (All class 1):** $H = -(1.0 \log_2 1.0) = \mathbf{0.0}$ (Zero uncertainty).
* **Maximum Impurity (50/50 Binary Split):** $H = -(0.5 \log_2 0.5 + 0.5 \log_2 0.5) = -(0.5(-1) + 0.5(-1)) = \mathbf{1.0 \text{ bit}}$.

---

## 19.2 Gini Impurity ($\text{Gini}(S)$)

Used in CART decision trees (Scikit-Learn's default):

$$
\text{Gini}(S) = 1 - \sum_{i=1}^{C} p_i^2
$$

* **Pure Node:** $\text{Gini} = 1 - (1.0)^2 = \mathbf{0.0}$.
* **50/50 Binary Split:** $\text{Gini} = 1 - (0.5^2 + 0.5^2) = 1 - 0.5 = \mathbf{0.5}$.

---

## 19.3 Information Gain in Decision Tree Splits (Hand Calculation)

**Information Gain** measures the reduction in entropy achieved by splitting on a feature $A$:

$$
IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)
$$

### Step-by-Step Hand Example:
* Parent dataset: $S$ with 14 samples (9 Positive, 5 Negative).
* $H(S) = -\left(\frac{9}{14} \log_2 \frac{9}{14} + \frac{5}{14} \log_2 \frac{5}{14}\right) \approx \mathbf{0.940}$.
* Feature split:
  * Left Child ($S_{\text{left}}$): 6 Positive, 2 Negative (8 samples total). $H(S_{\text{left}}) = -\left(\frac{6}{8}\log_2\frac{6}{8} + \frac{2}{8}\log_2\frac{2}{8}\right) \approx \mathbf{0.811}$.
  * Right Child ($S_{\text{right}}$): 3 Positive, 3 Negative (6 samples total). $H(S_{\text{right}}) = -\left(0.5\log_2 0.5 + 0.5\log_2 0.5\right) = \mathbf{1.000}$.
* **Weighted Child Entropy:**
  $$
  H(S \mid A) = \frac{8}{14}(0.811) + \frac{6}{14}(1.000) = 0.463 + 0.429 = \mathbf{0.892}
  $$
* **Information Gain:**
  $$
  IG(S, A) = 0.940 - 0.892 = \mathbf{0.048 \text{ bits}}
  $$

---

> 📖 **Navigation:** [← Previous: Part 18: Regularization Mathematics](./18_regularization_mathematics.md) | [🏠 Index](./README.md) | [Next: Part 20: ML Mathematics Roadmap Table →](./20_ml_mathematics_roadmap.md)
