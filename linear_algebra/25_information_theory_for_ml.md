> 📖 **Navigation:** [← Previous: Part 24: Optimization and Matrix Calculus](./24_optimization_and_derivatives.md) | [🏠 Index](./README.md) | [Next: Part 26: Paper & Pencil Self-Test Checklist →](./26_paper_and_pencil_checklist.md)

---

# PART 25 — INFORMATION THEORY FOR MACHINE LEARNING

Information theory provides the mathematical foundations for decision tree splitting criteria (ID3, C4.5, CART), probabilistic loss functions (Cross-Entropy), and generative modeling objectives (KL divergence in VAEs and diffusion models).

---

## 25.1 Self-Information & Shannon Entropy

### Self-Information (Surprisal)
If an event $x$ occurs with probability $P(x)$, the amount of surprise or information received upon observing $x$ is:

$$
I(x) = -\log_2 P(x) = \log_2 \frac{1}{P(x)}
$$

* If an event is certain ($P(x) = 1$), $I(x) = 0$ bits (no new information).
* If an event is rare ($P(x) = 0.01$), $I(x) \approx 6.64$ bits (high surprise).

### Shannon Entropy $H(X)$
**Entropy** is the expected value of information across all possible outcomes of a discrete random variable $X$:

$$
H(X) = \mathbb{E}[I(X)] = -\sum_{i=1}^{C} p_i \log_2(p_i)
$$

Where $\sum_{i=1}^C p_i = 1$ and by convention $0 \log_2 0 = 0$.

* **Pure Distribution (Certainty):** $p_1 = 1$, all other $p_i = 0$.
  $$
  H(X) = -(1 \log_2 1) = 0 \text{ bits}
  $$
* **Uniform Distribution (Maximum Uncertainty):** $p_i = \frac{1}{C}$ for all $i$.
  $$
  H_{\max}(X) = -\sum_{i=1}^C \frac{1}{C} \log_2 \frac{1}{C} = \log_2 C
  $$
* **Binary Entropy Function:** For a binary label with $P(y=1) = p$ and $P(y=0) = 1-p$:
  $$
  H(p) = -p \log_2 p - (1-p) \log_2 (1-p)
  $$
  * When $p = 0.5$, $H(0.5) = -(0.5(-1) + 0.5(-1)) = \mathbf{1.0\text{ bit}}$ (maximum).
  * When $p = 0$ or $p = 1$, $H = 0\text{ bits}$.

---

## 25.2 Gini Impurity

Used by the CART (Classification and Regression Trees) algorithm (e.g., Scikit-Learn's default):

$$
\text{Gini}(S) = 1 - \sum_{i=1}^{C} p_i^2 = \sum_{i \neq j} p_i p_j
$$

* **Meaning:** The probability that a randomly chosen element from the set would be incorrectly labeled if it were randomly labeled according to the distribution of labels in the subset.
* **Pure node:** $\text{Gini} = 1 - (1.0)^2 = \mathbf{0.0}$.
* **Balanced binary split ($p = 0.5$):**
  $$
  \text{Gini} = 1 - (0.5^2 + 0.5^2) = 1 - 0.50 = \mathbf{0.50}
  $$

> **Entropy vs. Gini:** Gini impurity is computationally faster because it avoids expensive $\log_2$ evaluations. In practice, both lead to very similar decision tree structures.

---

## 25.3 Information Gain (ID3 / C4.5)

**Information Gain** measures the reduction in entropy (or uncertainty) about target $Y$ achieved by partitioning the dataset $S$ using feature $A$:

$$
IG(S, A) = H(S) - H(S \mid A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)
$$

### Complete Step-by-Step Hand Calculation:
* **Parent Dataset $S$:** 14 total samples: 9 Positive ($+$), 5 Negative ($-$).
  $$
  H(S) = -\left(\frac{9}{14}\log_2\frac{9}{14} + \frac{5}{14}\log_2\frac{5}{14}\right) \approx - (0.643 \times (-0.637) + 0.357 \times (-1.485)) \approx \mathbf{0.940\text{ bits}}
  $$
* **Proposed Split on Feature $A$:**
  * **Left branch ($S_L$):** 8 samples (6 Positive, 2 Negative).
    $$
    H(S_L) = -\left(\frac{6}{8}\log_2\frac{6}{8} + \frac{2}{8}\log_2\frac{2}{8}\right) = -(0.75(-0.415) + 0.25(-2)) \approx \mathbf{0.811\text{ bits}}
    $$
  * **Right branch ($S_R$):** 6 samples (3 Positive, 3 Negative).
    $$
    H(S_R) = -(0.5\log_2 0.5 + 0.5\log_2 0.5) = \mathbf{1.000\text{ bit}}
    $$
* **Weighted Conditional Entropy:**
  $$
  H(S \mid A) = \frac{8}{14}(0.811) + \frac{6}{14}(1.000) = 0.463 + 0.429 = \mathbf{0.892\text{ bits}}
  $$
* **Information Gain:**
  $$
  IG(S, A) = H(S) - H(S \mid A) = 0.940 - 0.892 = \mathbf{0.048\text{ bits}}
  $$
The tree chooses the split feature $A^*$ that maximizes $IG(S, A^*)$.

---

## 25.4 Cross-Entropy & KL Divergence

### Kullback-Leibler (KL) Divergence (Relative Entropy)
Measures the excess surprise from using distribution $Q$ to approximate true distribution $P$:

$$
D_{\text{KL}}(P \parallel Q) = \sum_{i=1}^C p_i \log \left(\frac{p_i}{q_i}\right)
$$

* **Properties:**
  1. $D_{\text{KL}}(P \parallel Q) \ge 0$ (Gibbs' inequality; equals 0 if and only if $P = Q$).
  2. Asymmetric: $D_{\text{KL}}(P \parallel Q) \neq D_{\text{KL}}(Q \parallel P)$ (not a true distance metric).

### Cross-Entropy $H(P, Q)$
Cross-entropy measures the average number of bits needed to identify events from distribution $P$ using coding scheme optimized for $Q$:

$$
H(P, Q) = -\sum_{i=1}^C p_i \log q_i = H(P) + D_{\text{KL}}(P \parallel Q)
$$

### Why Cross-Entropy is the Default Classification Loss:
In machine learning classification, $P$ represents the true one-hot ground-truth label (fixed, so $H(P) = 0$ for a single sample), and $Q = \hat{\mathbf{y}}$ represents the model's predicted softmax probabilities.

$$
\mathcal{L}_{\text{CE}} = H(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{i=1}^C y_i \log \hat{y}_i
$$

Minimizing Cross-Entropy is mathematically identical to:
1. **Minimizing KL Divergence** $D_{\text{KL}}(\mathbf{y} \parallel \hat{\mathbf{y}})$ between predictions and ground truth.
2. **Maximizing Log-Likelihood** (MLE) under a categorical likelihood model.

---

## 25.5 Summary of Key Formulas

| Metric | Formula | Value Range | Primary ML Use Case |
|---|---|---|---|
| **Shannon Entropy** | $H(X) = -\sum p_i \log_2 p_i$ | $[0, \log_2 C]$ | Impurity measurement, ID3 trees |
| **Gini Impurity** | $\text{Gini}(X) = 1 - \sum p_i^2$ | $[0, 1 - 1/C]$ | CART decision trees (Scikit-Learn) |
| **Information Gain** | $IG = H(S) - \sum \frac{\|S_v\|}{\|S\|} H(S_v)$ | $[0, H(S)]$ | Feature selection, splitting rule |
| **KL Divergence** | $D_{\text{KL}}(P \parallel Q) = \sum p_i \log(p_i/q_i)$ | $[0, \infty)$ | VAEs, policy distillation, t-SNE |
| **Cross-Entropy** | $H(P, Q) = -\sum p_i \log q_i$ | $[0, \infty)$ | Logistic regression & Neural network loss |

---

> 📖 **Navigation:** [← Previous: Part 24: Optimization and Matrix Calculus](./24_optimization_and_derivatives.md) | [🏠 Index](./README.md) | [Next: Part 26: Paper & Pencil Self-Test Checklist →](./26_paper_and_pencil_checklist.md)
