# ML Gap Notes: Interview Add-ons

A companion to `ml_notes_revision.md` and `transformer_and_beyond.md`. This file only covers what those notes are missing or thin on, based on real Data Science interview rounds (Infoedge-style). Every number in the worked examples was checked by computation.

## Contents

0. [Correction: Dropout Scaling](#0-correction-dropout-scaling)
1. [Regression Gaps](#1-regression-gaps)
2. [Tree & Bagging Gaps](#2-tree--bagging-gaps)
3. [Boosting Trio: Hands-on Math](#3-boosting-trio-hands-on-math)
    - [3.1 XGBoost](#31-xgboost-hands-on-math)
    - [3.2 LightGBM](#32-lightgbm-hands-on-math)
    - [3.3 CatBoost](#33-catboost-hands-on-math)
    - [3.4 Comparison Table](#34-xgboost-vs-lightgbm-vs-catboost)
4. [PCA via the Covariance Matrix](#4-pca-via-the-covariance-matrix)
5. [SVM: Where the Kernel Trick Is Used](#5-svm-where-the-kernel-trick-is-used)
6. [Practical ML](#6-practical-ml)
7. [Deep Learning Gaps](#7-deep-learning-gaps)
8. [NLP Gaps](#8-nlp-gaps)
9. [Generative vs Discriminative & Naive Bayes](#9-generative-vs-discriminative--naive-bayes)
10. [Statistics Essentials](#10-statistics-essentials)
11. [Probability Question Bank](#11-probability-question-bank)
12. [Final Interview Checklist](#12-final-interview-checklist)

---

## 0. Correction: Dropout Scaling

The Dropout section in the main notes describes the **original (2014) convention**: multiply by the probability at test time. Two fixes:

1. If $p$ is the **dropout rate**, the test-time factor is the **keep probability $(1-p)$**, not $p$. The main notes' example uses $p = 0.5$, where both give 0.5, which hides the mistake. For $p = 0.2$ the factor is 0.8.
2. **Modern frameworks (PyTorch, Keras) use inverted dropout.**

**Inverted dropout (what to say in interviews):**
- **Training:** $\tilde{a} = \dfrac{m \odot a}{1-p}$, where $m_i \sim \text{Bernoulli}(1-p)$. Dividing by $(1-p)$ keeps the expected activation unchanged: $E[\tilde{a}] = a$.
- **Inference:** dropout is **off**, and no scaling is applied at all.
- **Why inverted?** Inference code stays untouched, and the rescaling cost moves to training.

**Dropout ≈ Bagging (asked directly):**
- Each mini-batch trains a different randomly "thinned" sub-network. With $n$ droppable units there are $2^n$ possible sub-networks.
- Like bagging, each member sees a different random version of the problem, and averaging reduces variance.
- The difference is that bagging trains independent models, while dropout's sub-networks **share weights**.
- At inference, the full scaled network approximates the **average (geometric mean) of the ensemble's predictions**.

---

## 1. Regression Gaps

### 1.1 VIF (Variance Inflation Factor)

**Procedure to detect multicollinearity:**
1. For each feature $X_j$, fit a regression of $X_j$ on **all the other features**.
2. Record that regression's $R_j^2$.
3. Compute $\text{VIF}_j = \dfrac{1}{1 - R_j^2}$.
4. Repeat for every feature.

| VIF | Meaning |
|---|---|
| 1 | No collinearity ($R_j^2 = 0$) |
| 1 to 5 | Moderate, usually fine |
| > 5 | Concerning |
| > 10 | Serious |

**Why does VIF depend on $R^2$?** $R_j^2$ measures how much of $X_j$ the other features already explain. That is exactly what multicollinearity means. If $R_j^2 = 0.9$, then VIF = 10.

**Where does the name come from?** $\text{Var}(\hat\beta_j) = \dfrac{\sigma^2}{(n-1)\,\text{Var}(X_j)} \cdot \dfrac{1}{1-R_j^2}$. The VIF is literally the factor by which collinearity **inflates the variance of the coefficient estimate**. That makes coefficients unstable, gives them wide confidence intervals, and can even flip their signs.

**Fixes:** drop one of the correlated features, combine them (for example, take a ratio or average), use PCA, or use Ridge (L2) regression.

### 1.2 Multicollinearity vs Autocorrelation

| | Multicollinearity | Autocorrelation |
|---|---|---|
| What's correlated | Feature **columns** with each other | **Residuals** across **rows**, usually over time |
| Which assumption | No multicollinearity in $X$ | Independence of errors |
| Detect | VIF, correlation matrix | Durbin–Watson test (≈ 2 means none; < 2 positive; > 2 negative), ACF plot |
| Consequence | Unstable coefficients with large variance | Standard errors underestimated, so p-values look too small |
| Fix | Drop or combine features, Ridge, PCA | Add lag features, use time-series models (ARIMA), GLS |

### 1.3 Correlation vs Covariance vs Collinearity

- **Covariance:** $\text{Cov}(X,Y) = E[(X-\mu_X)(Y-\mu_Y)]$, estimated as $\frac{1}{n-1}\sum (x_i-\bar{x})(y_i-\bar{y})$. It gives the direction of the relationship, but its magnitude depends on the units.
- **Correlation:** $r = \dfrac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$. It is unit-free, lies in $[-1, 1]$, and measures **linear** association. Spearman correlation (computed on ranks) captures any **monotonic** relationship.
- **Collinearity:** a *modelling problem*, where one feature is (nearly) a linear function of others. High pairwise correlation is one symptom, but collinearity can also involve **3 or more features jointly** even when every pairwise correlation is low. That is why VIF is better than looking at a correlation matrix.

### 1.4 R² and Adjusted R²

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}, \qquad R^2_{adj} = 1 - (1-R^2)\frac{n-1}{n-p-1}
$$

- **The biggest flaw of $R^2$:** it **never decreases** when you add a feature, even a pure-noise one, because least squares can always set that feature's coefficient to 0.
- **Adjusted $R^2$** penalizes each additional predictor $p$. It rises only if the new feature improves the fit more than chance would.

### 1.5 Logistic Regression: Two Classic "Why" Questions

**Why is it called "regression"?** It **linearly regresses the log-odds**:

$$
\log\frac{p}{1-p} = w^Tx + b
$$

It is a Generalized Linear Model (GLM) with a logit link. The output is continuous (a probability), and it becomes a classifier only after you apply a threshold.

**Why sigmoid and not tanh?**
1. Sigmoid's output range $(0,1)$ can be read directly as a **Bernoulli probability**. Tanh's range is $(-1,1)$.
2. Sigmoid is exactly the **inverse of the logit**: solving the log-odds equation for $p$ gives $p = \sigma(w^Tx+b)$. It falls out of the model rather than being chosen arbitrarily.
3. With log-loss, the gradient simplifies to $(p - y)x$, which gives a clean, convex optimization problem.

(Note that $\tanh(z) = 2\sigma(2z) - 1$, so tanh is just a rescaled sigmoid. The issue is the range and the interpretation, not the shape.)

---

## 2. Tree & Bagging Gaps

### 2.1 Entropy vs Gini Impurity

$$
\text{Entropy} = -\sum_k p_k \log_2 p_k \qquad \text{Gini} = 1 - \sum_k p_k^2
$$

**Worked example:** a node with 6 positive and 4 negative samples.
- Gini $= 1 - (0.6^2 + 0.4^2) = 1 - 0.52 = 0.48$
- Entropy $= -0.6\log_2 0.6 - 0.4\log_2 0.4 = 0.442 + 0.529 = 0.971$

**Entropy vs impurity:** "impurity" is the general concept (how mixed a node is). Entropy and Gini are two ways to measure it. Entropy is an information-theory measure (bits of uncertainty), and the reduction in entropy after a split is called *Information Gain*.

**Advantages of Gini over entropy:**
- It needs no logarithm, so it is **faster to compute**. This matters because split-finding evaluates it millions of times.
- It is the default in CART and scikit-learn.
- In practice, the two criteria pick the same split in the vast majority of cases. Entropy is slightly more sensitive to rare classes.

### 2.2 Bagging: How Data Is Sampled

- Each tree gets a **bootstrap sample**: $n$ rows drawn *with replacement* from the $n$ training rows.
- The probability that a specific row is never picked is $(1 - \frac{1}{n})^n \to e^{-1} \approx 0.368$.
- So each tree sees about **63.2% unique rows**. The remaining ~36.8% are **out-of-bag (OOB)** rows, which provide a free validation set.

### 2.3 Random Forest: What Is "Random"?

1. **Row randomness:** each tree is trained on its own bootstrap sample.
2. **Column randomness:** at **every split** (not once per tree), only a random subset of features is considered. The defaults are $\sqrt{p}$ for classification and $p/3$ for regression.

**Why the feature sampling?** Without it, one strong feature would dominate the top split of every tree, making the trees highly correlated. Averaging correlated trees removes little variance. Feature sampling **decorrelates** the trees, and the variance of an average of trees with correlation $\rho$ is $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$.

---

## 3. Boosting Trio: Hands-on Math

All three libraries share the same **second-order (Newton) core**. For each row, compute:

$$
g_i = \frac{\partial L}{\partial \hat{y}_i}, \qquad h_i = \frac{\partial^2 L}{\partial \hat{y}_i^2}
$$

For a node with $G = \sum g_i$ and $H = \sum h_i$:

$$
\text{Score (similarity)} \; S = \frac{G^2}{H+\lambda}, \qquad \text{Leaf weight} \; w^* = -\frac{G}{H+\lambda}
$$

$$
\text{Gain} = \frac{1}{2}\Big[S_L + S_R - S_{parent}\Big] - \gamma
$$

| Loss | $g_i$ | $h_i$ |
|---|---|---|
| MSE $\frac{1}{2}(y-\hat{y})^2$ | $\hat{y}_i - y_i$ | $1$ |
| Log-loss (on log-odds $\hat{y}$) | $p_i - y_i$ | $p_i(1-p_i)$ |

**Sign convention used throughout this file:** $g = \hat{y} - y$ (prediction minus target), which is the true derivative of the loss. When the model **under-predicts**, $g < 0$, so $w^* = -G/(H+\lambda) > 0$ and the prediction moves **up**. (StatQuest uses residuals $y - \hat{y}$ instead. Its "similarity score" $(\text{sum of residuals})^2 / (n + \lambda)$ is identical because the sign is squared away, and its output value $(\text{sum of residuals}) / (n + \lambda)$ equals $w^*$.)

> [!WARNING]
> The LightGBM PDF's first regression example flips sign conventions halfway through the calculation. The correct first-round predictions for that data are 22.36 (left leaf) and 27.12 (right leaf), which is where its "corrected" version ends up. Stick to $g = \hat{y} - y$ consistently.

The three libraries differ in **how they search for splits and grow trees**. That's what the examples below focus on.

---

### 3.1 XGBoost: Hands-on Math

**What to show:** exact greedy split search, level-wise growth, and how $\lambda$ and $\gamma$ prevent overfitting.

**Setup:** loss = MSE, $\lambda = 1$, $\gamma = 5$, $\eta = 0.3$, `max_depth = 2`

| $i$ | $x$ | $y$ |
|---|---|---|
| 1 | 1 | 4 |
| 2 | 2 | 6 |
| 3 | 3 | 5 |
| 4 | 4 | 15 |
| 5 | 5 | 17 |
| 6 | 6 | 16 |

#### Step 0: Base prediction

$$
\hat{y}^{(0)} = \bar{y} = \frac{4+6+5+15+17+16}{6} = 10.5
$$

(Older XGBoost versions used `base_score = 0.5` by default. Newer versions estimate it from the data. The mean is the optimal constant for MSE.)

#### Step 1: Gradients and Hessians
$g_i = \hat{y}_i - y_i$, $h_i = 1$

| $i$ | $x$ | $y$ | $g_i$ | $h_i$ |
|---|---|---|---|---|
| 1 | 1 | 4 | +6.5 | 1 |
| 2 | 2 | 6 | +4.5 | 1 |
| 3 | 3 | 5 | +5.5 | 1 |
| 4 | 4 | 15 | −4.5 | 1 |
| 5 | 5 | 17 | −6.5 | 1 |
| 6 | 6 | 16 | −5.5 | 1 |

Root: $G = 0$, $H = 6$, so $S_{root} = 0^2/(6+1) = 0$.

#### Step 2: Exact greedy search at the root
XGBoost's exact method **sorts** $x$ and tries **every midpoint** between consecutive values (5 candidates here).

| Split | $G_L, H_L$ | $G_R, H_R$ | $S_L$ | $S_R$ | Gain $=\frac{1}{2}(S_L+S_R-0)$ |
|---|---|---|---|---|---|
| $x<1.5$ | 6.5, 1 | −6.5, 5 | 21.125 | 7.042 | 14.083 |
| $x<2.5$ | 11, 2 | −11, 4 | 40.333 | 24.200 | 32.267 |
| $x<3.5$ | 16.5, 3 | −16.5, 3 | 68.063 | 68.063 | **68.063** |
| $x<4.5$ | 12, 4 | −12, 2 | 28.800 | 48.000 | 38.400 |
| $x<5.5$ | 5.5, 5 | −5.5, 1 | 5.042 | 15.125 | 10.083 |

The best split is $x < 3.5$. Checking against $\gamma$: $68.063 - 5 = 63.06 > 0$, so the split is **kept**.

#### Step 3: Level-wise growth (depth 2)
XGBoost grows **level by level**: it tries to split *every* node at depth 1.

**Left node** $\{1,2,3\}$: $G = 16.5$, $H = 3$, $S_{parent} = 16.5^2/4 = 68.063$

| Split | $S_L$ | $S_R$ | Gain |
|---|---|---|---|
| $x<1.5$ | $6.5^2/2 = 21.125$ | $10^2/3 = 33.333$ | $\frac{1}{2}(54.458 - 68.063) = -6.80$ |
| $x<2.5$ | $11^2/3 = 40.333$ | $5.5^2/2 = 15.125$ | $\frac{1}{2}(55.458 - 68.063) = -6.30$ |

**Right node** $\{4,5,6\}$: $G = -16.5$, $H = 3$, $S_{parent} = 68.063$

| Split | $S_L$ | $S_R$ | Gain |
|---|---|---|---|
| $x<4.5$ | 10.125 | 48.000 | −4.97 |
| $x<5.5$ | 40.333 | 15.125 | −6.30 |

All gains are **negative**, so neither node splits. **$\lambda$ alone blocked these splits**: with $\lambda = 1$, splitting 3 rows into groups of 1 and 2 costs more than it gains.

**What if $\lambda = 0$?** The left node's $S_{parent}$ becomes $16.5^2/3 = 90.75$, and the split $x<1.5$ gives $\frac{1}{2}(42.25 + 50 - 90.75) = 0.75 > 0$, so the tree *would* split to fit the noise among 4, 6, and 5. **$\gamma$ then catches it**: $0.75 - 5 < 0$, so the branch is **pruned**. This is the whole idea: $\lambda$ shrinks scores and $\gamma$ sets a minimum gain, and together they stop the tree from chasing noise.

(XGBoost first grows to `max_depth`, then prunes **bottom-up**: any branch with $\text{Gain} - \gamma < 0$ is removed, unless a split below it survives.)

#### Step 4: Leaf weights

$$
w_L = -\frac{16.5}{3+1} = -4.125, \qquad w_R = -\frac{-16.5}{3+1} = +4.125
$$

Compare these with the raw average residuals ($\mp 5.5$): **$\lambda$ shrinks the leaf outputs toward 0**, just like Ridge regression.

#### Step 5: Update the predictions

$$
\hat{y}^{(1)} = \hat{y}^{(0)} + \eta \cdot w
$$

- Rows 1–3: $10.5 + 0.3(-4.125) = 9.2625$
- Rows 4–6: $10.5 + 0.3(4.125) = 11.7375$

MSE drops from **30.92 to 18.84**. The next tree is fit to the new gradients (for example, row 1: $9.2625 - 4 = +5.2625$).

#### Classification note: cover / `min_child_weight`
For log-loss, $h_i = p_i(1-p_i)$. At the start ($p = 0.5$) each row has $h = 0.25$, so a leaf with 3 rows has **cover** $H = 0.75$. With the default `min_child_weight = 1`, that leaf is **not allowed**. Beginners are often surprised when trees refuse to split on small classification data, and this is why: `min_child_weight` limits the sum of Hessians, not the row count.

#### XGBoost improvements over plain GBM (interview answer)
1. **Second-order Taylor expansion:** the Newton step uses curvature ($h$), not just the gradient.
2. **Regularized objective** $\Omega = \gamma T + \frac{1}{2}\lambda\sum w_j^2$ (plus an optional $\alpha$ L1 term). This produces the similarity score and gain formulas above.
3. **Pruning with $\gamma$**, applied bottom-up after the tree is grown.
4. **Sparsity-aware splits:** missing values learn a default direction at each split.
5. **Engineering:** parallel split finding, cache-aware blocks, and a histogram method (`tree_method='hist'`).
6. **Shrinkage ($\eta$) and column subsampling.**

---

### 3.2 LightGBM: Hands-on Math

**What to show:** histogram split search, the subtraction trick, **leaf-wise** growth, and **GOSS**. (These are the parts that differ from XGBoost; the PDF covers the basic gain math.)

**Setup:** MSE, $\lambda = 1$, $\eta = 0.3$, `num_leaves = 3`

| $i$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| $x$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| $y$ | 3 | 4 | 3 | 5 | 20 | 22 | 40 | 41 |

#### Step 0 and Step 1
$\hat{y}^{(0)} = \bar{y} = 138/8 = 17.25$. With $g_i = 17.25 - y_i$ and $h_i = 1$:

| $i$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| $g_i$ | 14.25 | 13.25 | 14.25 | 12.25 | −2.75 | −4.75 | −22.75 | −23.75 |

Root: $G = 0$, $H = 8$.

#### Step 2: Build the histogram (4 bins)
LightGBM bins $x$ **once**, before training starts (the default is up to 255 bins).

| Bin | $x$ values | $G_{bin}$ | $H_{bin}$ |
|---|---|---|---|
| 0 | 1, 2 | 27.5 | 2 |
| 1 | 3, 4 | 26.5 | 2 |
| 2 | 5, 6 | −7.5 | 2 |
| 3 | 7, 8 | −46.5 | 2 |

#### Step 3: Split search over bin boundaries (only 3 candidates, versus 7 for exact greedy)

| Split | $G_L, H_L$ | $G_R, H_R$ | $S_L$ | $S_R$ | Gain |
|---|---|---|---|---|---|
| bin 0 \| 1 ($x \le 2$) | 27.5, 2 | −27.5, 6 | 252.08 | 108.04 | 180.06 |
| bin 1 \| 2 ($x \le 4$) | 54, 4 | −54, 4 | 583.20 | 583.20 | **583.20** |
| bin 2 \| 3 ($x \le 6$) | 46.5, 6 | −46.5, 2 | 308.89 | 720.75 | 514.82 |

The best root split is $x \le 4$. With $N$ rows and $K$ bins, the search costs $O(K)$ per feature instead of $O(N)$ after sorting.

#### Step 4: The subtraction trick
To split the children further, LightGBM needs a histogram for each child. It **scans only the smaller child**, then gets the other one for free:

$$
\text{Hist}_{right} = \text{Hist}_{parent} - \text{Hist}_{left}
$$

For example, the right child's bin 3 is $(-46.5, 2) - (0, 0) = (-46.5, 2)$. That's $O(K)$ work with no pass over the data.

#### Step 5: Leaf-wise growth
After the root split there are two leaves:
- $L = \{1..4\}$: $G = 54$, $H = 4$, $S = 583.2$
- $R = \{5..8\}$: $G = -54$, $H = 4$, $S = 583.2$

LightGBM finds the **best split within every current leaf**, then splits **only the single best leaf in the whole tree**:

| Leaf | Best split | Children $S$ | Gain |
|---|---|---|---|
| $L$ | $x \le 2$ | $27.5^2/3 = 252.08$, $26.5^2/3 = 234.08$ | $\frac{1}{2}(486.17 - 583.2) = -48.52$ |
| $R$ | $x \le 6$ | $(-7.5)^2/3 = 18.75$, $(-46.5)^2/3 = 720.75$ | $\frac{1}{2}(739.5 - 583.2) = \mathbf{78.15}$ |

So **leaf $R$ is split**, and that uses up the budget of `num_leaves = 3`. Level-wise growth would instead expand *all* depth-1 nodes, spending leaves where the gain is small. Leaf-wise growth always spends the next leaf wherever the loss drops most, which is why it gives lower loss for the same number of leaves but can create deep, narrow branches (hence `num_leaves`, `max_depth`, and `min_data_in_leaf`).

#### Step 6: Leaf weights and update

| Leaf | Rows | $G$ | $w^* = -G/(H+1)$ | $\hat{y}^{(1)} = 17.25 + 0.3w$ |
|---|---|---|---|---|
| $x \le 4$ | 1–4 | 54 | −10.8 | 14.01 |
| $4 < x \le 6$ | 5, 6 | −7.5 | +2.5 | 18.00 |
| $x > 6$ | 7, 8 | −46.5 | +15.5 | 21.90 |

#### Step 7: GOSS (Gradient-based One-Side Sampling)
The idea: rows with **large $|g|$** are badly fit and carry the learning signal, so keep all of them. Rows with **small $|g|$** are already well fit, so keep only a random sample and **up-weight** that sample.

**Parameters:** $a = 0.25$ (top fraction kept) and $b = 0.25$ (fraction sampled from the rest).

1. Sort by $|g|$: 23.75 ($x$=8), 22.75 ($x$=7), 14.25 ($x$=1), 14.25 ($x$=3), 13.25 ($x$=2), 12.25 ($x$=4), 4.75 ($x$=6), 2.75 ($x$=5)
2. **Keep the top $a \cdot N = 2$ rows:** $x = 8$ and $x = 7$, with weight 1.
3. **Randomly sample $b \cdot N = 2$ of the other 6 rows.** Suppose the draw picks $x = 3$ and $x = 6$.
4. **Amplify the sampled rows** by $\dfrac{1-a}{b} = \dfrac{0.75}{0.25} = 3$. This restores their share of the total gradient.

The histogram now uses **4 of the 8 rows**. Re-evaluating the splits with weighted sums ($G = \sum w_i g_i$, $H = \sum w_i h_i$), where the parent has $G = -18$, $H = 8$, $S_{parent} = 324/9 = 36$:

| Split | $G_L, H_L$ | $G_R, H_R$ | GOSS gain | Exact gain |
|---|---|---|---|---|
| $x \le 2$ | 0, 0 | −18, 8 | 0 | 180.06 |
| $x \le 4$ | $3(14.25) = 42.75$, 3 | $3(-4.75) - 22.75 - 23.75 = -60.75$, 5 | **517.99** | 583.20 |
| $x \le 6$ | $42.75 - 14.25 = 28.5$, 6 | −46.5, 2 | 400.39 | 514.82 |

GOSS **picks the same split ($x \le 4$) using half the data**. The gains are approximate, and with larger $N$ the approximation gets tighter. Without the ×3 amplification, the small-gradient rows would be under-represented, which would bias the split search toward the badly fit rows.

#### EFB (Exclusive Feature Bundling), in one line
Sparse features that are **never non-zero at the same time** (for example, one-hot columns of the same category) are merged into one feature by **offsetting their value ranges**. If feature A takes values in $[0, 10)$ and B in $[0, 20)$, the bundle is A's value, or B's value + 10. The histogram cost then scales with the number of *bundles*, not the number of features.

---

### 3.3 CatBoost: Hands-on Math

**What to show:** ordered target statistics (how categorical features are handled), ordered boosting, and **oblivious (symmetric) trees**.

#### Part A: Ordered target statistics (categorical encoding)

**The problem with naive target encoding:** replacing a category with the mean target of *all* rows in that category uses each row's own label to build its own feature. That is **target leakage**. In the extreme case, a category that appears only once gets encoded as exactly its own $y$, a perfect predictor in training that is useless at test time.

**CatBoost's fix:** shuffle the rows into a random permutation, and encode each row using **only the rows before it** in that order:

$$
\text{TS}_i = \frac{\sum_{j<i,\; c_j = c_i} y_j \;+\; a \cdot P}{\\#\{j<i : c_j = c_i\} \;+\; a}
$$

where $P$ is a prior (here the global mean) and $a$ is the prior's weight (here 1).

**Data (already in permutation order):** $P = \bar{y} = 299/8 = 37.375$, $a = 1$

| Row | City | $y$ | Earlier same-city $y$'s | $\text{TS}_i$ |
|---|---|---|---|---|
| 1 | A | 20 | none | $(0 + 37.375)/1 = 37.375$ |
| 2 | B | 44 | none | $37.375$ |
| 3 | A | 29 | 20 | $(20 + 37.375)/2 = 28.688$ |
| 4 | B | 55 | 44 | $(44 + 37.375)/2 = 40.688$ |
| 5 | A | 22 | 20, 29 | $(49 + 37.375)/3 = 28.792$ |
| 6 | B | 54 | 44, 55 | $(99 + 37.375)/3 = 45.458$ |
| 7 | A | 30 | 20, 29, 22 | $(71 + 37.375)/4 = 27.094$ |
| 8 | B | 45 | 44, 55, 54 | $(153 + 37.375)/4 = 47.594$ |

Compare this with naive encoding: A = 25.25 and B = 49.5 for every row, each computed using the row's own label.

**What to notice:**
- **No row sees its own label**, so there's no leakage.
- **Early rows are noisy** (rows 1 and 2 both get the prior, so they can't be told apart). CatBoost reduces this by using **several permutations** and averaging.
- **At inference**, the TS is computed from *all* the training data.
- CatBoost also builds **combinations** of categorical features (for example, city × device) greedily as it grows trees.

#### Part B: Ordered boosting (prediction shift)

**The problem:** in standard boosting, the residual for row $i$ comes from a model that was **trained on row $i$ itself**. So training residuals are systematically smaller than residuals on unseen data. This is called *prediction shift*, and it's a form of overfitting.

**The fix:** the residual for row $i$ is computed by a model trained only on the rows **before $i$** in the permutation.

**Toy illustration** (with a "model" that just predicts the mean, using the rows above): for row 3 ($y = 29$):
- **Ordered:** the model sees rows 1–2 only, predicts $(20+44)/2 = 32$, so the residual is $-3$.
- **Leaky:** the model sees rows 1–3, including row 3 itself, predicts $(20+44+29)/3 = 31$, so the residual is $-2$.

The leaky residual is **pulled toward 0** because the model already "knew" $y_3$. Ordered boosting keeps residuals honest, the same way ordered TS keeps the encodings honest.

#### Part C: Oblivious (symmetric) tree

In an **oblivious tree**, every node at the same depth uses **the same split condition**. A depth-$d$ tree is therefore just $d$ yes/no questions, which gives $2^d$ leaves. At each level, the chosen split is the one that maximizes the **total score summed over all resulting leaves**.

**Setup:** MSE, $\lambda = 1$ (CatBoost's default `l2_leaf_reg` is 3; we use 1 for clean arithmetic), $\eta = 0.3$, depth = 2

| $i$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| size | 50 | 55 | 60 | 65 | 85 | 90 | 95 | 100 |
| floor | 1 | 8 | 2 | 7 | 1 | 8 | 2 | 7 |
| $y$ | 20 | 33 | 21 | 32 | 40 | 52 | 41 | 53 |

$\hat{y}^{(0)} = 292/8 = 36.5$, and $g_i = 36.5 - y_i$:

| $i$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| $g_i$ | 16.5 | 3.5 | 15.5 | 4.5 | −3.5 | −15.5 | −4.5 | −16.5 |

**Level 1:** try every candidate, scoring $S_L + S_R$ with $S = G^2/(H+\lambda)$.

| Candidate | Leaves $(G, n)$ | Total score |
|---|---|---|
| size ≤ 75 | (40, 4), (−40, 4) | $1600/5 + 1600/5 = \mathbf{640.0}$ |
| size ≤ 87.5 | (36.5, 5), (−36.5, 3) | 555.1 |
| size ≤ 62.5 | (35.5, 3), (−35.5, 5) | 525.1 |
| floor ≤ 4.5 | (24, 4), (−24, 4) | 230.4 |

The level-1 split is **size ≤ 75**.

**Level 2:** the **same** condition is applied to *both* halves, and the score is summed over all **4 leaves**.

| Candidate | 4 leaves $(G, n)$ | Total score |
|---|---|---|
| **floor ≤ 4.5** | (32, 2), (8, 2), (−8, 2), (−32, 2) | $341.33 + 21.33 + 21.33 + 341.33 = \mathbf{725.33}$ |
| size ≤ 87.5 | (40, 4), (0, 0), (−3.5, 1), (−36.5, 3) | 659.19 |
| size ≤ 62.5 | (35.5, 3), (4.5, 1), (0, 0), (−40, 4) | 645.19 |

The level-2 split is **floor ≤ 4.5**, applied on both sides. The final tree:

| Leaf (size, floor) | Rows | $G$ | $w^* = -G/(n+1)$ | $\hat{y}^{(1)} = 36.5 + 0.3w$ |
|---|---|---|---|---|
| small, low | 1, 3 | 32 | −10.667 | 33.3 |
| small, high | 2, 4 | 8 | −2.667 | 35.7 |
| large, low | 5, 7 | −8 | +2.667 | 37.3 |
| large, high | 6, 8 | −32 | +10.667 | 39.7 |

MSE drops from **136.25 to 87.29**.

**Why use oblivious trees?**
- **Very fast inference:** a leaf's index is just $d$ bits (for example, bit 1 = size > 75, bit 2 = floor > 4.5), computed without any branching. This is ideal for CPU-vectorized, low-latency serving.
- **Built-in regularization:** the same split has to work for every node at that depth, so the trees are less able to chase quirks of small sub-populations.
- **Trade-off:** each individual tree is less flexible, so CatBoost typically needs more trees or greater depth (the default depth is 6).

(CatBoost's default split-scoring function is slightly different, a "Cosine" score, but the level-by-level, same-split-everywhere logic is exactly as shown.)

---

### 3.4 XGBoost vs LightGBM vs CatBoost

| | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Tree growth | Level-wise (depth-wise) | **Leaf-wise** (best-first) | **Oblivious / symmetric** |
| Split search | Exact greedy or histogram | Histogram + subtraction trick | Histogram (quantized borders) |
| Row sampling | Uniform subsample | **GOSS** (gradient-based) | Bayesian / Bernoulli bootstrap |
| Categoricals | One-hot (native support in newer versions) | Native (sorts categories by gradient statistics) | **Ordered target statistics** plus feature combinations |
| Overfitting guard | $\lambda$, $\gamma$, `min_child_weight` | `num_leaves`, `min_data_in_leaf`, $\lambda$ | Ordered boosting, symmetric trees, `l2_leaf_reg` |
| Strength | Mature, well-understood, strong regularization | **Fastest on large data**, low memory | **Best with many categoricals**, good defaults with little tuning |
| Watch out for | Slower on huge data with exact mode | Overfits small datasets | Slower training; the defaults matter |

**One-line interview answer:** "All three are second-order gradient boosting methods. XGBoost added regularization and pruning, LightGBM made it fast with histograms, GOSS, and leaf-wise growth, and CatBoost made it robust for categorical data with ordered statistics and symmetric trees."

---

## 4. PCA via the Covariance Matrix

The main notes derive PCA through SVD. Interviewers often ask, **"Why eigenvectors of the *covariance matrix*, and not anything else?"** Here is the derivation.

1. Mean-center the data $X$ ($n \times d$). The covariance matrix is $\Sigma = \frac{1}{n-1}X^TX$.
2. Projecting the data onto a unit vector $w$ gives the scores $Xw$, whose variance is $\text{Var}(Xw) = w^T\Sigma w$.
3. PCA wants the direction of **maximum variance**: $\max_w \; w^T\Sigma w \quad \text{s.t.} \quad w^Tw = 1$ (without the constraint, you could make the variance infinite by scaling $w$ up).
4. The Lagrangian is $\mathcal{L} = w^T\Sigma w - \lambda(w^Tw - 1)$. Setting its derivative to zero gives $2\Sigma w - 2\lambda w = 0$, so $\Sigma w = \lambda w$. **The optimal directions are eigenvectors of $\Sigma$.**
5. Plugging back in: $w^T\Sigma w = w^T\lambda w = \lambda$. **The variance captured along each direction equals its eigenvalue**, so you sort the eigenvalues in descending order and keep the top $k$.
6. Because $\Sigma$ is symmetric, its eigenvectors are **orthogonal**, so the principal components are uncorrelated.

**Connecting to the SVD route:** if $X = U S V^T$, then $X^TX = V S^2 V^T$. The right singular vectors $V$ are the eigenvectors of $\Sigma$, and $\lambda_i = s_i^2/(n-1)$. Libraries use SVD because it is numerically more stable than forming $X^TX$ explicitly.

**Why not another matrix?** Variance along a direction is a quadratic form in the covariance matrix, and nothing else encodes the spread of the data in every direction. (If features are on different scales, use the **correlation matrix**, which is the same as standardizing first. Otherwise, large-unit features dominate.)

**Rank, eigenvectors, and SVD in one breath:**
- **Rank** is the number of linearly independent rows or columns, which is the dimension of the space the data actually spans.
- An **eigenvector** $v$ of $A$ is only scaled by it: $Av = \lambda v$.
- **SVD**, $A = U S V^T$, works for *any* matrix (even non-square). It rotates ($V^T$), scales ($S$), and rotates again ($U$). The number of non-zero singular values equals the rank.

---

## 5. SVM: Where the Kernel Trick Is Used

**Question:** "If the kernel trick computes a dot product, where is that used in the SVM?"

**Answer:** in the **dual formulation**. The data appears *only* through dot products $x_i^Tx_j$.

**Dual problem:**

$$
\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_i\sum_j \alpha_i\alpha_j y_i y_j \,\underbrace{x_i^Tx_j}_{\text{replace with } K(x_i, x_j)} \quad \text{s.t.} \; 0 \le \alpha_i \le C,\; \sum_i \alpha_i y_i = 0
$$

**Prediction:**

$$
f(x) = \text{sign}\Big(\sum_{i \in SV}\alpha_i y_i \, K(x_i, x) + b\Big)
$$

- The trick: replace $x_i^Tx_j$ with $K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$. You get the dot product in a high-dimensional (for RBF, infinite-dimensional) feature space **without ever computing $\phi(x)$**.
- **Example:** $K(x,z) = (x^Tz)^2$ in 2D equals $\phi(x)^T\phi(z)$ with $\phi(x) = (x_1^2, \sqrt{2}x_1x_2, x_2^2)$.
- **Support vectors** are the training points with $\alpha_i > 0$. These are the points on or inside the margin. Every other point has $\alpha_i = 0$ and **has no effect on the model**. Remove all non-support vectors and retrain, and you get exactly the same boundary.

---

## 6. Practical ML

### 6.1 Imbalanced Data

**Data level:**
- **Random oversampling** duplicates minority rows, which risks overfitting to them.
- **Random undersampling** drops majority rows, which loses information.
- **SMOTE** creates synthetic minority points by picking a minority point $x$, one of its $k$ nearest minority-class neighbours $x_{nn}$, and interpolating: $x_{new} = x + u(x_{nn} - x)$ with $u \sim U(0,1)$.
- ⚠️ **Resample only the training fold**, after the split and inside cross-validation. Resampling before splitting leaks synthetic copies into the validation set.

**Algorithm level:**
- **Class weights:** `class_weight='balanced'` weights each class by $\frac{n}{k \cdot n_c}$. In XGBoost and LightGBM, `scale_pos_weight` ≈ #negatives / #positives.
- **Focal loss:** $FL = -(1-p_t)^\gamma \log p_t$. The factor $(1-p_t)^\gamma$ down-weights easy, well-classified examples, so training focuses on hard ones.
- **Boosting** naturally concentrates on misclassified rows, especially combined with class weights.

**Evaluation level:**
- Don't use accuracy. Use **PR-AUC**, F1, or recall at a fixed precision.
- **Tune the decision threshold** on validation data instead of using 0.5.
- Use **stratified** splits and stratified K-fold cross-validation.

### 6.2 Missing Data

**Types of missingness:**

| Type | Meaning | Example | Implication |
|---|---|---|---|
| **MCAR** (completely at random) | Missingness is unrelated to anything | Sensor glitch | Dropping rows is unbiased, just less data |
| **MAR** (at random) | Depends on *other observed* columns | Older people skip the "social media" field | Impute using the other columns (KNN, MICE) |
| **MNAR** (not at random) | Depends on the *missing value itself* | High earners hide their income | Imputation is biased. Add a **missing-indicator** column and use domain knowledge. |

**Techniques when you can't drop rows or columns:**
1. Mean, median (robust to skew and outliers), or mode (for categoricals). Better still, compute them **within groups**, e.g. median income by city.
2. **Regression or iterative imputer (MICE):** predict the missing column from the other columns, and iterate.
3. **KNN imputer:** use the average of the $k$ most similar rows.
4. **Missing-indicator column** (`is_missing`): lets the model learn that "missing" itself carries information. This is essential for MNAR.
5. **Models that handle missing values natively:** XGBoost and LightGBM learn a default branch direction, and CART can use surrogate splits.
6. Time series: forward fill or backward fill, or interpolation.

⚠️ **Fit every imputer on the training split only**, then apply it to validation and test data. Otherwise, statistics from the test data leak into training.

### 6.3 Cross-Validation & Data Leakage

- **K-fold:** split the data into $k$ parts, train on $k-1$ and validate on the remaining one, rotate, and average the scores. Use **stratified** K-fold for classification.
- **Time series:** use a forward-chaining split (`TimeSeriesSplit`), never a random shuffle. The future must never train on the past's test period.
- **Group K-fold:** keep all rows from the same user or patient in the same fold.
- **Leakage checklist:** Is scaling, imputation, encoding, or resampling fit on the full data? Are there features that are only known *after* the event (e.g. "refund issued" when predicting fraud)? Are there duplicate rows across the split? Is target encoding done without out-of-fold values? **Fix:** put every preprocessing step inside an sklearn `Pipeline`, so it is re-fit inside each fold.

### 6.4 Bias–Variance Trade-off

$$
E[(y-\hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2_{\text{irreducible}}
$$

| | High bias (underfitting) | High variance (overfitting) |
|---|---|---|
| Symptom | Train error high, validation error high | Train error low, validation error much higher |
| Fix | More features, more complex model, less regularization | More data, regularization, simpler model, bagging, dropout |

- **Bagging** reduces **variance**: it averages many low-bias, high-variance trees.
- **Boosting** reduces **bias**: it adds weak, high-bias learners one after another.

### 6.5 Why L1 Gives Sparsity (and L2 Doesn't)

- **Geometric view:** the L1 constraint region $|w_1| + |w_2| \le t$ is a **diamond with corners on the axes**. The elliptical loss contours usually touch it first at a corner, where some $w_j = 0$. The L2 region is a **circle** with no corners, so the touching point is almost never exactly on an axis.
- **Gradient view:** the L1 penalty's gradient is $\lambda \cdot \text{sign}(w)$, a **constant push** toward 0 no matter how small $w$ gets, so weights reach exactly 0. The L2 gradient is $2\lambda w$, which **shrinks proportionally**: the push weakens as $w \to 0$, so weights get small but never exactly 0.

---

## 7. Deep Learning Gaps

### 7.1 Batch Normalization: The Full Story

**Training (per feature, over a mini-batch of size $m$):**

$$
\mu_B = \frac{1}{m}\sum x_i, \quad \sigma_B^2 = \frac{1}{m}\sum(x_i-\mu_B)^2, \quad \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma\hat{x}_i + \beta
$$

- **Why normalize?** The original paper's reason is **internal covariate shift**: as earlier layers update, the distribution of each layer's inputs keeps changing, so every layer is chasing a moving target. Later research (Santurkar et al., 2018) showed that the bigger benefit is a **smoother loss landscape**, which allows higher learning rates and makes training less sensitive to initialization. Mention both.
- **Why the learnable $\gamma$ (scale) and $\beta$ (shift)?** Forcing every layer's input to have mean 0 and variance 1 would limit what the network can represent. For example, a sigmoid would be stuck in its near-linear region. $\gamma$ and $\beta$ let the network choose the best scale and offset, and if $\gamma = \sqrt{\sigma_B^2+\epsilon}$ and $\beta = \mu_B$, the network can **undo the normalization entirely** if that is optimal.
- **Why apply it before the activation?** The original paper normalizes the pre-activation $Wx + b$, so that the input to the non-linearity is well-scaled and doesn't **saturate** (sigmoid or tanh stuck in their flat regions, or ReLUs that never fire). It also makes the bias $b$ redundant, since $\beta$ plays that role. In practice, placing it after the activation also works, so you can say "the paper placed it before, and it's debated."

**Inference (the "single example" question):**
- With a batch of 1, $\sigma_B^2 = 0$ and $\hat{x}$ is meaningless, and predictions shouldn't depend on which other examples happen to share the batch.
- So during training, BatchNorm keeps **running averages**: $\mu_{run} \leftarrow m\,\mu_{run} + (1-m)\,\mu_B$ (similarly for $\sigma^2$), with momentum $m \approx 0.9$.
- **At inference**, those fixed running statistics are used: $y = \gamma\frac{x - \mu_{run}}{\sqrt{\sigma^2_{run}+\epsilon}} + \beta$. This is a deterministic, fixed transformation.

**Why an *exponentially weighted* average?**
- It is cheap: $O(1)$ memory, with no need to store the statistics of every batch.
- It **favors recent batches**. The network's weights change during training, so the activation statistics from early batches are stale. The EWA gives weight $(1-m)m^k$ to the batch from $k$ steps ago, so old batches fade out exponentially.

### 7.2 Exponentially Weighted Averages: The Intuition

$$
v_t = \beta v_{t-1} + (1-\beta)\,x_t
$$

- Unrolling gives $v_t = (1-\beta)\big[x_t + \beta x_{t-1} + \beta^2 x_{t-2} + \dots\big]$, so the weights decay geometrically.
- It **averages over roughly the last $\frac{1}{1-\beta}$ values**: $\beta = 0.9$ gives about 10 steps, and $\beta = 0.99$ about 100. (This is because $\beta^{1/(1-\beta)} \approx 1/e$.)
- The same idea is used in **momentum**, **RMSProp**, **Adam**, and **BatchNorm's running statistics**.

### 7.3 Adam and Its Bias Correction

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \qquad \theta_t = \theta_{t-1} - \alpha\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

**The techniques Adam combines:**
- Momentum: $m_t$, the first moment, which smooths the direction of the gradient.
- RMSProp: $v_t$, the second moment, which gives each parameter its own step size.
- Bias correction.

**Why bias correction?** $m_0 = v_0 = 0$, so early estimates are biased toward 0. **Example:** $\beta_1 = 0.9$, $g_1 = 1$. Then $m_1 = 0.1$, even though the true gradient is 1. The correction gives $\hat{m}_1 = 0.1/(1 - 0.9) = 1$. As $t$ grows, $\beta^t \to 0$ and the correction fades away.

**"Why do we need optimizers in DL but not in classical ML?"** Many classical models have **closed-form solutions** (OLS: $w = (X^TX)^{-1}X^Ty$) or **convex** losses where plain gradient descent works reliably. Deep networks have **non-convex** losses with many local minima, saddle points, ravines, and flat plateaus, plus millions of parameters whose gradients differ in scale. Adaptive optimizers with momentum handle all of that.

### 7.4 Batch GD vs SGD vs Mini-batch

| | Batch GD | SGD | Mini-batch |
|---|---|---|---|
| Gradient computed from | All $n$ rows | 1 random row | $b$ rows (e.g. 32–512) |
| Cost per update | $O(n)$ | $O(1)$ | $O(b)$ |
| Path | Smooth | Very noisy | Moderately noisy |

**Why "stochastic"?** Each update uses a **randomly sampled** row or batch. The gradient is therefore a *random variable*: an **unbiased but noisy estimate** of the true full-data gradient ($E[\nabla L_i] = \nabla L$). That noise can even help the optimizer escape saddle points and sharp minima.

---

## 8. NLP Gaps

### 8.1 Word2Vec

- **Goal:** learn dense word vectors in which words that appear in similar contexts end up close together (the *distributional hypothesis*).
- **CBOW:** predict the **center word** from the average of its context words. Faster, and good for frequent words.
- **Skip-gram:** predict the **context words** from the center word. Better for rare words.
- **Architecture:** a shallow network with a single linear hidden layer. **The learned hidden-layer weight matrix *is* the embedding table.**
- **Efficiency:** a full softmax over the whole vocabulary is expensive, so training uses **negative sampling** (a binary "real context pair or random pair?" task with a few negatives) or hierarchical softmax.
- **The famous property:** $\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$.
- **Limitation:** each word gets **one static vector** ("bank" means the same thing next to "river" as next to "money"). Contextual models like ELMo and BERT fix this.

### 8.2 Why Sinusoidal Positional Encoding?

$$
PE_{(pos, 2i)} = \sin\!\big(pos / 10000^{2i/d}\big), \quad PE_{(pos, 2i+1)} = \cos\!\big(pos / 10000^{2i/d}\big)
$$

1. **Relative positions are linear:** for any offset $k$, $PE_{pos+k}$ is a **rotation (a linear map) of $PE_{pos}$**, via $\sin(a+b) = \sin a\cos b + \cos a\sin b$. So attention can easily learn "look $k$ tokens back."
2. **Bounded values** in $[-1, 1]$, unlike raw integer positions, which grow without limit.
3. **Unique** for each position: the different wavelengths work like the hands of a clock, from fast to slow.
4. **No learned parameters**, and it can in principle extrapolate to sequences longer than those seen in training.

**If asked "why *only* sin/cos?", push back politely:** sin/cos is not the only choice. BERT and GPT-2 use **learned** absolute embeddings, and modern LLMs use **RoPE** (covered in your transformer notes) or ALiBi. Sin/cos was the original paper's parameter-free choice with the nice properties above.

### 8.3 Causal Masking: How It's Implemented

1. Compute the scores $QK^T/\sqrt{d_k}$, an $n \times n$ matrix.
2. Add a mask $M$ with $M_{ij} = 0$ if $j \le i$ and $M_{ij} = -\infty$ if $j > i$. That is **$-\infty$ on the upper triangle**.
3. Apply the softmax: $e^{-\infty} = 0$, so each token gives **zero attention to future tokens**.

In code: `scores.masked_fill(torch.triu(torch.ones(n,n), diagonal=1).bool(), float('-inf'))`

It uses $-\infty$ *before* the softmax rather than zeroing weights *after* it, so that each row still sums to 1.

### 8.4 Why Cross-Attention Only in Encoder–Decoder Models?

In cross-attention, **Q comes from the decoder** and **K and V come from the encoder's output**. A decoder-only model (GPT) has **no separate encoder output** to attend to. Its "source" (the prompt) is just the earlier tokens of the same sequence, which ordinary causal self-attention already covers. Cross-attention exists to connect **two different sequences**: source and target in translation, or image and text in vision–language models.

### 8.5 BERT Quick Facts

- An **encoder-only**, bidirectional Transformer.
- **Pretraining objectives:**
  1. **Masked Language Modeling (MLM):** 15% of tokens are selected. Of those, 80% are replaced with `[MASK]`, 10% with a random token, and 10% are left unchanged. The model predicts the original token.
  2. **Next Sentence Prediction (NSP):** is sentence B the actual next sentence after A? (Later work, such as RoBERTa, dropped NSP.)
- **Learning paradigm:** **self-supervised** learning, where the labels are created from the raw text itself.
- **The `[CLS]` token:** prepended to every input. Through self-attention it can gather information from the whole sequence, so its final hidden state is used as the **sequence-level representation** for classification (a linear head is placed on top of `[CLS]`). It was trained for this role by NSP.

---

## 9. Generative vs Discriminative & Naive Bayes

| | Generative | Discriminative |
|---|---|---|
| Learns | $P(x, y)$, i.e. $P(x \mid y)P(y)$ | $P(y \mid x)$ directly (or just a decision boundary) |
| Examples | Naive Bayes, GMM, HMM, LDA, VAEs, GANs | Logistic regression, SVM, trees, standard neural networks |
| Can generate new $x$? | Yes | No |

**Naive Bayes:**

$$
P(y \mid x) \propto P(y)\prod_{j} P(x_j \mid y)
$$

- The "naive" part is the assumption that **features are conditionally independent given the class**.
- **Variants:** Gaussian (continuous features), Multinomial (word counts), Bernoulli (binary features).
- **Laplace smoothing:** add 1 to every count, so that an unseen word doesn't produce $P = 0$ and wipe out the whole product.
- **Strengths:** very fast, works well on text, and needs little data.

**On "first generative algorithms":** some interviewers name Naive Bayes, which is fair, since it is a classic generative classifier. If someone names **t-SNE** as the second, be careful: t-SNE is a **visualization and dimensionality-reduction** method. It does not model $P(x)$ and cannot generate samples. Safer examples of classical generative models are **GMMs, HMMs, and LDA (Latent Dirichlet Allocation)**. Answer with the definition (a model of $P(x, y)$ or $P(x)$) and back it up with these examples.

---

## 10. Statistics Essentials

### 10.1 Random Variables & Distributions

A **random variable** is a function that maps each outcome of an experiment to a number. Discrete random variables have a PMF, continuous ones have a PDF, and both have a CDF, $F(x) = P(X \le x)$.

| Distribution | PMF / PDF | Mean | Variance |
|---|---|---|---|
| Bernoulli($p$) | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ |
| Binomial($n,p$) | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ |
| Geometric($p$), trials until first success | $(1-p)^{k-1}p$ | $1/p$ | $(1-p)/p^2$ |
| Poisson($\lambda$) | $\lambda^k e^{-\lambda}/k!$ | $\lambda$ | $\lambda$ |
| Uniform($a,b$) | $1/(b-a)$ | $(a+b)/2$ | $(b-a)^2/12$ |
| Exponential($\lambda$) | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ |
| Normal($\mu,\sigma^2$) | $\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ |

**Derivations to be able to write:**
- **Bernoulli:** $E[X] = 0(1-p) + 1(p) = p$. Since $E[X^2] = p$, $\text{Var} = p - p^2 = p(1-p)$.
- **Binomial:** $X = \sum_{i=1}^n B_i$ is a sum of independent Bernoullis. So $E[X] = np$ by linearity of expectation, and $\text{Var} = np(1-p)$ because the variances of independent variables add. **Example:** 10 coin tosses have mean 5 heads and variance 2.5.
- **Geometric:** $E[X] = \sum_{k\ge1} k(1-p)^{k-1}p = p \cdot \frac{1}{p^2} = \frac{1}{p}$, using $\sum k q^{k-1} = \frac{1}{(1-q)^2}$. **Example:** a fair die needs 6 rolls on average to show the first six.
- **Bernoulli vs Binomial:** Bernoulli is one trial. Binomial counts the successes in $n$ independent Bernoulli trials, so Bernoulli = Binomial($1, p$).

### 10.2 CLT vs LLN

- **CLT:** for iid samples with finite variance, $\bar{X} \approx N(\mu, \sigma^2/n)$ for large $n$ (rule of thumb $n \ge 30$), **whatever the shape of the original distribution**. This is what justifies z-tests, t-tests, and confidence intervals on means.
- **LLN:** $\bar{X} \to \mu$. It says the average converges, but nothing about the *shape* of its distribution.

### 10.3 Why the Normal Distribution Is Assumed So Often

1. **CLT:** sums of many small independent effects (such as measurement noise) are approximately normal.
2. **Maximum entropy:** among all distributions with a given mean and variance, the normal distribution assumes the least additional structure.
3. **Maths:** assuming Gaussian errors makes **MLE equivalent to minimizing MSE**, and the normal distribution is closed under linear combinations and has closed-form results.

### 10.4 Hypothesis Testing: A Worked Example

**Claim:** "The average delivery time is 30 minutes." A sample of $n = 36$ has $\bar{x} = 32$, with known $\sigma = 6$.
1. $H_0: \mu = 30$ vs $H_1: \mu \ne 30$, with $\alpha = 0.05$.
2. $SE = \sigma/\sqrt{n} = 6/6 = 1$, so $z = (32 - 30)/1 = 2$.
3. The two-sided p-value is $2(1 - \Phi(2)) = 2(1 - 0.9772) = 0.0455$.
4. $0.0455 < 0.05$, so **reject $H_0$**.

**p-value:** the probability, *assuming $H_0$ is true*, of seeing data at least as extreme as what was observed. It is **not** $P(H_0 \text{ is true})$.

| | $H_0$ true | $H_0$ false |
|---|---|---|
| Reject $H_0$ | **Type I error** (false positive, probability $\alpha$) | Correct (**power** $= 1-\beta$) |
| Fail to reject | Correct | **Type II error** (false negative, probability $\beta$) |

Power increases with larger $n$, a larger effect size, and a higher $\alpha$.

**Confidence interval:** "95%" means the *procedure* captures the true parameter in 95% of repeated samples. It does not mean there's a 95% chance that this particular interval contains it.

### 10.5 Which Test?

| Situation | Test |
|---|---|
| Mean, with $\sigma$ known or large $n$ | z-test |
| Mean, with $\sigma$ unknown and small $n$ | t-test |
| Same subjects measured before and after | Paired t-test |
| 3 or more group means | ANOVA (F-test) |
| Two **categorical** variables | **Chi-square test of independence** |
| Observed frequencies vs expected | Chi-square goodness-of-fit |
| Non-normal data, comparing 2 groups | Mann–Whitney U |

### 10.6 Chi-square Test: A Worked Example with Intuition

**Question:** is gender related to product preference?

| | Product A | Product B | Row total |
|---|---|---|---|
| Male | 30 | 20 | 50 |
| Female | 20 | 30 | 50 |
| Column total | 50 | 50 | $N = 100$ |

1. **Expected counts under independence:** $E_{ij} = \dfrac{\text{row}_i \times \text{col}_j}{N}$. **Why this formula?** If the variables are independent, $P(\text{male} \cap A) = P(\text{male})P(A) = \frac{50}{100}\cdot\frac{50}{100}$, and multiplying by $N$ gives $E = 25$ for every cell.
2. **The statistic:** $\chi^2 = \sum \dfrac{(O-E)^2}{E}$.
   - **Why square?** So that positive and negative deviations don't cancel each other out.
   - **Why divide by $E$?** A gap of 5 matters more when you expected 10 than when you expected 1,000. (Also, counts are roughly Poisson with variance ≈ $E$, so each term is like a squared z-score.)
   - Here: $4 \times \frac{(30-25)^2}{25} = 4 \times 1 = 4.0$
3. **Degrees of freedom:** $(r-1)(c-1) = 1$. Once the margins are fixed, only one cell is free to vary.
4. The critical value at $\alpha = 0.05$ with df = 1 is **3.841**, and $4.0 > 3.841$, so **reject independence** ($p \approx 0.0455$). Preference does depend on gender.

### 10.7 p-hacking

- **Definition:** trying many analyses (different metrics, subgroups, stopping times, or outlier rules) and reporting only the ones that reach $p < 0.05$.
- **Why it's a problem:** with 20 independent tests at $\alpha = 0.05$ and no real effects anywhere, you should still **expect 1 false positive**.
- **Fixes:**
  - Pre-register the hypothesis and the metric before looking at data.
  - **Bonferroni correction:** test each at $\alpha/m$.
  - FDR control (Benjamini–Hochberg).
  - Hold out a confirmation dataset.

### 10.8 A/B Testing Essentials

1. Fix **one primary metric** and the hypothesis *before* the test starts.
2. Compute the sample size from $\alpha$, power (typically 0.8), and the **minimum detectable effect**.
3. **Randomize at the right unit** (user, not session) and check the traffic split is actually as intended.
4. **Don't peek** at results and stop early; this inflates the Type I error. Run for full weekly cycles.
5. Watch for novelty effects, network effects, and multiple-comparison problems.

---

## 11. Probability Question Bank

**The method:**
1. Define the events in words.
2. Identify what's being asked.
3. Pick the tool:
   - "given that" → conditional probability
   - "which source did it come from" → Bayes (draw a tree)
   - "at least one" → $1 - P(\text{none})$
   - "expected number" → linearity of expectation, or states for sequence problems
4. Sanity-check the answer.

All the answers below were verified by exact computation.

### Counting & Classical
1. **Two dice: $P(\text{sum} = 8)$?** The outcomes are (2,6), (3,5), (4,4), (5,3), (6,2). → **5/36**
2. **3 coins: $P(\ge 2 \text{ heads})$?** → 4/8 = **1/2**
3. **A committee of 3 from 5 men and 4 women: $P(\text{exactly 2 women})$?** → $\binom{4}{2}\binom{5}{1}/\binom{9}{3} = 30/84$ = **5/14**
4. **$P(\text{exactly 2 aces in a 5-card hand})$?** → $\binom{4}{2}\binom{48}{3}/\binom{52}{5} \approx$ **0.0399**
5. **Arrangements of MISSISSIPPI?** → $\frac{11!}{4!\,4!\,2!}$ = **34,650**
6. **Two cards drawn without replacement: $P(\text{same colour})$?** → **25/51**

### Conditional Probability
7. **Two dice, the sum is 8: $P(\text{at least one 6})$?** → Of the 5 outcomes, 2 contain a 6. → **2/5**
8. **Two children, at least one is a boy: $P(\text{both boys})$?** → The equally likely cases are BB, BG, GB. → **1/3** (If you're told the *older* one is a boy: 1/2.)
9. **A box with 3 red and 2 blue balls, 2 drawn without replacement.** $P(\text{both red}) = \frac{3}{5}\cdot\frac{2}{4}$ = **3/10**. $P(\text{2nd red} \mid \text{1st red})$ = **1/2**. $P(\text{2nd red})$ = **3/5** (with no information about the first draw, the second behaves just like the first).
10. **The first die shows 6: $P(\text{sum} \ge 10)$?** → The second die must be 4, 5, or 6. → **1/2**

### Bayes (the most tested)
11. **Bag A has 3R 2B, bag B has 1R 4B. A random bag is chosen and a red ball drawn. $P(\text{bag A} \mid \text{red})$?** → $\frac{0.5(0.6)}{0.5(0.6)+0.5(0.2)}$ = **3/4**
12. **Machines make 50%, 30%, and 20% of output, with defect rates 2%, 3%, and 5%.** $P(\text{defective})$ = **0.029**. $P(\text{made by the 3rd machine} \mid \text{defective}) = 0.010/0.029$ = **10/29**
13. **A disease with 1% prevalence; the test has 99% sensitivity and a 5% false-positive rate. $P(\text{sick} \mid +)$?** → $\frac{0.0099}{0.0099 + 0.0495}$ = **1/6** (false positives swamp the true positives because the disease is rare)
14. **A fair coin and a two-headed coin. Pick one at random; it shows heads. $P(\text{two-headed})$?** → **2/3**. After 3 heads in a row → **8/9**
15. **A tells the truth 3/4 of the time and says a die showed six. $P(\text{it was six})$?** → $\frac{\frac{1}{6}\cdot\frac{3}{4}}{\frac{1}{6}\cdot\frac{3}{4} + \frac{5}{6}\cdot\frac{1}{4}}$ = **3/8**

### Independence & Complement
16. **$P(\text{at least 1 head in 5 tosses})$?** → $1 - (1/2)^5$ = **31/32**
17. **Three shooters hit with probabilities 1/2, 1/3, and 1/4. $P(\text{target hit})$?** → $1 - \frac{1}{2}\cdot\frac{2}{3}\cdot\frac{3}{4}$ = **3/4**
18. **$P(\text{sum} = 7 \text{ or } 11)$ with two dice?** → 8/36 = **2/9**
19. **Birthday problem: $P(\text{all } n \text{ birthdays different})$?** → $\prod_{k=0}^{n-1}\frac{365-k}{365}$. For $n = 23$ this is ≈ 0.493, so $P(\text{shared}) \approx$ **0.507**

### Expectation
20. **$E[\max \text{ of two dice}]$?** → $P(\max = k) = \frac{2k-1}{36}$, so $E = \frac{161}{36} \approx$ **4.47**
21. **Roll a die, win its value, with one optional reroll. Best expected value?** → Reroll on 1–3. $\frac{1}{2}(3.5) + \frac{4+5+6}{6}$ = **4.25**
22. **Expected tosses until the first H? Until HH? Until HT?** → **2, 6, 4.** For HH, set up states: $E = 1 + \frac{1}{2}E_1 + \frac{1}{2}E$ and $E_1 = 1 + \frac{1}{2}E$, so $E = 6$.
23. **$n$ hats returned at random. Expected number of correct matches?** → Each person matches with probability $\frac{1}{n}$, so the expectation is **1** for any $n$.
24. **Expected rolls to see all 6 faces?** → $6(1 + \frac{1}{2} + \dots + \frac{1}{6})$ = **14.7**

### Distributions
25. **Guess randomly on 10 MCQs with 4 options each: $P(\text{exactly 3 right})$?** → $\binom{10}{3}(0.25)^3(0.75)^7 \approx$ **0.250**
26. **Poisson with 3 calls/hour: $P(0 \text{ calls})$ and $P(\ge 2 \text{ calls})$?** → $e^{-3} \approx$ **0.0498**, and $1 - 4e^{-3} \approx$ **0.801**
27. **$P(\text{first six on the 3rd roll})$?** → $(5/6)^2(1/6)$ = **25/216**
28. **Heights are $N(170, 10^2)$: $P(> 190)$?** → $P(Z > 2) \approx$ **0.0228**

### Classics & Traps
29. **Two friends each arrive uniformly between 12:00 and 1:00 and wait 15 minutes. $P(\text{meet})$?** → $1 - (3/4)^2$ = **7/16**
30. **Break a stick at 2 random points: $P(\text{the pieces form a triangle})$?** → **1/4**
31. **Monty Hall: should you switch?** → Yes; switching wins with probability **2/3**
32. **Two children, one is a boy born on a Tuesday: $P(\text{both boys})$?** → **13/27**

---

## 12. Final Interview Checklist

You should be able to explain each of these **out loud in about 2 minutes, without notes**:

- [ ] VIF procedure, and why it depends on $R^2$
- [ ] Multicollinearity vs autocorrelation, and the Durbin–Watson test
- [ ] $R^2$'s flaw, and the adjusted $R^2$ formula
- [ ] Why "logistic regression", and why sigmoid instead of tanh
- [ ] Gini vs entropy, with a worked number
- [ ] Bootstrap sampling (63.2% unique) and per-split feature sampling in random forests
- [ ] XGBoost similarity score, gain, and leaf weight, and the roles of $\lambda$ and $\gamma$
- [ ] LightGBM: histogram binning, subtraction trick, leaf-wise growth, GOSS, EFB
- [ ] CatBoost: ordered target statistics, ordered boosting, oblivious trees
- [ ] PCA derivation ($\Sigma w = \lambda w$)
- [ ] Where the kernel trick appears in the SVM dual, and what support vectors are
- [ ] Imbalanced data (with the resampling-leakage warning) and missing data (MCAR/MAR/MNAR)
- [ ] BatchNorm: $\gamma, \beta$, running statistics at inference, and why EWA
- [ ] Inverted dropout, and the link between dropout and bagging
- [ ] Adam bias correction, with the numeric example
- [ ] Word2Vec, sinusoidal positional encoding, the causal mask, cross-attention, BERT and `[CLS]`
- [ ] Hypothesis test and chi-square worked examples, p-value, p-hacking
- [ ] At least 15 of the 32 probability questions solved on paper