# Machine Learning Placement Master Notes

**Table of Contents:**
1. [Part 1: Linear Regression & Regularization](#part-1-linear-regression--regularization)
2. [Part 1.5: Data Transformations & Feature Scaling](#part-15-data-transformations--feature-scaling)
3. [Part 2: Logistic Regression (Classification Baseline)](#part-2-logistic-regression-classification-baseline)
4. [Part 3: Classification Evaluation Metrics](#part-3-classification-evaluation-metrics)
5. [Part 4: Support Vector Machines (SVM)](#part-4-support-vector-machines-svm)

---

## Part 1: Linear Regression & Regularization

### 1. Core Concept of Linear Regression
*   **Goal:** Predict a continuous target variable ($y$) based on one or more input features ($X$) by fitting a "best-fit" straight line.
*   **Equation:** $y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \epsilon$
    *   $\beta_0$: Intercept (where the line crosses the y-axis)
    *   $\beta_i$: Coefficients (weights). *Interpretation: A 1-unit increase in $X_i$ changes $y$ by $\beta_i$, assuming all else is held constant.*
![Linear Regression Best Fit and Residuals](./assets/linear_fit_residuals.png)
*   **Algorithm (Optimization):**
    *   **Ordinary Least Squares (OLS):** Finds the exact best line mathematically by minimizing the sum of squared residuals. Good for small datasets.
    *   **Gradient Descent:** Iteratively updates coefficients ($\beta$) to minimize the loss using a learning rate. Crucial for large datasets.
*   **Loss Function (MSE Cost Function):** $J(\beta) = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$. 
    *   *Why do we square the errors?* 1) It heavily penalizes larger errors. 2) It mathematically ensures the cost function is differentiable and strictly convex (a perfect bowl shape), guaranteeing Gradient Descent finds the single global minimum.
![Gradient Descent Convex Bowl](./assets/gradient_descent_bowl.png)

### 2. How We Train Linear Regression (Optimization Methods)
To find the best weights ($\beta$), we have two primary methods: The Normal Equation and Gradient Descent.

**A. The Normal Equation (The Analytical Way)**
*   **Formula:** $\beta = (X^T X)^{-1} X^T y$
*   **Pros:** No need to choose a learning rate ($\alpha$), no iterations, gives the exact mathematical global minimum.
*   **Cons:** Matrix inversion $(X^T X)^{-1}$ is computationally expensive (roughly $O(n^3)$).

**B. Gradient Descent (The Iterative Way)**
*   **Batch Gradient Descent:** Uses the **entire training dataset** for one step. Slow on large data, smooth convergence.
*   **Stochastic Gradient Descent (SGD):** Uses **exactly one random data point** per step. Extremely fast, jumps out of local minima, noisy convergence.
*   **Mini-Batch Gradient Descent:** Uses a **small batch** (e.g., 32, 64) per step. Industry standard. Fast, stable, hardware-optimized.

**C. Hand-Worked Example: Gradient Descent in Action**
*   **Dataset:** $X = [1, 2, 3]$, $y = [2, 4, 6]$ *(True rule: $y = 2X$)*
*   **Step 1 (Initialize):** Guess $\beta_1 = 1$, $\beta_0 = 0$. Model: $\hat{y} = 1X$.
*   **Step 2 (Predict):** Predictions $\hat{y} = [1, 2, 3]$.
*   **Step 3 (Calculate Error):** Errors $(y - \hat{y}) = [1, 2, 3]$. MSE Loss = $(1^2 + 2^2 + 3^2) / 3 = 14 / 3 = 4.67$.
*   **Step 4 (Find the Gradient):** Gradient $= \frac{-2}{n} \sum X_i (y_i - \hat{y}_i)$. 
    *   $= \frac{-2}{3} \times [ (1\times1) + (2\times2) + (3\times3) ] = -9.33$
*   **Step 5 (Update Weight):** Learning rate $\alpha = 0.05$.
    *   $\text{New } \beta_1 = 1 - (0.05 \times -9.33) = 1 + 0.466 = 1.466$ *(Closer to 2.0!)*

### 2.6 Mathematical Derivation of the Gradient (Optional but highly recommended)
Walking through the calculus step-by-step builds an incredibly strong foundation for understanding how Gradient Descent actually learns.

**The Setup:**
* **Hypothesis (Prediction):** $h_\theta(x) = \theta^T x$
* **Cost Function (MSE):**
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

*Explanation of the Cost Function:* We use $m$ to represent the total number of training examples. We add a $\frac{1}{2}$ to the formula purely as a mathematical convenience—when we take the derivative, the exponent $2$ will drop down and cancel out the $\frac{1}{2}$, making the final math cleaner without changing where the global minimum is located.

**Step 1: Set up the Partial Derivative**
Our goal is to find how a tiny change in a single specific weight, $\theta_j$, impacts the overall error. We do this by taking the partial derivative of $J(\theta)$ with respect to $\theta_j$.
$$\frac{\partial}{\partial \theta_j} J(\theta) = \frac{\partial}{\partial \theta_j} \left[ \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 \right]$$

**Step 2: Apply the Power Rule and Chain Rule**
In calculus, the Power Rule tells us to bring the exponent $2$ down to the front. The Chain Rule tells us we then have to multiply the whole thing by the derivative of whatever was *inside* the parentheses.
$$\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot \frac{\partial}{\partial \theta_j} (h_\theta(x^{(i)}) - y^{(i)})$$
*Explanation:* Notice how the $\frac{1}{2}$ and the $2$ canceled each other out, leaving just $\frac{1}{m}$. Now, we just need to solve that lingering derivative on the far right.

**Step 3: Differentiate the Inside**
Remember that our hypothesis $h_\theta(x)$ is just a sum of all weights multiplied by their features: $\theta_0x_0 + \theta_1x_1 + \dots + \theta_jx_j$. Because we are taking a *partial* derivative with respect to just $\theta_j$, every other weight acts like a constant (a flat number) and turns to $0$. The true label $y$ is also a constant, so it turns to $0$. The derivative of $\theta_jx_j$ with respect to $\theta_j$ is simply the feature $x_j$.
$$\frac{\partial}{\partial \theta_j} (h_\theta(x^{(i)}) - y^{(i)}) = x_j^{(i)}$$

**Step 4: The Final Gradient**
We substitute $x_j^{(i)}$ back into our equation from Step 2 to get the final gradient.
$$\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$$

### 3. The 5 Core Assumptions (L.I.N.E. + M)
1.  **Linearity:** Relationship between $X$ and $y$ must be linear. *(Fix: Polynomial features or log transform).*
2.  **Independence:** Observations must be independent (no autocorrelation).
3.  **Normality of Residuals:** Errors ($\epsilon$) should be normally distributed.
4.  **Equal Variance (Homoscedasticity):** Residuals should have a constant variance, not fanning out like a cone. *(Fix: Log transform target $y$).*
    ![Heteroscedasticity Cone](./assets/heteroscedasticity.png)
5.  **No Multicollinearity:** Features should not be highly correlated. *(Fix: Drop redundant features, PCA, or L1/L2 Regularization).*

### 4. Regularization: Ridge (L2) vs. Lasso (L1)
*Used when the model overfits, or when you have multicollinearity.*
*   **Bias-Variance Tradeoff:** By intentionally adding a little bit of *bias* (penalizing the weights), we can massively reduce the model's *variance* (overfitting to training data).
*   **Ridge Regression (L2 Penalty):** Minimizes $MSE + \lambda \sum \beta_i^2$. Shrinks coefficients close to zero, but **never exactly zero**. Good for multicollinearity.
*   **Lasso Regression (L1 Penalty):** Minimizes $MSE + \lambda \sum |\beta_i|$. Shrinks coefficients to **exactly zero**. Acts as built-in **Feature Selection**.
*   **Elastic Net:** Combines L1 and L2 penalties. 

### 5. Regression Evaluation Metrics
*   **MSE (Mean Squared Error):** Average squared errors. Heavily penalizes outliers.
*   **RMSE (Root Mean Squared Error):** $\sqrt{MSE}$. Brings error back to original units.
*   **MAE (Mean Absolute Error):** Average absolute errors. **Robust to outliers**.
*   **$R^2$ (Coefficient of Determination):** Measures the proportion of variance in the dependent variable ($y$) that is predictable from the independent variables ($X$). **Range:** $(-\infty, 1]$.
    *   *Formula:* $R^2 = 1 - \frac{\text{Sum of Squared Residuals (SSR)}}{\text{Total Sum of Squares (SST)}}$
    *   *Flaw:* It assumes every independent variable in the model helps to explain variation in the dependent variable. It mathematically **never decreases** as you add more features, leading to overfitting if relied upon blindly.
*   **Adjusted $R^2$:** 
    *   *Formula:* $1 - [\frac{(1 - R^2)(n - 1)}{n - k - 1}]$ (where $n$ is sample size, $k$ is number of features).
    *   *Why we need it:* It penalizes the model for adding features that do not actually improve predictions. It will **decrease** if a useless feature is added. **Always use Adjusted $R^2$ when evaluating multiple linear regression.**

### 6. Linear Regression Interview Cheatsheet
*   **Q: When does Linear Regression fail?**
    *   A: When relationships are non-linear, when outliers drag the OLS line away (because OLS squares errors, outliers exert massive leverage), or when features are highly correlated.
*   **Q: How do you handle outliers in Linear Regression?**
    *   A: Remove them, cap/clip them, use MAE to evaluate, or switch to robust models.
*   **Q: What is the difference between R-squared and Adjusted R-squared?**
    *   A: R-squared mathematically always increases as you add more features. Adjusted R-squared penalizes you for adding features that do not actually improve the model.
*   **Q: Does Linear Regression require feature scaling?**
    *   A: If using Ordinary Least Squares (OLS), no. However, if using **Gradient Descent**, **Ridge**, or **Lasso**, then **YES**. Feature scaling is required to ensure gradient descent converges efficiently and regularization penalties are applied equally.
*   **Q: How do you handle categorical variables?**
    *   A: Use One-Hot Encoding. *Interview Trap:* Always drop one of the newly created columns to avoid the "Dummy Variable Trap" (perfect multicollinearity).

---

## Part 1.5: Data Transformations & Feature Scaling
*(Often used to fix regression assumptions or prepare data for gradient descent)*

### 1. Feature Scaling (Changes the Range, NOT the Shape)
These techniques do not change the underlying distribution of your data; they just put all features on the same numerical playing field so the algorithm doesn't get biased by large numbers.
*   **Standardization (Z-Score / StandardScaler):** Centers data so the mean is $0$ and the standard deviation is $1$.
    *   *When to use:* This is your default choice for algorithms that use Gradient Descent (Logistic Regression, Neural Networks) or calculate distances between points (KNN, SVM). It is more robust to outliers than Min-Max scaling.
*   **Normalization (MinMaxScaler):** Squishes all values to fit strictly between $0$ and $1$.
    *   *When to use:* When your algorithm requires a strict bounded range (e.g., image pixels from $0-255$ scaled to $0-1$).
    *   *Interview Warning:* It is extremely sensitive to outliers! A single massive outlier will crush all your normal data points down to $0.0001$.

### 2. Distribution Transformations (Changes the Shape)
These techniques fundamentally alter the physical shape of the data curve.
*   **Log Transformation ($\log(x)$ or $\log(x+1)$):** Heavily compresses massive numbers while slightly spreading out smaller numbers.
    *   *When to use:* To fix **Right-Skewed data** (e.g., Income, House Prices, where a few massive values drag the tail to the right). To fix **Heteroscedasticity** (the expanding "cone" of errors in linear models) by applying it to the target variable $y$.
*   **Square Root Transformation ($\sqrt{x}$):** A milder version of the log transform.
    *   *When to use:* Typically used for Count Data (e.g., number of customers arriving per hour, number of defects) to stabilize variance.
*   **Box-Cox & Yeo-Johnson:** Advanced power transformations that use machine learning to automatically find the mathematically optimal exponent to make your data perfectly Normal (Gaussian).
    *   *When to use:* When your algorithm strictly assumes normality (like Gaussian Naive Bayes or Linear Discriminant Analysis).
    *   *Interview Gotcha:* Box-Cox only works on strictly positive numbers ($> 0$). If your data contains zeros or negatives, you must use Yeo-Johnson.

### 3. Non-Linear Transformations
*   **Polynomial Features ($X^2, X^3$):** Squares or cubes your existing features to create brand new columns.
    *   *When to use:* When the relationship between $X$ and $y$ is a curve, violating the Linearity assumption. It allows a linear algorithm to draw a curved line through the data.

---

## Part 2: Logistic Regression (Classification Baseline)

### 1. The Core Mathematical Intuition
**Goal:** Predict the probability that an instance belongs to a specific class (binary classification: 0 or 1).

**A. Why Linear Regression Fails for Classification**
1. **Outlier Sensitivity:** A straight line shifts drastically if you introduce an extreme outlier, changing your 0.5 decision boundary and ruining predictions.
2. **Meaningless Bounds:** Linear regression predicts continuous values that can fall below 0 or exceed 1 (meaningless probabilities).

**B. Odds vs. Probability & The Logit Function**
*   **Probability ($p$):** Bound between $0$ and $1$.
*   **Odds ($\frac{p}{1-p}$):** Bound between $0$ and $\infty$.
*   **The Logit Function (Log-Odds):** Taking the natural log of the odds stretches the bounds to $[-\infty, \infty]$. This allows us to map it to a linear equation:
    *   $\ln(\frac{p}{1-p}) = \beta_0 + \beta_1X$
    *   *Interpretation:* A 1-unit increase in $X$ changes the **log-odds** of the outcome by $\beta_1$ (not the probability directly).

**C. The Sigmoid Function**
Working backward (taking the inverse of the logit) gives us the Sigmoid function. To convert log-odds back into a usable probability, we pass the linear equation ($z = \beta_0 + \beta_1X$) through it:
*   **Formula:** $p = \frac{1}{1 + e^{-z}}$
*   *Effect:* It squishes any linear output into an S-curve strictly between 0 and 1.
![Logistic Regression Sigmoid Curve](./assets/sigmoid_curve.png)

### 2. The Cost Function (Log-Loss / Cross-Entropy)
*Interviewers love to ask why we don't use MSE for Logistic Regression.*

*   **The Problem with MSE:** Plugging the non-linear Sigmoid function into the MSE formula results in a **non-convex** error landscape (wavy with local minima). Gradient Descent will get permanently stuck. It also suffers from the *vanishing gradient* problem when predictions are confidently wrong.
*   **Log-Loss (Cross-Entropy):** The correct cost function. It creates a strictly **convex** bowl shape, guaranteeing a global minimum.
    *   **Formula:** $J(\beta) = - \frac{1}{N} \sum \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$
    *   **Behavior:** It heavily penalizes models that are "confidently wrong" (e.g., predicting 0.99 for a target that is actually 0) because the logarithmic penalty shoots to infinity.
*   **MLE Connection:** Log-Loss is mathematically derived from Maximum Likelihood Estimation (specifically, taking the Negative Log-Likelihood of a Bernoulli distribution).
![MSE vs Log-Loss Landscape](./assets/mse_vs_logloss.png)

### 2.5 How We Train Logistic Regression (Gradient Descent Example)
*The mathematical magic: Even though Log-Loss looks terrifying, its derivative simplifies to the EXACT same gradient formula as Linear Regression!*

Let's do a 1-variable hand-worked example: predicting class $y$ from feature $X$.
*   **Dataset:** $X = [1, 2]$, $y = [0, 1]$ *(Higher $X$ means class 1).*
*   **Step 1 (Initialize):** Let's guess the weight $\beta_1 = 0$ and intercept $\beta_0 = 0$.
*   **Step 2 (Linear Output):** Calculate $z = \beta_0 + \beta_1X$. For both data points, $z = 0$.
*   **Step 3 (Sigmoid Probabilities):** Pass $z$ through the Sigmoid function: $p = \frac{1}{1 + e^0} = \frac{1}{1+1} = 0.5$. Our predicted probabilities $\hat{y} = [0.5, 0.5]$.
*   **Step 4 (Find the Gradient):** The derivative of Log-Loss simplifies beautifully to: $\text{Gradient} = \frac{1}{n} \sum X_i (\hat{y}_i - y_i)$.
    *   Gradient for $\beta_1 = \frac{1}{2} \times [ 1(0.5 - 0) + 2(0.5 - 1) ]$
    *   $= \frac{1}{2} \times [ 0.5 - 1.0 ] = -0.25$
*   **Step 5 (Update Weight):** We move opposite to the gradient. Using learning rate $\alpha = 0.1$:
    *   $\text{New } \beta_1 = \text{Old } \beta_1 - (\alpha \times \text{Gradient})$
    *   $\text{New } \beta_1 = 0 - (0.1 \times -0.25) = 0.025$
*   **Result:** $\beta_1$ correctly increased to a positive number! Now, higher values of $X$ will result in higher probabilities (closer to 1), moving perfectly in the right direction.

### 2.6 Mathematical Derivation of Log-Loss Gradient (Optional)
**The Setup:**
* **Hypothesis:** $h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$
* **Cost Function (Log-Loss):**
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$

**Step 1: The Prerequisite Sigmoid Derivative**
Because the math gets messy, we first need to know the derivative of the Sigmoid function itself. A known property of the Sigmoid function $\sigma(z)$ is that its derivative is the function multiplied by one minus the function:
$$\frac{\partial}{\partial z} \sigma(z) = \sigma(z)(1 - \sigma(z))$$

*   **Crucial Interview Fact:** What is the maximum possible value for the gradient of the Sigmoid function? **0.25**. At $z=0$, $\sigma(0)=0.5$, so the gradient is $0.5 \times (1-0.5) = 0.25$. This tiny maximum gradient is the primary cause of the *Vanishing Gradient Problem* in deep neural networks!

When we apply the Chain Rule to take the derivative of our hypothesis with respect to our specific weight $\theta_j$, we get the Sigmoid derivative multiplied by the feature $x_j$:
$$\frac{\partial}{\partial \theta_j} h_\theta(x) = h_\theta(x)(1 - h_\theta(x)) x_j$$

**Step 2: Differentiate the Log Terms**
To make the algebra easier to see, let's ignore the summation and the $-\frac{1}{m}$ for a moment and just take the derivative of a single example's loss, which we will call $L$. In calculus, the derivative of $\log(x)$ is $\frac{1}{x}$. Applying the Chain Rule to both parts of the Log-Loss equation gives us:
$$\frac{\partial L}{\partial \theta_j} = y \left( \frac{1}{h_\theta(x)} \right) \frac{\partial h_\theta(x)}{\partial \theta_j} + (1 - y) \left( \frac{1}{1 - h_\theta(x)} \right) (-1) \frac{\partial h_\theta(x)}{\partial \theta_j}$$
*Explanation:* The $(-1)$ in the second term is incredibly important. It comes from applying the Chain Rule to the $(1 - h_\theta(x))$ part inside the second log.

**Step 3: Factor and Find a Common Denominator**
Let's factor out the $\frac{\partial h_\theta(x)}{\partial \theta_j}$ term that is shared by both sides, and combine the fractions by finding a common denominator of $h_\theta(x)(1 - h_\theta(x))$.
$$\frac{\partial L}{\partial \theta_j} = \left( \frac{y(1 - h_\theta(x)) - (1 - y)h_\theta(x)}{h_\theta(x)(1 - h_\theta(x))} \right) \frac{\partial h_\theta(x)}{\partial \theta_j}$$
If you expand the numerator ($y - y \cdot h_\theta(x) - h_\theta(x) + y \cdot h_\theta(x)$), the $y \cdot h_\theta(x)$ terms cancel out, leaving just:
$$\frac{\partial L}{\partial \theta_j} = \left( \frac{y - h_\theta(x)}{h_\theta(x)(1 - h_\theta(x))} \right) \frac{\partial h_\theta(x)}{\partial \theta_j}$$

**Step 4: The Beautiful Cancellation**
Now, substitute the Sigmoid derivative we calculated in Step 1 back into the equation.
$$\frac{\partial L}{\partial \theta_j} = \left( \frac{y - h_\theta(x)}{h_\theta(x)(1 - h_\theta(x))} \right) \left[ h_\theta(x)(1 - h_\theta(x)) x_j \right]$$
*Explanation:* Look closely! The denominator of our fraction is perfectly identical to the first half of the Sigmoid derivative. They completely cancel each other out. This reduces the massive equation down to:
$$\frac{\partial L}{\partial \theta_j} = (y - h_\theta(x)) x_j$$

**Step 5: Final Assembly**
Finally, we bring back the summation and the $-\frac{1}{m}$ that we temporarily dropped from the very beginning.
$$\frac{\partial}{\partial \theta_j} J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - h_\theta(x^{(i)})) x_j^{(i)}$$
To make this match our Linear Regression formula perfectly, we distribute the negative sign into the parentheses, which flips the $y$ and the $h_\theta(x)$ around. This gives us the exact same gradient formula as Linear Regression:
$$\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$$

### 3. The 4 Core Assumptions
Unlike Linear Regression's L.I.N.E., remember these for Logistic Regression:
1.  **Binary Outcome:** The target must be binary (for standard Logistic Regression).
2.  **Linearity of Log-Odds:** The features ($X$) must have a linear relationship with the *log-odds* of the target. *(Tested via Box-Tidwell test).*
3.  **No Multicollinearity:** Independent variables should not be highly correlated. *(Fix: Remove features, L1/L2 Regularization).*
4.  **Large Sample Size:** MLE needs more data to converge accurately compared to some algorithms like Naive Bayes or simple Decision Trees.

### 4. Thresholding & Business Context
*   **Default:** Classify as 1 if $p \ge 0.5$.
*   **When to Lower Threshold (e.g., to 0.3):** When missing a positive is dangerous/expensive (e.g., Cancer detection, Fraud). You are optimizing for **Recall**.
*   **When to Raise Threshold (e.g., to 0.8):** When false alarms are annoying/expensive (e.g., Spam filters). You are optimizing for **Precision**.

### 5. Multiclass Classification & The Softmax Function
*How to handle 3+ categories (e.g., Red, Green, Blue) instead of just binary 0 or 1:*
*   **One-vs-Rest (OvR):** Trains a separate binary classifier for each class (Red vs. Not Red, Green vs. Not Green) and chooses the one with the highest probability.
*   **Multinomial Logistic Regression (Softmax):** This is the true multi-class extension. Instead of using the Sigmoid function (which gives one probability for one class), it uses the **Softmax function**.
*   **The Softmax Formula:** $P(y=k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$
    *   *How it works:* It takes the raw output scores (logits, $z$) for all classes, exponentiates them to make them positive, and then divides each by the sum of all exponentiated scores.
    *   *The Result:* It outputs a **Probability Distribution**. All class probabilities will be strictly between 0 and 1, and they will **sum up to exactly 1.0**. (e.g., [Red: 0.7, Green: 0.2, Blue: 0.1]).

---

## Part 3: Classification Evaluation Metrics

### 1. The Confusion Matrix
For a binary classification problem (e.g., 1 = Positive/Defect, 0 = Negative/Normal):
*   **True Positive (TP):** Predicted 1, actual 1. (Correct detection).
*   **True Negative (TN):** Predicted 0, actual 0. (Correct rejection).
*   **False Positive (FP) - "Type I Error":** Predicted 1, actual 0. (False alarm).
*   **False Negative (FN) - "Type II Error":** Predicted 0, actual 1. (Missed detection).
![Confusion Matrix Heatmap](./assets/confusion_matrix.png)

### 2. Core Metrics
*   **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$. 
    *   *Avoid when:* **Highly Imbalanced Datasets.** (e.g., a model that always predicts "Not Fraud" on a 99% legitimate dataset has 99% accuracy but is useless).
*   **Precision (Quality of Positives):** $\frac{TP}{TP + FP}$. 
    *   *Use case:* When **False Positives** are costly (e.g., Spam detection - don't want good emails in spam).
*   **Recall / Sensitivity (Quantity of Positives):** $\frac{TP}{TP + FN}$. 
    *   *Use case:* When **False Negatives** are costly (e.g., Cancer detection - better to false alarm than miss cancer).
*   **F1-Score:** $2 \times \frac{Precision \times Recall}{Precision + Recall}$. 
    *   *Use case:* The harmonic mean. Use when you need an equal balance between Precision and Recall.
*   **F-$\beta$ Score:** A generalized version of the F1-score where you can assign more weight to either Precision or Recall based on business needs.
    *   *Formula:* $(1 + \beta^2) \times \frac{Precision \times Recall}{(\beta^2 \times Precision) + Recall}$
    *   *How $\beta$ affects it:*
        *   $\beta = 1$: Weights Precision and Recall equally (This is exactly the F1-score).
        *   $\beta > 1$ (e.g., F2-score): **Weights Recall higher**. Use when False Negatives are worse than False Positives (e.g., Disease detection).
        *   $\beta < 1$ (e.g., F0.5-score): **Weights Precision higher**. Use when False Positives are worse than False Negatives (e.g., Spam filter).

### 3. ROC Curve & PR Curve
*   **ROC Curve (Receiver Operating Characteristic):** Plots True Positive Rate (Recall) vs. False Positive Rate ($\frac{FP}{FP + TN}$) across thresholds.
    *   **AUC (Area Under the Curve):** Scalar value [0, 1]. *Interview Definition:* "Probability that the model ranks a random positive example more highly than a random negative example."
    ![ROC Curve and AUC](./assets/roc_curve.png)
*   **Precision-Recall (PR) Curve:** Plots Precision vs. Recall. 
    *   **When to use over ROC?** Use PR Curve for **severe class imbalance**. ROC can be overly optimistic due to True Negatives diluting the FPR formula.

### 4. Classification & Logistic Regression Interview Cheatsheet
*   **Q: What is the difference between Type I and Type II errors?**
    *   A: Type I is a False Positive (False Alarm). Type II is a False Negative (Missed Detection).
*   **Q: My model has 99% accuracy on a fraud detection dataset, but catches 0 frauds. Why?**
    *   A: Accuracy is misleading on imbalanced datasets (majority class trap). Look at Recall to see how many frauds you actually caught.
*   **Q: How do you decide whether to optimize for Precision or Recall?**
    *   A: Look at business costs. If false alarms are expensive (e.g., permanent bans), optimize Precision. If missing events is dangerous (e.g., engine failure), optimize Recall.
*   **Q: What happens to Precision and Recall if we increase the decision threshold from 0.5 to 0.7?**
    *   A: The model becomes more conservative. **Precision increases** (fewer false alarms), but **Recall decreases** (we miss more actual Positives).
*   **Q: Why is it called Logistic *Regression* if it's used for Classification?**
    *   A: Because under the hood, it fits a linear regression line to the *log-odds* of the probability: $\log(\frac{P}{1-P}) = \beta_0 + \beta_1X$.
*   **Q: Does Logistic Regression require feature scaling?**
    *   A: Yes! Because it uses Gradient Descent for optimization and often includes L1/L2 regularization. Unscaled features will distort regularization and slow down convergence.
*   **Q: Is Logistic Regression a Linear or Non-Linear classifier?**
    *   A: It is a **Linear Classifier**. Even though the output is a non-linear S-curve (Sigmoid), the underlying *decision boundary* that separates the classes in the feature space is a straight line (or flat hyperplane).
*   **Q: What is the `C` parameter in scikit-learn's Logistic Regression?**
    *   A: It controls Regularization (which scikit-learn applies by default!). **$C$ is the inverse of regularization strength ($\lambda$)**. Smaller $C$ = stronger penalty = simpler model (less overfitting). Larger $C$ = weaker penalty = fits training data more closely.
*   **Q: What is the maximum possible value for the gradient of the Sigmoid function $\sigma(z)$?**
    *   A: **0.25**. The derivative is $\sigma(z)(1-\sigma(z))$. At $z=0$, $\sigma(0)=0.5$, so the gradient is $0.5 \times 0.5 = 0.25$. This tiny maximum gradient is a primary cause of the *Vanishing Gradient Problem* in deep learning.

---

## Part 4: Support Vector Machines (SVM)

*These notes are heavily inspired by Josh Starmer's StatQuest series, blending his brilliant analogies (Mice & Drug Dosages) with interview-ready mathematical explanations.*

### 1. The Geometry & Math of the Margin

Before looking at data analogies, we must understand how SVM mathematically draws its boundaries. Unlike Logistic Regression, which draws any line to separate classes, SVM wants to build the **widest possible street** between the classes.
![SVM Training Process](./assets/svm_training_process.png)

**The Equations of the Street:**
*   **The Hyperplane (Decision Boundary):** The median of the street. Its equation is $w^T x + b = 0$.
*   **The Positive Gutter:** The right guardrail touching the closest positive data point. Its equation is $w^T x + b = 1$.
*   **The Negative Gutter:** The left guardrail touching the closest negative data point. Its equation is $w^T x + b = -1$.
*   **Support Vectors:** The specific data points that physically touch these guardrails. If you delete all other data points in the dataset, the model does not change.

**Why do we use exactly 1 and -1? (Scale Invariance)**
The actual geometric street doesn't care if the boundaries are labeled $1$ and $-1$, or $100$ and $-100$. If you multiply the weights $w$ and bias $b$ by $100$, it represents the exact same geometric street, just scaled differently. To stop the math from having infinite possible answers, we mathematically force (constrain) the closest points to equal exactly $+1$ and $-1$. This locks in a single, unique mathematical solution for $w$ and $b$.

**How Inference Works (Making Predictions):**
The $+1$ and $-1$ gutters are *only* used during training to calculate the margin width. Once the model is trained (we found the optimal $w$ and $b$), inference is incredibly simple. We take a new, unseen data point $x_{new}$ and calculate its score:

$$z = w^T x_{new} + b$$

*   If $z \ge 0 \implies$ Classify as **Positive (+1)**
*   If $z < 0 \implies$ Classify as **Negative (-1)**
*(Notice how the $\pm 1$ margins don't matter anymore for the final prediction; inference only cares which side of the $0$ hyperplane the point lands on!)*

**The Margin Math (Core Interview Question):**
The total width of this street (from the left guardrail to the right guardrail) is mathematically defined as:

$$\text{Margin Width} = \frac{2}{||w||}$$

Because SVM's primary goal is to make this street as wide as possible (maximize the margin), the optimization algorithm must do the exact mathematical opposite to the denominator. Therefore, the core optimization goal of SVM is to **minimize**:

$$\frac{1}{2} ||w||^2$$

**Hard Margin vs. Soft Margin (The $C$ Parameter):**
*   **Hard Margin (The Flawed Ideal):** A strict margin that allows absolutely zero data points inside the street or on the wrong side. If there is a single extreme outlier, a Hard Margin will severely contort the street to avoid it, leading to massive **Overfitting**.
*   **Soft Margin (The Realistic Fix):** We allow some data points to violate the margin. We control this using the **$C$ Parameter (Cost of Misclassification)**.
    *   **High $C$ (Strict):** The model severely penalizes mistakes. It draws a very **narrow margin** to get almost every training point correct. (Low Bias, High Variance $\rightarrow$ Overfitting).
    *   **Low $C$ (Relaxed):** The model doesn't mind a few mistakes. It prefers to keep the **margin as wide as possible**, ignoring extreme outliers. (High Bias, Low Variance $\rightarrow$ Underfitting).

---

### 2. The Main Idea: Maximal Margin & Soft Margins (The Mice Analogy)

**The Problem with Simple Thresholds:**
Imagine we are classifying mice based on their mass into **Not Obese (Red)** and **Obese (Green)**. 
If we just draw a line halfway between the closest Red and Green mouse, we create a **Maximal Margin Classifier**. 
*   **The Margin:** The shortest distance between the threshold and the closest observations.
*   **The Flaw:** It is super sensitive to outliers. If one skinny mouse randomly has a high mass, the threshold shifts drastically, and we suddenly misclassify lots of normal mice.

**The Solution: Support Vector Classifiers (Soft Margin)**
To solve the outlier problem, we must allow **misclassifications**. 
*   We use **Cross-Validation** to determine exactly how many misclassifications we should allow to get the best results on unseen data.
*   This introduces the **Bias-Variance Tradeoff**: By allowing a few mistakes on the training data (higher bias), the model becomes much more robust to new data (lower variance).
*   **Support Vectors:** The specific observations that sit on the edge of, or inside, this new "Soft Margin". 
![Soft Margin SVM Training](./assets/svm_soft_training_process.png)

---

### 3. The Polynomial Kernel (The Drug Dosage Analogy)

**When 1D Lines Fail:**
Imagine testing a drug dosage. 
*   Dosage too low = Not Cured (Red)
*   Dosage just right = Cured (Green)
*   Dosage too high = Not Cured (Red)

In a 1D number line, the Green dots are trapped between the Red dots. **No single point (threshold) can separate them.**

**Projecting to 2D:**
To fix this, we move the data into a higher dimension. We create a Y-axis by taking the **Dosage Squared ($X^2$)**.
*   Now, the low dosages have a small Y value.
*   The high dosages have a massive Y value.
*   The "just right" dosages have a medium Y value.
Suddenly, we can draw a straight 2D line (a Support Vector Classifier) right underneath the high/low Red dots and above the medium Green dots!
![Polynomial Kernel Projection](./assets/svm_polynomial_kernel.png)

**The Kernel Trick:**
Transforming data into higher dimensions is computationally expensive. 
*   The **Polynomial Kernel** calculates the high-dimensional relationships (the dot product) between every pair of points *as if* they were in a higher dimension, without actually transforming the data! 

---

### 4. The Radial Basis Function (RBF) Kernel

If the Polynomial Kernel moves data to 2D or 3D, the **RBF Kernel finds Support Vector Classifiers in Infinite Dimensions.**

**How it Works (Weighted Nearest Neighbor):**
Because we can't draw infinite dimensions, we visualize RBF as a **Weighted Nearest Neighbor** model. 
*   The influence one observation has on another is a function of the **Squared Distance** between them. 
*   The closer two points are, the more influence they have on each other's classification.

**The Gamma ($\gamma$) Parameter:**
Gamma scales the squared distance, controlling the influence:
*   **High Gamma:** Influence drops off very quickly. Points only care about their immediate neighbors. (Very strict, can lead to overfitting).
*   **Low Gamma:** Influence drops off slowly. Points care about neighbors far away. (Very relaxed, leads to smoother boundaries).
![RBF Gamma Parameter](./assets/svm_rbf_gamma.png)

**The Math of Infinity:**
How does it calculate infinite dimensions? 
The RBF kernel uses the exponential function ($e^{- \gamma \cdot \text{distance}^2}$). Using a **Taylor Series Expansion**, the function $e^x$ can be expanded into an infinite sum of polynomial terms ($1 + x + x^2/2! + x^3/3! ...$). Because it uses this infinite sum, it mathematically evaluates the dot product in infinite dimensions!

---

### 5. Support Vector Regression (SVR): Flipping the Goal

**How to Explain SVR in an Interview (The Garden Hose Analogy):**
Imagine laying down a thick, transparent garden hose over a scatter plot of data points on the floor. Your goal is to cover as many data points as possible with the hose. 
*   The thickness of the hose is **$\epsilon$**. 
*   You don't care exactly where the points sit *inside* the hose; as long as they are covered, their error is completely ignored (it is precisely $0$). 
*   You only penalize the points that stick out *outside* of the hose. SVR's mathematical goal is to lay the hose down in a way that minimizes the total distance of those outside points.

**The Core Intuition:**
SVM Classification tries to fit the widest possible empty street between two classes while keeping margin violations out. **SVM Regression (SVR) does the exact mathematical opposite:** it tries to fit as many training instances as possible *inside* the street while limiting margin violations (points that fall *outside* the street).

**The $\epsilon$ (Epsilon) Tube:**
Instead of a margin, SVR creates an $\epsilon$-insensitive tube around the regression line. 
*   The width of this tube is controlled by the hyperparameter **$\epsilon$**.
*   A **higher $\epsilon$** creates a wider tube (fitting more points inside, creating a simpler/flatter model).
*   A **lower $\epsilon$** creates a narrower tube (forcing the model to bend more tightly to the data, risking overfitting).
![Linear SVR Epsilon Tubes](./assets/svr_linear.png)
*(In the image above, the pink highlighted points are the Support Vectors! Notice how the wider street on the left captures almost all points easily, while the narrower street on the right has many points spilling out).*

**The $\epsilon$-Insensitive Property (Core Interview Fact):**
Any data point that falls *inside* the $\epsilon$-tube is considered "correct" and incurs zero loss. Because of this, adding more training points inside the margin does **not affect the model's predictions at all**. The model is completely insensitive to them!

**Slack Variables & Support Vectors in SVR:**
What happens to the points that fall *outside* the tube? 
*   We use **Slack Variables ($\xi$)** to measure how far outside the tube these errors are. 
*   The $C$ parameter dictates how heavily we penalize these slack variables (High $C$ = strict penalty for points outside the tube).
*   *Mind-Bending Fact:* In SVR, the points that fall **outside or on the edge** of the tube are the actual **Support Vectors**! They are the only points that dictate the shape of the regression line. 

**Non-Linear Regression:**
Just like classification, SVR can use the **Kernel Trick** (Polynomial or RBF kernels) to map data into higher dimensions. This allows SVR to draw incredibly complex, curved $\epsilon$-tubes to fit non-linear data effortlessly!
![Polynomial SVR Kernels](./assets/svr_poly.png)
*(In the image above, the model on the left has $C=100$, meaning it strictly penalizes points outside the tube, forcing the curve to wiggle and overfit. The model on the right has $C=0.01$, meaning it is very relaxed, resulting in a smoother, more generalized curve).*

#### The Mathematics of Support Vector Regression (SVR)

To formalize the "Garden Hose" analogy, we must define the prediction function, the loss function, and the optimization objective.

**1. The Prediction Function (The Center of the Tube)**
Just like in linear regression, the SVR model predicts a continuous value $y$ by computing the dot product of the weights $w$ and the input features $x$, plus a bias term $b$:

$$f(x) = w^T x + b$$

**2. The $\epsilon$-Insensitive Loss Function (The Tube Walls)**
The defining mathematical feature of SVR is its loss function. It states that if the absolute difference between the actual value $y$ and the predicted value $f(x)$ is less than $\epsilon$, the error is exactly zero. The model only incurs a penalty if the prediction falls outside this boundary:

$$L_\epsilon(y, f(x)) = \max(0, |y - f(x)| - \epsilon)$$

**3. The Optimization Objective (Soft Margin SVR)**
In reality, it is rarely possible to fit every single data point perfectly inside the $\epsilon$-tube. We must allow some points to exist outside the tube while still trying to keep the tube as flat and robust as possible.

To do this, we introduce **Slack Variables** ($\xi$ and $\xi^*$):

* $\xi_i$ represents the distance of a point that falls *above* the upper boundary of the tube.
* $\xi_i^*$ represents the distance of a point that falls *below* the lower boundary of the tube.

The goal of SVR is to keep the weights as small as possible (to keep the model simple and flat) while minimizing the sum of these slack variables (the errors outside the tube).

**The Core SVR Cost Function:**

$$\min_{w, b, \xi, \xi^*} \frac{1}{2} ||w||^2 + C \sum_{i=1}^{m} (\xi_i + \xi_i^*)$$

**Subject to the following constraints:**

1. The upper boundary constraint (points cannot be too high above the tube):

$$y_i - (w^T x_i + b) \leq \epsilon + \xi_i$$

2. The lower boundary constraint (points cannot be too far below the tube):

$$(w^T x_i + b) - y_i \leq \epsilon + \xi_i^*$$

3. Slack variables must be non-negative (distance cannot be negative):

$$\xi_i, \xi_i^* \geq 0$$

**How the Hyperparameters fit into the Math:**

* **Minimizing $\frac{1}{2} ||w||^2$:** This flattens the function, maximizing the generalized robustness of the tube.
* **The $C$ Parameter:** This is the trade-off multiplier attached to the sum of the slack variables $\sum (\xi_i + \xi_i^*)$. If $C$ is massive, the math heavily penalizes any $\xi$ greater than 0, forcing the model to violently contort the tube to capture outliers (Overfitting). If $C$ is small, the math allows $\xi$ to grow, creating a smoother, more generalized tube (Underfitting).

---

### 6. Placement Prep: SVMs in the Real World

#### A: The Visual Guide to the $C$ Parameter
The $C$ parameter dictates how strictly the model avoids misclassifications. It physically alters both the margin width and the number of Support Vectors.

| Parameter State | Margin Width | Number of Support Vectors | Impact on Bias/Variance |
| :--- | :--- | :--- | :--- |
| **Low $C$ (Relaxed)** | **Wide Margin** | **MORE Support Vectors** | High Bias, Low Variance (Underfitting risk) |
| **High $C$ (Strict)** | **Narrow Margin** | **FEWER Support Vectors** | Low Bias, High Variance (Overfitting risk) |

*Counter-intuitive fact:* A wider margin physically covers more space, meaning more data points will fall inside of it. Therefore, a lower $C$ results in **MORE** Support Vectors!

#### B: Solving an SVM "By Hand" (The Geometric Toy Example)
Let's solve a 2D SVM geometrically.
![SVM Geometric Solution](./assets/svm_by_hand.png)

**The Dataset:**
*   **Negative Class (-1):** Point $A$ at $(1, 1)$
*   **Positive Class (+1):** Point $B$ at $(3, 3)$

**Step 1: Place the Hyperplane**
The best decision boundary separates the classes perfectly in the middle. 
*   The midpoint between $(1,1)$ and $(3,3)$ is **$(2,2)$**.
*   The boundary must be perpendicular to the line connecting $A$ and $B$. Since the line $A \rightarrow B$ has a slope of $1$, the boundary must have a slope of $-1$. 
*   Equation of the boundary: $y - 2 = -1(x - 2) \implies x + y = 4 \implies$ **$x + y - 4 = 0$**.

**Step 2: Calculate the Margin**
The margin distance is the perpendicular distance from the midpoint $(2,2)$ to either point. 
*   Distance from $(2,2)$ to $(3,3)$: $\sqrt{(3-2)^2 + (3-2)^2} = \sqrt{1^2 + 1^2} =$ **$\sqrt{2}$**.
*   Both Point A and Point B lie exactly on the margin boundaries, meaning **both A and B are Support Vectors**.

**Step 3: Finding Weights ($w$) and Bias ($b$)**
Our SVM equations are $w^Tx + b = \pm 1$.
*   Since the normal vector to $x + y = 4$ is $(1, 1)$, $w$ must be some scalar multiple: $w = c(1, 1)$.
*   We know Margin Width $= \frac{2}{||w||}$.
*   The distance from the center to one margin is $\sqrt{2}$, so the total width is $2\sqrt{2}$.
*   $\frac{2}{||w||} = 2\sqrt{2} \implies ||w|| = \frac{1}{\sqrt{2}}$.
*   Since $w = c(1, 1)$, its magnitude is $\sqrt{c^2 + c^2} = c\sqrt{2}$. 
*   $c\sqrt{2} = \frac{1}{\sqrt{2}} \implies c = \frac{1}{2}$.
*   Therefore, **$w = [\frac{1}{2}, \frac{1}{2}]$**. 

To find $b$, plug $w$ and Point B $(3,3)$ into the positive margin equation:
*   $w^T x_B + b = 1 \implies (\frac{1}{2} \cdot 3) + (\frac{1}{2} \cdot 3) + b = 1 \implies 3 + b = 1 \implies$ **$b = -2$**.

*Final SVM Equation:* $0.5x_1 + 0.5x_2 - 2 = 0$.

#### C: Top 5 OA & Interview Questions
1.  **Q: Why do we need to scale features before applying SVM?**
    *   **A:** Because SVM relies on physical distances (margins and dot products) to draw boundaries. Unscaled features will drastically distort the geometry and ruin the margin calculation.
2.  **Q: How does SVM handle outliers?**
    *   **A:** Via the Soft Margin $C$ parameter. A low $C$ tells the model to ignore outliers and prioritize a wider, more generalized margin.
3.  **Q: What is the time complexity of training an SVM?**
    *   **A:** Roughly $O(N^2)$ to $O(N^3)$, depending on the kernel and implementation. This makes standard SVMs very bad for massive datasets.
4.  **Q: What happens if you delete a non-support vector from the dataset and retrain?**
    *   **A:** Nothing. The decision boundary is entirely defined by the Support Vectors. Deleting any other point has zero effect on the model.
5.  **Q: Why is the RBF kernel said to map to infinite dimensions?**
    *   **A:** Because the RBF kernel utilizes the exponential function ($e^x$). By the Taylor Series expansion, the exponential function expands into an infinite sum of polynomial terms, effectively computing the dot product in infinite dimensional space.




## Part 5: Decision Trees (Classification)

Decision Trees represent a shift from geometry to pure logic. Instead of measuring spatial distances or calculating margins, this algorithm plays a game of "20 Questions" with the data, making it highly interpretable and completely immune to the scale of your features. **You never need to apply feature scaling to a Decision Tree.**

### 1. The Core Intuition: Slicing the Space
A Decision Tree splits the dataset into smaller and smaller orthogonal (axis-parallel) boxes. 
*   It looks at a single feature and finds a threshold (e.g., `Weight > 20`).
*   It draws a straight horizontal or vertical line at that threshold.
*   It repeats this process inside the newly created boxes until the boxes contain mostly a single class.

### 2. The Math of the Split: Gini Impurity
To decide *which* feature to split on and at *what* threshold, the tree uses a cost function called **Gini Impurity** to measure how "mixed" a node is. The algorithm desperately wants to minimize this impurity.

$$ G_i = 1 - \sum_{k=1}^{n} p_{i,k}^2 $$

*   $n$ is the total number of classes.
*   $p_{i,k}$ is the ratio of class $k$ instances inside that specific node $i$.

**Gini Extremes:**
*   **$0$ (Perfectly Pure):** The box contains only one class. The tree stops splitting.
*   **$0.5$ (Maximum Impurity for Binary):** The box is a perfect 50/50 coin flip. The tree must split this node again.

---

### 3. The Overfitting Trap & Pruning
Because Decision Trees are **non-parametric**, they have no predefined constraints. If left alone, a tree will keep slicing the data until every single leaf has a Gini Impurity of 0. This results in a massive, hyper-complex tree that memorizes outliers (100% training accuracy) but fails miserably on new data. 

To fix this, we apply **Pruning** (cutting back the branches).

**Pre-Pruning (Early Stopping):**
We set strict hyperparameters before training to force the tree to stop growing early.
*   `max_depth`: The maximum number of levels the tree can grow.
*   `min_samples_split`: The minimum number of instances a node must have to be allowed to split.
*   `min_samples_leaf`: The minimum number of instances that must exist inside a final leaf node.

**Post-Pruning (Cost Complexity Pruning):**
We let the tree overfit completely, then work backwards from the leaves, snipping off specific branches that do not provide statistically significant improvements to accuracy.

---

### 4. Visualizing the Tree & Pruning in Python

To truly understand how Decision Trees slice the feature space and to see the catastrophic effects of an unconstrained tree, you can run the following standalone Python script. It trains two trees (one unconstrained and one pruned via early stopping) on a noisy `make_moons` dataset and generates the exact decision boundary plots, as well as the structural node map of the final pruned tree!

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Generate Toy 2D Dataset (Slightly noisy moons)
X, y = make_moons(n_samples=200, noise=0.25, random_state=42)

# 2. Train Two Models
# Unconstrained (will perfectly memorize the noise, causing severe overfitting)
tree_clf_unconstrained = DecisionTreeClassifier(random_state=42)
tree_clf_unconstrained.fit(X, y)

# Pruned (generalized via early stopping: limits depth to 3, ensures 5 samples per leaf)
tree_clf_pruned = DecisionTreeClassifier(min_samples_leaf=5, max_depth=3, random_state=42)
tree_clf_pruned.fit(X, y)

# 3. Generate Decision Boundary Plots
def plot_decision_boundary(clf, X, y, axes):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    
    # Plot the background boundary colors
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=plt.cm.brg)
    # Plot the raw data points
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=14)
    plt.ylabel(r"$x_2$", fontsize=14, rotation=0)

plt.figure(figsize=(12, 5))
plt.subplot(121)
plot_decision_boundary(tree_clf_unconstrained, X, y, [-1.5, 2.5, -1, 1.5])
plt.title("Unconstrained Tree (Overfitting)", fontsize=16)

plt.subplot(122)
plot_decision_boundary(tree_clf_pruned, X, y, [-1.5, 2.5, -1, 1.5])
plt.title("Pruned Tree (min_samples_leaf=5, max_depth=3)", fontsize=16)

plt.tight_layout()
plt.savefig("Tree_Boundaries_Comparison.png", dpi=300)

# 4. Generate Tree Structure Plot
plt.figure(figsize=(14, 10))
plot_tree(
    tree_clf_pruned, 
    filled=True, 
    rounded=True, 
    class_names=["Blue Square", "Green Triangle"], 
    feature_names=["x1", "x2"]
)
plt.title("Pruned Tree Structure", fontsize=18)
plt.savefig("Pruned_Tree_Structure.png", dpi=300)
```

**Output 1: Decision Boundaries**
Notice how the Unconstrained Tree creates jagged, unnatural slivers to capture single outlier points (100% training accuracy but terrible generalization). The Pruned Tree draws clean, robust rectangular boundaries that ignore the noise!
![Decision Boundaries Comparison](./assets/Tree_Boundaries_Comparison.png)

**Output 2: The Structural Map**
Here is the actual logic that the pruned model generated to draw those boxes. Notice how the root node splits on $x_2 \le 0.177$, and how the Gini impurity drops closer to $0$ as the boxes become purer!
![Pruned Tree Structure](./assets/Pruned_Tree_Structure.png)

---

### 5. Placement Prep: Top Decision Tree Interview Questions & Must-Knows

To ace a technical round on Decision Trees, you need to understand not just how they work, but their mathematical flaws and when *not* to use them.

#### A. The Advantages vs. Disadvantages (The "Trade-off" Question)
*   **Pros:** 
    *   **Highly Interpretable:** You can literally print out the tree and show stakeholders exactly *why* a decision was made (unlike Neural Networks).
    *   **No Scaling Required:** Because it splits on hard thresholds (`Age > 20`), it is completely immune to the scale of the features. You do not need to standardize or normalize your data.
    *   **Handles Collinearity:** If two features are highly correlated, the tree will just pick one to split on and ignore the other. It doesn't break the math (like it does in Linear Regression).
*   **Cons:**
    *   **High Variance (Instability):** This is the biggest flaw. A tiny change in the training data can cause the root node to choose a different feature, completely altering the entire structure of the tree below it. 
    *   **Orthogonal Boundaries Only:** Trees can only draw straight horizontal or vertical lines. If your true decision boundary is diagonal, the tree has to draw a jagged staircase to approximate it, which is highly inefficient.

#### B. Entropy vs. Gini Impurity
Interviewers will often ask how a tree decides to split. Scikit-learn defaults to **Gini Impurity**, but you can also use **Entropy / Information Gain**.
*   **Gini Impurity:** Measures the probability of misclassifying a random sample. It is slightly faster to compute because it doesn't use logarithms.
*   **Entropy:** Derived from thermodynamics and information theory, it measures the level of "disorder" or uncertainty in a node. 
    *   $H = -\sum p_k \log_2(p_k)$
    *   The goal of a split is to maximize **Information Gain** (the drop in Entropy from the parent node to the child nodes).

#### C. Top 4 Interview Questions
1.  **Q: Are Decision Trees parametric or non-parametric?**
    *   **A:** They are **non-parametric**. They do not assume a predetermined structure or mathematical function (like a straight line in regression). The number of parameters (nodes) grows with the complexity of the training data.
2.  **Q: What is the difference between a Classification Tree and a Regression Tree?**
    *   **A:** A Classification Tree splits data to minimize **Gini Impurity** and predicts the majority class in a leaf. A Regression Tree splits data to minimize **Mean Squared Error (MSE)** and predicts the average (mean) value of the instances in a leaf.
3.  **Q: Why do Decision Trees tend to overfit?**
    *   **A:** Because they are "greedy" algorithms that make the locally optimal choice at every split without looking ahead. Without constraints (like `max_depth`), they will recursively split the data until every single leaf contains exactly 1 data point (perfectly memorizing the training data and noise).
4.  **Q: Can a Decision Tree handle categorical data?**
    *   **A:** Conceptually, yes. However, implementation matters. Scikit-learn's CART implementation currently requires categorical variables to be converted to numerical values (e.g., via One-Hot Encoding) before training.