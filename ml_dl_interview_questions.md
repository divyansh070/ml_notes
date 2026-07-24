# Comprehensive Data Science Question Bank: ML & DL

This question bank is structured by algorithm/topic, and within each topic, questions are categorized into 4 levels: Basics, Intermediate, Advanced, and "By Hand" Calculations.

---

## Topic 1: Machine Learning Fundamentals & Data Preprocessing

### Level 1: Basics
1. What is the fundamental difference between supervised and unsupervised learning? Give an example of each.
2. Define the components of a Confusion Matrix (True Positive, True Negative, False Positive, False Negative).
3. What is one-hot encoding, and why is it necessary for categorical data?
4. How do you handle missing or corrupted data in a dataset?

### Level 2: Intermediate
5. Explain the Bias-Variance tradeoff. How does it relate to underfitting and overfitting?
6. What is cross-validation, and why is it preferred over a simple train-test split?
7. Explain the ROC curve and the AUC (Area Under the Curve) metric. In what scenarios is the PR (Precision-Recall) AUC preferred over ROC AUC?
8. Explain the conceptual difference between L1 (Lasso) and L2 (Ridge) regularization. When would you use one over the other?

### Level 3: Advanced ("Make You Think")
9. You have a highly imbalanced dataset (99% negative, 1% positive) for a medical diagnosis task. A colleague boasts a model with 99% accuracy. Why might this model be completely useless, and what evaluation strategy should you use to prove it?
10. Explain geometrically or mathematically how L1 Regularization (Lasso) tends to produce sparse models (weights exactly equal to zero), while L2 (Ridge) does not.

### Level 4: "By Hand" Mathematical Calculations
11. **Metrics Calculation:** A classification model predicts 100 positive instances. Out of these, 80 are true positives and 20 are false positives. There are also 40 false negatives. Calculate the Precision, Recall, and F1-score.
12. **Naive Bayes:** You have a dataset where $P(Class = A) = 0.6$ and $P(Class = B) = 0.4$. Given a feature $X$, $P(X | Class = A) = 0.3$ and $P(X | Class = B) = 0.8$. According to Bayes' Theorem, what is the probability of $Class = A$ given $X$?

---

## Topic 2: Supervised Learning (Regression & Classification)

### Level 1: Basics
13. What are the core assumptions of Linear Regression? What happens if they are violated?

### Level 2: Intermediate
14. How does a Decision Tree split nodes? Explain concepts like Gini Impurity and Information Gain conceptually.
15. Explain how a Random Forest works. Why does it generally perform better than a single decision tree?
16. What is the "kernel trick" in Support Vector Machines (SVM)?
17. How does the choice of 'K' in K-Nearest Neighbors (KNN) affect the bias-variance tradeoff?

### Level 3: Advanced ("Make You Think")
18. Describe the concept of Gradient Boosting. How does it differ fundamentally from Random Forests?
19. **Support Vector Machines:** Explain the conceptual idea behind the "Dual Formulation" of SVMs and how the "Kernel Trick" allows us to compute non-linear boundaries in high-dimensional spaces without explicitly transforming the data points.
20. **Decision Trees:** Is it possible for a Decision Tree to split on the exact same continuous feature multiple times in a single path from the root to a leaf node? Why or why not?

### Level 4: "By Hand" Mathematical Calculations
21. **Linear Regression:** Given the equation $y = 2x_1 + 3x_2 - 5$, if $x_1$ increases by 2 units and $x_2$ decreases by 1 unit, what is the net change in $y$?
22. **Gini Impurity & Information Gain by Hand:**
    You have a node with 10 samples (6 Positive, 4 Negative). You split it using a feature.
    *   The Left Child Node gets 5 samples (4 Positive, 1 Negative).
    *   The Right Child Node gets 5 samples (2 Positive, 3 Negative).
    Calculate the Gini Impurity of the parent node, the Gini Impurities of both child nodes, and the Weighted Gini Impurity of the split. Did this split reduce impurity?

---

## Topic 3: Unsupervised Learning

### Level 1: Basics
23. Explain how the K-Means clustering algorithm works conceptually. How do you typically choose the optimal 'K'?

### Level 2: Intermediate
24. What is Principal Component Analysis (PCA) and how does it reduce dimensionality while preserving variance?

### Level 3: Advanced ("Make You Think")
25. What happens to the K-Means algorithm if the true underlying clusters are non-spherical (e.g., crescent moons) or have vastly different densities? What alternative algorithms would you use?

### Level 4: "By Hand" Mathematical Calculations
26. **K-Means Clustering by Hand:**
    You have 1D data points: `[2, 3, 4, 10, 11, 12, 20, 25, 30]`.
    Your initial centroids are $c_1 = 2$ and $c_2 = 12$.
    *   **Iteration 1:** Assign points to the nearest centroid, then calculate the new centroids.
    *   **Iteration 2:** Re-assign points based on the new centroids, then calculate the final centroids for this step.

---

## Topic 4: Deep Learning Fundamentals

### Level 1: Basics
27. In neural network training, define the terms: Epoch, Batch Size, and Iteration.
28. What is the purpose of a Loss Function? Give one standard loss function for a regression task and one for a classification task.
29. What is the role of an Activation Function in a neural network? Name three common activation functions.

### Level 2: Intermediate
30. Explain the concept of Backpropagation. How do weights get updated in a neural network?
31. What is the difference between Batch Gradient Descent, Stochastic Gradient Descent (SGD), and Mini-batch Gradient Descent?
32. Why is dropout used in deep neural networks?
33. What is the vanishing gradient problem, and how do activation functions like ReLU help solve it conceptually?

### Level 3: Advanced ("Make You Think")
34. How does Batch Normalization work during training and inference, and why does it help networks train faster?
35. **Optimization Debugging:** If your neural network's training loss is oscillating wildly (jumping up and down significantly) instead of steadily decreasing, what are the most likely causes, and how would you fix them?
36. **Vanishing Gradients (Math):** Derive the derivative of the Sigmoid function $\sigma(x) = \frac{1}{1 + e^{-x}}$. Express the derivative in terms of $\sigma(x)$ itself. Using this result, explain mathematically why deep networks using sigmoid activations suffer from the vanishing gradient problem.

### Level 4: "By Hand" Mathematical Calculations
37. **Backpropagation by Hand:**
    Consider a simple 2-layer neural network with 1 input $x$, 1 hidden neuron $z$, and 1 output $y_{pred}$.
    *   Hidden layer: $z = w_1 \cdot x + b_1$
    *   Activation: $a = \text{ReLU}(z)$
    *   Output layer: $y_{pred} = w_2 \cdot a + b_2$
    *   Loss function (MSE for a single point): $L = \frac{1}{2}(y_{pred} - y_{true})^2$
    You are given one data point: $x = 2$, $y_{true} = 5$.
    Initial weights and biases: $w_1 = 0.5$, $b_1 = 1$, $w_2 = 2$, $b_2 = -1$.
    *   **Part A (Forward Pass):** Calculate the values of $z$, $a$, $y_{pred}$, and the Loss $L$.
    *   **Part B (Backward Pass):** Use the chain rule to calculate the gradients $\frac{\partial L}{\partial w_2}$ and $\frac{\partial L}{\partial w_1}$. Show your step-by-step calculus.

---

## Topic 5: CNNs, RNNs, and Advanced Architectures

### Level 1: Basics
38. What is the purpose of a pooling layer in a Convolutional Neural Network (CNN)?
39. How do Recurrent Neural Networks (RNNs) handle sequential data differently from feedforward networks? What is the main drawback of standard RNNs?

### Level 2: Intermediate
40. Briefly explain the concept of the attention mechanism in modern NLP models.
41. Explain the internal architecture of a standard LSTM (Long Short-Term Memory) cell. What specific problem does it solve compared to a vanilla RNN?
42. What is Transfer Learning? Describe the process of fine-tuning a large pre-trained model (like ResNet or BERT) for a specific downstream task.

### Level 3: Advanced ("Make You Think")
43. **Transformers (Math):** The Self-Attention mechanism in Transformers is defined as $Attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$. What is the mathematical reasoning behind scaling the dot product by $\frac{1}{\sqrt{d_k}}$? What happens if you omit it?

### Level 4: "By Hand" Mathematical Calculations
44. **CNN Output Size:** You have an input image of size $32 \times 32 \times 3$. You apply a convolutional layer with 10 filters of size $5 \times 5$, a stride of 1, and no padding (valid padding). What are the dimensions of the output feature map?
45. **RNN Step Unrolling by Hand:**
    Consider a simple RNN cell with a ReLU activation function: $h_t = \max(0, W_{hh} h_{t-1} + W_{xh} x_t + b_h)$.
    Given parameters: $W_{hh} = 0.5$, $W_{xh} = 2.0$, $b_h = -0.5$.
    The initial hidden state is $h_0 = 0$.
    You are given an input sequence of 3 time steps: $x_1 = 1$, $x_2 = -1$, $x_3 = 2$.
    Calculate the exact values of the hidden states $h_1$, $h_2$, and $h_3$.


---

## Topic 6: Statistics, Probability & Experimentation

### Level 1: Basics
46. What is the Central Limit Theorem and why is it important in machine learning?
47. Define Type I and Type II errors. Give an example where a Type II error is much worse than a Type I error.

### Level 2: Intermediate
48. What is a p-value? How is it used in Hypothesis Testing?
49. Explain the setup of an A/B test. How do you determine the required sample size?

### Level 3: Advanced ("Make You Think")
50. What are "novelty effects" and "network effects" in A/B testing, and how do they bias your results?

### Level 4: "By Hand" Mathematical Calculations
51. **Binomial Distribution:** A biased coin has a 60% chance of landing heads. If you flip it 5 times, calculate the exact probability of getting exactly 3 heads.

---

## Topic 7: Advanced Data Processing & Evaluation

### Level 1: Basics
52. Explain the difference between Normalization (Min-Max scaling) and Standardization (Z-score). When would you use one over the other?
53. What are the common regression metrics (MSE, RMSE, MAE, R-squared)?

### Level 2: Intermediate
54. How do you detect outliers in a dataset (e.g., Z-score vs IQR method)? How should you handle them?
55. What is the "Curse of Dimensionality" and how does it affect distance-based algorithms like KNN?

### Level 3: Advanced ("Make You Think")
56. Explain the difference between Grid Search and Random Search for hyperparameter tuning. Why is Random Search often more efficient?

### Level 4: "By Hand" Mathematical Calculations
57. **Z-Score Calculation:** Given a feature column with values `[2, 4, 4, 4, 5, 5, 7, 9]`. The mean is `5` and standard deviation is `2`. Calculate the Z-score for the value `9`. Is it an outlier (assuming a threshold of 3)?

---

## Topic 8: NLP, Generative AI & LLMs

### Level 1: Basics
58. What is the difference between Bag of Words (BoW) and TF-IDF?
59. What are Word Embeddings (like Word2Vec) and why are they superior to one-hot encoding for text?

### Level 2: Intermediate
60. Explain the difference between Encoder-only (like BERT) and Decoder-only (like GPT) architectures.
61. What is RAG (Retrieval-Augmented Generation)? Explain the basic pipeline involving Vector Databases.

### Level 3: Advanced ("Make You Think")
62. What is Parameter-Efficient Fine-Tuning (PEFT), and how does LoRA (Low-Rank Adaptation) work conceptually to fine-tune massive LLMs?

### Level 4: "By Hand" Mathematical Calculations
63. **TF-IDF Calculation:** You have two short documents: Doc 1 ("the cat sat"), Doc 2 ("the dog sat on the mat"). Calculate the Term Frequency (TF) for "cat" in Doc 1, and the Inverse Document Frequency (IDF) for "the" across the corpus.

---

## Topic 9: MLOps, System Design & Interpretability

### Level 1: Basics
64. What is the difference between Data Drift and Concept Drift in production ML systems?

### Level 2: Intermediate
65. Explain how SHAP values or LIME are used for Explainable AI (XAI) to interpret complex models.

### Level 3: Advanced ("Make You Think")
66. System Design: How would you handle the "Cold Start" problem in a recommendation engine for a new user with no history?

### Level 4: "By Hand" Mathematical Calculations
67. **Data Drift (PSI):** Imagine a feature's distribution across two bins (A and B).
    Training Data distribution: Bin A = 40%, Bin B = 60%.
    Production Data distribution: Bin A = 50%, Bin B = 50%.
    Calculate the Population Stability Index (PSI) for this feature by hand.


---

## Topic 10: Recommendation Systems & Time Series

### Level 1: Basics
68. What is the fundamental difference between Collaborative Filtering and Content-Based Filtering?
69. In Time Series analysis, what does it mean for a series to be "Stationary", and why is it important?

### Level 2: Intermediate
70. Explain how Matrix Factorization works in recommendation systems (e.g., using ALS or SVD).
71. What do the AR, I, and MA components stand for in an ARIMA model?

### Level 3: Advanced ("Make You Think")
72. How do you evaluate ranking algorithms? Explain metrics like NDCG (Normalized Discounted Cumulative Gain) and MAP@K (Mean Average Precision at K).
73. How do you detect and remove seasonality and trend from a time series dataset before modeling?

### Level 4: "By Hand" Mathematical Calculations
74. **NDCG Calculation:** You recommend 3 items to a user. The true relevance scores of these items in your recommended order are `[3, 0, 2]`. The ideal ordering (highest relevance first) would be `[3, 2, 0]`.
    Calculate the DCG and the NDCG for this recommendation list. (Use base 2 for the log penalty: $DCG = \sum \frac{rel_i}{\log_2(i+1)}$).

---

## Topic 11: SQL, Data Wrangling & OA Preparation

### Level 1: Basics
75. Explain the differences between INNER JOIN, LEFT JOIN, RIGHT JOIN, and FULL OUTER JOIN.
76. In Pandas, what is the difference between `loc` and `iloc`?

### Level 2: Intermediate
77. Explain SQL Window Functions. What is the difference between `ROW_NUMBER()`, `RANK()`, and `DENSE_RANK()`?
78. What is "Vectorization" in Pandas/NumPy, and why is it significantly faster than using `.apply()` or `for` loops?

### Level 3: Advanced ("Make You Think")
79. If a SQL query is running extremely slowly on a massive table, what steps would you take to optimize it? (e.g., Execution plans, Indexes, Partitioning).

### Level 4: "By Hand" Mathematical Calculations
80. **SQL Execution by Hand:**
    You have a table `Sales` with columns `(employee_id, amount)`.
    Data: `(1, 100), (1, 200), (2, 50), (2, 50), (3, 300)`.
    Manually execute this query and provide the exact output table:
    ```sql
    SELECT employee_id, SUM(amount) as total_sales
    FROM Sales
    GROUP BY employee_id
    HAVING SUM(amount) > 100
    ORDER BY total_sales DESC;
    ```

---

## Topic 12: Core Mathematical Foundations

### Level 1: Basics
81. Define the dot product of two vectors. Geometrically, what does it represent if the vectors are normalized?

### Level 2: Intermediate
82. What are Eigenvectors and Eigenvalues? How do they relate mathematically to Principal Component Analysis (PCA)?

### Level 3: Advanced ("Make You Think")
83. Explain the difference between Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP). How does MAP relate to Regularization in machine learning?

### Level 4: "By Hand" Mathematical Calculations
84. **Matrix Multiplication:**
    Given Matrix $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and Vector $x = \begin{bmatrix} 5 \\ 6 \end{bmatrix}$.
    Calculate the resulting vector $Ax$.


---

## Topic 13: Business Acumen, Product Sense & Strategy

### Level 1: Basics
85. In product analytics, define the metrics: DAU/MAU ratio, CAC (Customer Acquisition Cost), and LTV (Customer Lifetime Value). Why are they important for a Data Scientist?
86. What is the difference between an offline metric (like AUC or F1-score) and an online metric (like Click-Through Rate or Revenue)?

### Level 2: Intermediate
87. A product manager comes to you and says, "We need to predict which users will cancel their subscription." How do you translate this vague business problem into a concrete Machine Learning problem? (Define the target, timeframe, and features).
88. How do you decide whether to build a simple, highly interpretable model (like Logistic Regression) versus a complex, highly accurate "black box" model (like a deep neural network) for a business problem?

### Level 3: Advanced ("Make You Think")
89. "Our user retention dropped by 5% last week." As a Data Scientist, how would you systematically investigate the root cause of this drop?
90. You trained a new Deep Learning recommendation model that increases offline accuracy by 1%. However, deploying it will increase cloud compute costs by 200% and add 50ms of latency. How do you evaluate if deploying this model is a good business decision?

### Level 4: "By Hand" Mathematical Calculations
91. **Cost-Benefit Matrix Analysis:** You are building a fraud detection model.
    - A False Positive (blocking a real transaction) costs the business $10 in customer service.
    - A False Negative (letting fraud through) costs the business $500 in chargebacks.
    - True Positives and True Negatives cost $0.
    Your model's Confusion Matrix on a test set of 100 transactions is:
    TP: 10, FP: 20, FN: 5, TN: 65.
    Calculate the total expected cost of this model's errors. If a "dumb" rule-based system currently costs $3,000 on the same 100 transactions, should you deploy your ML model?
