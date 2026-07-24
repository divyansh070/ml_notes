

## 1. Statistics & Probability

Before writing any code, you need a strong mathematical foundation to justify your decisions.

* **Probability:** Distributions (Normal, Binomial, Poisson, Uniform), Bayes' Theorem, Conditional probability.
* **Statistical Concepts:** Central Limit Theorem, Law of Large Numbers, Expected value, Variance.
* **Hypothesis Testing:** p-values, t-tests, ANOVA, Chi-Square tests, Type I vs. Type II errors.
* **Experimentation:** A/B testing setup, sample size determination, statistical significance, and handling novelty effects.

## 2. Data Processing & Feature Engineering

Models are only as good as the data fed into them. Expect questions on how you clean and prepare messy datasets.

* **Missing Data:** Imputation strategies (Mean/Median, KNN, predictive imputation) vs. dropping rows.
* **Outliers:** Detection techniques (Z-score, IQR, Isolation Forests) and treatment.
* **Categorical Encoding:** One-Hot Encoding vs. Label Encoding, Target Encoding, handling high-cardinality features.
* **Scaling:** Normalization (Min-Max) vs. Standardization (Z-score) and when to use which.
* **Imbalanced Data:** SMOTE, random oversampling/undersampling, and class weights.

## 3. Classical Machine Learning Models

You must know how these work under the hood, not just how to import them from `scikit-learn`.

* **Linear Models:** Linear Regression (Assumptions, Ordinary Least Squares), Ridge (L2 penalty), and Lasso (L1 penalty).
* **Classification:** Logistic Regression, Support Vector Machines (SVM, Kernels, Margin).
* **Tree-Based Models:** Decision Trees (Entropy, Gini Impurity, Information Gain), Random Forests (Bagging, Out-of-Bag error).
* **Boosting Algorithms:** Gradient Boosting, XGBoost, LightGBM (you will almost certainly be asked how boosting differs from bagging).
* **Instance-Based:** K-Nearest Neighbors (KNN), Naive Bayes (and why it is "naive").
* **Unsupervised Learning:** K-Means Clustering, Hierarchical Clustering, DBSCAN, Principal Component Analysis (PCA) for dimensionality reduction.

## 4. Model Evaluation & Optimization

Knowing how to build a model is useless if you can't measure how well it works.

* **Regression Metrics:** MSE, RMSE, MAE, R-squared, Adjusted R-squared.
* **Classification Metrics:** Confusion Matrix, Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC.
* **Trade-offs:** The Bias-Variance Tradeoff, Overfitting vs. Underfitting, and the Curse of Dimensionality.
* **Validation:** K-Fold Cross-Validation, Stratified splits, Time-series cross-validation.
* **Optimization:** Gradient Descent (Batch vs. Stochastic), Loss functions, Hyperparameter tuning (Grid search vs. Random search).

## 5. Deep Learning & Neural Networks

Required for computer vision, complex NLP, and specialized roles.

* **Fundamentals:** Multi-Layer Perceptrons (MLPs), Backpropagation, Chain rule, Epochs vs. Batches.
* **Activation Functions:** ReLU, Sigmoid, Tanh, Softmax (and the Vanishing/Exploding Gradient problem).
* **Optimizers:** Adam, RMSprop, SGD with Momentum.
* **Regularization:** Dropout layers, Batch Normalization, Early Stopping.
* **Architectures:** Convolutional Neural Networks (CNNs, pooling, filters) for vision, and Recurrent Neural Networks (RNNs, LSTMs, GRUs) for sequential data.

## 6. Natural Language Processing (NLP)

The bridge between classical ML and modern LLMs.

* **Text Preprocessing:** Tokenization, Stemming vs. Lemmatization, Stop-word removal.
* **Text Representation:** Bag of Words (BoW), TF-IDF.
* **Word Embeddings:** Word2Vec (Skip-gram, CBOW), GloVe, FastText.
* **Advanced Architecture:** The Attention Mechanism and the Transformer architecture (Self-attention, Multi-head attention).

## 7. Generative AI & LLMs (The Modern Tier)

As of 2024–2025, if the company builds AI products, you will be asked about this.

* **LLM Fundamentals:** Encoder vs. Decoder-only architectures (BERT vs. GPT), Token limits, Context windows.
* **RAG (Retrieval-Augmented Generation):**
* Vector Databases (e.g., Pinecone, Milvus) and Embedding models.
* Document chunking strategies (overlap, size limits).
* Retrieval techniques (Semantic search vs. keyword-based BM25, Hybrid search).
* Re-ranking algorithms.


* **Model Tuning:** Fine-tuning vs. Parameter-Efficient Fine-Tuning (PEFT), LoRA, QLoRA.
* **Prompting & Orchestration:** Few-shot prompting, Chain-of-Thought (CoT), Agents (LangChain, LlamaIndex, Tool calling, ReAct framework).

## 8. MLOps, System Design & Interpretability

Interviewers want to know if your models can survive in production.

* **Deployment:** REST APIs (FastAPI/Flask), Containerization (Docker), Batch vs. Real-time inference.
* **Monitoring:** Detecting Data Drift vs. Concept Drift, A/B Testing in production.
* **System Design:** Handling "Cold Starts" in recommendation engines, utilizing Feature Stores.
* **Explainable AI (XAI):** SHAP values, LIME, and extracting feature importance to explain models to non-technical stakeholders.