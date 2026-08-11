import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, auc

out_dir = "/Users/divyanshverma/Desktop/ml_interview_questions/assets/"

# Set style
sns.set_theme(style="whitegrid")

# 1. Linear Fit & Residuals
np.random.seed(42)
x = np.random.rand(20) * 10
y = 2.5 * x + np.random.randn(20) * 2
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)
y_pred = model.predict(x.reshape(-1, 1))

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, y_pred, color='red', label='Line of Best Fit (OLS)')
for i in range(len(x)):
    plt.plot([x[i], x[i]], [y[i], y_pred[i]], color='gray', linestyle='dotted')
plt.title('Linear Regression: Best Fit & Residuals')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.legend()
plt.tight_layout()
plt.savefig(out_dir + 'linear_fit_residuals.png', dpi=300)
plt.close()

# 2. Gradient Descent Bowl
beta = np.linspace(-10, 10, 100)
cost = beta**2
plt.figure(figsize=(8, 5))
plt.plot(beta, cost, color='blue')
# Draw steps
steps = [-9, -6, -3, -1, 0]
for i in range(len(steps)-1):
    plt.annotate('', xy=(steps[i+1], steps[i+1]**2), xytext=(steps[i], steps[i]**2),
                 arrowprops=dict(facecolor='red', shrink=0, width=1.5, headwidth=8))
plt.title('Gradient Descent: Convex Cost Function')
plt.xlabel('Weight (beta)')
plt.ylabel('Cost (MSE)')
plt.tight_layout()
plt.savefig(out_dir + 'gradient_descent_bowl.png', dpi=300)
plt.close()

# 3. Heteroscedasticity
x_het = np.random.rand(100) * 10
residuals_het = np.random.randn(100) * x_het
plt.figure(figsize=(8, 5))
plt.scatter(x_het, residuals_het, color='purple', alpha=0.6)
plt.axhline(0, color='red', linestyle='dashed')
plt.title('Heteroscedasticity: Cone-Shaped Residuals')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.tight_layout()
plt.savefig(out_dir + 'heteroscedasticity.png', dpi=300)
plt.close()

# 4. Sigmoid Curve
z = np.linspace(-10, 10, 100)
p = 1 / (1 + np.exp(-z))
plt.figure(figsize=(8, 5))
plt.plot(z, p, color='blue', linewidth=2, label='Sigmoid Function')
plt.axhline(0.5, color='red', linestyle='dotted', label='Decision Boundary (p=0.5)')
plt.axvline(0, color='gray', linestyle='dashed')
plt.title('Logistic Regression: The Sigmoid Curve')
plt.xlabel('Logits (z = b0 + b1X)')
plt.ylabel('Probability (p)')
plt.legend()
plt.tight_layout()
plt.savefig(out_dir + 'sigmoid_curve.png', dpi=300)
plt.close()

# 5. MSE vs Log Loss
p_vals = np.linspace(0.01, 0.99, 100)
log_loss_1 = -np.log(p_vals)
mse_1 = (1 - p_vals)**2

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(p_vals, mse_1, color='orange')
plt.title('MSE Loss (when y=1)\nNon-Convex with Sigmoid')
plt.xlabel('Predicted Probability (p)')
plt.ylabel('Cost')

plt.subplot(1, 2, 2)
plt.plot(p_vals, log_loss_1, color='blue')
plt.title('Log-Loss (when y=1)\nStrictly Convex')
plt.xlabel('Predicted Probability (p)')
plt.ylabel('Cost')
plt.tight_layout()
plt.savefig(out_dir + 'mse_vs_logloss.png', dpi=300)
plt.close()

# 6. ROC Curve
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.3, 0.7, 0.85, 0.15])
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('Receiver Operating Characteristic (ROC)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(out_dir + 'roc_curve.png', dpi=300)
plt.close()

# 7. Confusion Matrix Heatmap
cm = np.array([[45, 5], [10, 40]]) # [[TN, FP], [FN, TP]]
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative (0)', 'Positive (1)'], 
            yticklabels=['Negative (0)', 'Positive (1)'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Truth')
plt.xlabel('Model Prediction')

# Add specific labels (adjusting for Seaborn heatmap coordinate system)
plt.text(0.5, 0.3, 'True Negative (TN)', ha='center', va='center', color='black')
plt.text(1.5, 0.3, 'False Positive (FP)', ha='center', va='center', color='black')
plt.text(0.5, 1.3, 'False Negative (FN)', ha='center', va='center', color='white')
plt.text(1.5, 1.3, 'True Positive (TP)', ha='center', va='center', color='white')

plt.tight_layout()
plt.savefig(out_dir + 'confusion_matrix.png', dpi=300)
plt.close()

print("Plots successfully generated and saved.")
