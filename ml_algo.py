# Comparison of ML algorithm on balanced and unbalanced datasets

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    accuracy_score,
)
import matplotlib.pyplot as plt

# Create a balanced dataset
X_balanced, y_balanced = make_classification(
    n_samples=1000, n_features=20, n_classes=2, weights=[0.5, 0.5], random_state=42
)

# Create an unbalanced dataset
X_unbalanced, y_unbalanced = make_classification(
    n_samples=1000, n_features=20, n_classes=2, weights=[0.9, 0.1], random_state=42
)

# Split datasets into training and testing sets
X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
    X_balanced, y_balanced, test_size=0.3, random_state=42
)
X_train_unbal, X_test_unbal, y_train_unbal, y_test_unbal = train_test_split(
    X_unbalanced, y_unbalanced, test_size=0.3, random_state=42
)

# Train Logistic Regression on both datasets
model_bal = LogisticRegression()
model_unbal = LogisticRegression()

model_bal.fit(X_train_bal, y_train_bal)
model_unbal.fit(X_train_unbal, y_train_unbal)

# Predictions and evaluation
def evaluate_model(model, X_test, y_test, dataset_type="Balanced"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{dataset_type} Dataset:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{dataset_type} (AUC = {roc_auc_score(y_test, y_prob):.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

# Evaluate models
plt.figure(figsize=(10, 6))
evaluate_model(model_bal, X_test_bal, y_test_bal, "Balanced")
evaluate_model(model_unbal, X_test_unbal, y_test_unbal, "Unbalanced")
plt.show()
