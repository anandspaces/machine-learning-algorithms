# Comparison of ML algorithms

import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Load Dataset
data = load_iris()
X, y = data.data, data.target

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "MLP": MLPClassifier(max_iter=1000),
    "AdaBoost": AdaBoostClassifier(algorithm='SAMME'),
}

# Evaluate Models
results = []

for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr') if y_prob is not None else "N/A"
    elapsed_time = time.time() - start_time
    results.append({"Model": name, "Accuracy": accuracy, "ROC-AUC": roc_auc, "Time (s)": elapsed_time})

# Convert Results to DataFrame
results_df = pd.DataFrame(results)

# Display Results
print("\nModel Comparison:")
print(results_df.sort_values(by="Accuracy", ascending=False))
