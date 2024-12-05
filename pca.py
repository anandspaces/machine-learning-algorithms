# Principal Component Analysis for dimensionality reduction

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the Iris dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Target labels
feature_names = data.feature_names

# Standardize the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
n_components = 2  # Number of components to reduce to
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# Print explained variance ratio
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
print(f"Total Explained Variance: {sum(pca.explained_variance_ratio_):.2f}")

# Create a DataFrame for visualization
pca_df = pd.DataFrame(data=X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
pca_df['Target'] = y

# Plot PCA results
plt.figure(figsize=(8, 6))
for target, color in zip(np.unique(y), ['red', 'green', 'blue']):
    subset = pca_df[pca_df['Target'] == target]
    plt.scatter(subset['PC1'], subset['PC2'], label=data.target_names[target], color=color)
plt.title("PCA: Iris Dataset (2 Components)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()
