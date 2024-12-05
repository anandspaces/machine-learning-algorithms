# K-means Clustering

# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the Iris dataset
data = load_iris()
X = data.data  # Features
y = data.target  # True labels (for evaluation purposes)

# Determine the optimal number of clusters using the Elbow Method
inertia_values = []
silhouette_scores = []
k_values = range(2, 11)  # Range of clusters to evaluate

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Plot the Elbow Method
plt.figure(figsize=(8, 4))
plt.plot(k_values, inertia_values, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid()
plt.show()

# Plot Silhouette Scores
plt.figure(figsize=(8, 4))
plt.plot(k_values, silhouette_scores, marker='o', color='purple')
plt.title("Silhouette Scores for Different K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid()
plt.show()

# Choose optimal K (e.g., K=3 based on the elbow plot)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(X)

# Add cluster labels to the dataset
df = pd.DataFrame(X, columns=data.feature_names)
df['Cluster'] = kmeans.labels_
df['True Label'] = y  # For comparison

# Plot clustering results (using first two features for visualization)
sns.scatterplot(x=df[data.feature_names[0]], y=df[data.feature_names[1]], 
                hue=df['Cluster'], palette='viridis', s=100, legend='full')
plt.title("K-means Clustering (K=3)")
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.legend(title="Cluster")
plt.show()

# Evaluate the clustering
silhouette_avg = silhouette_score(X, kmeans.labels_)
print(f"Silhouette Score for K={optimal_k}: {silhouette_avg:.2f}")

# Compare with true labels
sns.scatterplot(x=df[data.feature_names[0]], y=df[data.feature_names[1]], 
                hue=df['True Label'], palette='viridis', s=100, legend='full')
plt.title("True Labels")
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.legend(title="True Label")
plt.show()
