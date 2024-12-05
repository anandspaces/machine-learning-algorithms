# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target  # True labels (for evaluation purposes)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform hierarchical clustering (using Ward's method)
linked = linkage(X_scaled, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=12)
plt.title("Hierarchical Clustering Dendrogram (Truncated)")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

# Choose the number of clusters and form flat clusters
num_clusters = 3  # Based on dendrogram observation
labels = fcluster(linked, num_clusters, criterion='maxclust')

# Create a DataFrame to analyze clustering
df = pd.DataFrame(X, columns=data.feature_names)
df['Cluster'] = labels
df['True Label'] = y

# Visualize the clusters (using the first two features)
sns.scatterplot(x=df[data.feature_names[0]], y=df[data.feature_names[1]],
                hue=df['Cluster'], palette='viridis', s=100, legend='full')
plt.title(f"Hierarchical Clustering (K={num_clusters})")
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.legend(title="Cluster")
plt.show()

# Evaluate clustering using silhouette score
silhouette_avg = silhouette_score(X_scaled, labels)
print(f"Silhouette Score for Hierarchical Clustering (K={num_clusters}): {silhouette_avg:.2f}")

# Compare with true labels
sns.scatterplot(x=df[data.feature_names[0]], y=df[data.feature_names[1]],
                hue=df['True Label'], palette='viridis', s=100, legend='full')
plt.title("True Labels")
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.legend(title="True Label")
plt.show()
