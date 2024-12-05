# Import necessary libraries
from kmodes.kmodes import KModes
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Create a sample dataset (categorical data)
data = {
    'Color': ['Red', 'Blue', 'Green', 'Blue', 'Green', 'Red', 'Green', 'Red', 'Blue', 'Green'],
    'Shape': ['Circle', 'Square', 'Triangle', 'Circle', 'Triangle', 'Square', 'Circle', 'Triangle', 'Square', 'Circle'],
    'Label': ['A', 'B', 'A', 'A', 'B', 'A', 'B', 'A', 'B', 'A']
}

# Convert dataset to a DataFrame
df = pd.DataFrame(data)

# Encode categorical data numerically
df_encoded = df.apply(lambda x: pd.factorize(x)[0])

# Specify the number of clusters
num_clusters = 2

# Perform K-modes clustering
kmodes = KModes(n_clusters=num_clusters, init='Huang', n_init=10, verbose=1)
clusters = kmodes.fit_predict(df_encoded)

# Add cluster labels to the original dataset
df['Cluster'] = clusters

# Display cluster centroids
print("Cluster Centroids:")
print(kmodes.cluster_centroids_)

# Visualize the clusters
sns.scatterplot(x=df_encoded.iloc[:, 0], y=df_encoded.iloc[:, 1],
                hue=df['Cluster'], palette='viridis', s=100, legend='full')
plt.title(f"K-modes Clustering (K={num_clusters})")
plt.xlabel("Encoded Feature 1")
plt.ylabel("Encoded Feature 2")
plt.legend(title="Cluster")
plt.show()

# Compute silhouette score (only works for more than one cluster)
if num_clusters > 1:
    silhouette_avg = silhouette_score(df_encoded, clusters)
    print(f"Silhouette Score for K-modes Clustering: {silhouette_avg:.2f}")
else:
    print("Silhouette Score cannot be computed for a single cluster.")
