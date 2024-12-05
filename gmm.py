# Gaussian Mixture Model using Expectation-Maximization

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate synthetic data (2D Gaussian blobs)
X, _ = make_blobs(n_samples=500, centers=3, cluster_std=0.60, random_state=0)

# Visualize the synthetic data
plt.scatter(X[:, 0], X[:, 1], s=30, cmap='viridis')
plt.title("Generated Data")
plt.show()

# Fit a Gaussian Mixture Model (GMM) using sklearn's GaussianMixture
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# Predict the component labels for each data point
labels = gmm.predict(X)

# Visualize the clusters and the GMM components
plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')
plt.title("Clustered Data with GMM")
plt.show()

# Print the means and covariances of the Gaussian components
print("Means of the Gaussian components:")
print(gmm.means_)

print("\nCovariances of the Gaussian components:")
print(gmm.covariances_)

# Plot the GMM components
x = np.linspace(X[:, 0].min(), X[:, 0].max(), 1000)
y = np.linspace(X[:, 1].min(), X[:, 1].max(), 1000)
X_grid, Y_grid = np.meshgrid(x, y)
grid = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T

# Get the probability densities for each component
Z = gmm.score_samples(grid)
Z = Z.reshape(X_grid.shape)

# Plot the decision boundaries and Gaussian contours
plt.contour(X_grid, Y_grid, np.exp(Z), levels=10, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')
plt.title("GMM Contours and Data Points")
plt.show()
