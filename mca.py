# Multiple Correspondence Analysis for dimensionality reduction

# Import necessary libraries
import pandas as pd
import prince
import matplotlib.pyplot as plt

# Example Dataset (Categorical data)
data = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'Education': ['High School', 'Bachelors', 'Masters', 'Masters', 'High School', 'Bachelors', 'Masters', 'High School'],
    'Employment': ['Employed', 'Unemployed', 'Employed', 'Unemployed', 'Employed', 'Unemployed', 'Employed', 'Employed'],
    'Marital Status': ['Single', 'Married', 'Married', 'Single', 'Married', 'Single', 'Married', 'Single']
}

# Convert data to a DataFrame
df = pd.DataFrame(data)

# Print the dataset
print("Original Dataset:")
print(df)

# Apply MCA
mca = prince.MCA(n_components=2, random_state=42)
mca_result = mca.fit(df)

# Transform the data
mca_transformed = mca.transform(df)

# Print the explained variance
print("\nExplained Variance:")
print(mca.explained_inertia_)

# Create a DataFrame for visualization
mca_df = pd.DataFrame(mca_transformed, columns=['Dim1', 'Dim2'])

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(mca_df['Dim1'], mca_df['Dim2'], color='blue', alpha=0.7, s=100)
plt.title('MCA: Dimensionality Reduction')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid()
plt.show()
