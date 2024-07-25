import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Set this to the number of cores you want to use

# Load the data
data = pd.read_csv('Mall_Customers.csv')

# Display the first few rows of the data
print(data.head())

# Selecting relevant features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Display the first few rows of the features
print(X.head())

# Using the Elbow Method to find the optimal number of clusters
inertia = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Applying K-means Clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)

# Add the cluster labels to the original data
data['Cluster'] = clusters

# Display the first few rows of the data with cluster labels
print(data.head())

# Visualizing the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis', data=data, s=100, alpha=0.7, edgecolors='w')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
