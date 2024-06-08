import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, precision_score, recall_score, accuracy_score, f1_score

# Load dataset
file_path = 'C:/Users/Administrator/Documents/permissions_and_intents.csv'
data = pd.read_csv(file_path)

# Drop the 'File' column as it's not needed for model building
X = data.drop('File', axis=1)
y = data['Labels'] 

# Check for missing values
X.isnull().sum()

# If there are missing values, fill missing values with 0
X.fillna(0, inplace=True)

# Remove low-variance features
selector = VarianceThreshold(threshold=(.82 * (1 - .82)))  # Removing features with more than 82% same values
X_var = selector.fit_transform(X)

# Reduce dimensionality
pca = PCA(n_components=10)  # Adjust number of components as needed
X_pca = pca.fit_transform(X_var)

# Apply K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0)  # Clustering into 2 clusters: ransomware and benign
kmeans.fit(X_pca)

# Predict cluster labels
cluster_labels = kmeans.predict(X_pca)

# Calculate silhouette score to evaluate clustering
silhouette_avg = silhouette_score(X_pca, cluster_labels)
print(f'Silhouette Score: {silhouette_avg}')

# Optionally, append cluster labels to the original data
data['Cluster'] = cluster_labels
print(data)

# Evaluate clustering
precision = precision_score(y, cluster_labels, average='macro')
recall = recall_score(y, cluster_labels, average='macro')
accuracy = accuracy_score(y, cluster_labels)
f1 = f1_score(y, cluster_labels, average='macro')

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')