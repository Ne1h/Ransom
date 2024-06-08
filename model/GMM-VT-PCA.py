import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import time

# Load dataset
file_path = 'C:/Users/Administrator/Documents/2extracted_features.csv'
data = pd.read_csv(file_path)

# Drop the 'File' column as it's not needed for model building
X = data.drop(['File', 'Labels'], axis=1)  # Drop 'Labels' from features
y = data['Labels']

# Check for missing values
X.isnull().sum()

# If there are missing values, fill missing values with 0
X.fillna(0, inplace=True)

# Measure total computation time
start_time = time.time()

# Remove low-variance features
selector = VarianceThreshold(threshold=(.82 * (1 - .82)))  # Removing features with more than 82% same values
X_var = selector.fit_transform(X)

# Reduce dimensionality
pca = PCA(n_components=20)  # Adjust number of components as needed
X_pca = pca.fit_transform(X_var)


# Apply Gaussian Mixture Model (GMM) clustering
gmm = GaussianMixture(n_components=2, random_state=7)  # Clustering into 2 clusters: ransomware and benign
gmm.fit(X_pca)

# Predict cluster labels
cluster_labels = gmm.predict(X_pca)

# Calculate total computation time
total_computation_time = (time.time() - start_time) * 1000  # Convert to milliseconds

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

# Confusion matrix to get TP, TN, FP, FN
tn, fp, fn, tp = confusion_matrix(y, cluster_labels).ravel()

# Print metrics
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Accuracy: {accuracy}')
print(f'F1 Score: {f1}')
print(f'True Positives (TP): {tp}')
print(f'True Negatives (TN): {tn}')
print(f'False Positives (FP): {fp}')
print(f'False Negatives (FN): {fn}')
print(f'Total Computation Time: {total_computation_time} milliseconds')
