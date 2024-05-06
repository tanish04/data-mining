Q1 data cleaning
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
diabetes_data = pd.read_csv(url, names=column_names)

# Replace missing values (e.g., 0) with NaN
diabetes_data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)

# Impute missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
diabetes_data_imputed = imputer.fit_transform(diabetes_data)

# Perform outlier detection using Isolation Forest
outlier_detector = IsolationForest(contamination=0.05)
outlier_mask = outlier_detector.fit_predict(diabetes_data_imputed)
cleaned_data = diabetes_data_imputed[outlier_mask == 1]

# Scale the data using RobustScaler
scaler = RobustScaler()
scaled_data = scaler.fit_transform(cleaned_data)

# Print cleaned and scaled dataset
print(pd.DataFrame(scaled_data, columns=column_names[:-1]))


Q2 DATA PREPROCESSING
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
iris = datasets.load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Adding a synthetic categorical column for aggregation
data['Species_category'] = iris.target

# Normalization
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data.drop('Species_category', axis=1))

# Transformation
pca = PCA(n_components=2)
transformed_data = pca.fit_transform(normalized_data)

# Aggregation
aggregated_data = data.groupby('Species_category').mean()

# Correlation
correlation_matrix = data.drop('Species_category', axis=1).corr()

# You can further explore other preprocessing techniques like handling missing values, outliers, etc.

Q3 K MEANS

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = np.array([[1, 1], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])

n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(data)

print("Cluster centroids:")
print(kmeans.cluster_centers_)

print("Assigned cluster labels:")
print(kmeans.labels_)


# Plot data before clustering
plt.scatter(data[:, 0], data[:, 1], label='Data')
plt.title('Data before Clustering')
plt.legend()
plt.show()

# Plot data after clustering with centroids
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, label='Data')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', s=200, c='red', label='Centroids')
plt.title('Data after Clustering')
plt.legend()
plt.show()

Q4 DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Generate some example data (you can replace this with your own dataset)
X, _ = make_moons(n_samples=500, noise=0.1)

# Initialize DBSCAN with parameters epsilon (eps) and minimum samples
dbscan = DBSCAN(eps=0.2, min_samples=5)

# Fit the model and predict clusters
labels = dbscan.fit_predict(X)

# Plotting the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

Q5 Agglomerative
rom sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Generate some example data (you can replace this with your own dataset)
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

# Initialize AgglomerativeClustering with the number of clusters
# You can also specify other parameters like linkage and affinity
agglomerative = AgglomerativeClustering(n_clusters=4)

# Fit the model and predict clusters
labels = agglomerative.fit_predict(X)

# Plotting the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('Agglomerative Hierarchical Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


Q6 DECISION TREE
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset (you can replace this with your own dataset)
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the DecisionTreeClassifier with optional parameters
# For example, you can specify max_depth, min_samples_split, etc.
decision_tree = DecisionTreeClassifier()

# Fit the model to the training data
decision_tree.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = decision_tree.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


Q7 NAIVE BAYES
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the Iris dataset (you can replace this with your own dataset)
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gaussian Naive Bayes classifier
naive_bayes = GaussianNB()

# Fit the model to the training data
naive_bayes.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = naive_bayes.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

Q8 Cross validation
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Load the Iris dataset (you can replace this with your own dataset)
iris = load_iris()
X = iris.data
y = iris.target

# Initialize the classifier
classifier = DecisionTreeClassifier()

# Perform k-fold cross-validation with k=5
# You can adjust the number of folds (cv parameter) as needed
scores = cross_val_score(classifier, X, y, cv=5)

# Print the cross-validation scores
print("Cross-validation scores:", scores)

# Calculate and print the mean accuracy of the cross-validation
print("Mean accuracy:", scores.mean())

Q9 holdout method
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset (you can replace this with your own dataset)
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets using holdout validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifier
classifier = DecisionTreeClassifier()

# Fit the model to the training data
classifier.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy
