import mglearn
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


iris_dataset = load_iris()

print(f"Keys of iris_dataset: {iris_dataset.keys()}")

# Brief description
print(iris_dataset['DESCR'][:193] + "\n...")

# Properties (width...)
print(iris_dataset['feature_names'])

# Specie names
print(iris_dataset['target_names'])

# Specie name encoded in numbers 0-2
print(iris_dataset['target'])

# Values
print(iris_dataset['data'][:3])
print(f"Data shape of iris_dataset: {iris_dataset['data'].shape}")

# Separate training and test data (75% and 25%)
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")


print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Access data (create dataframe from X_train, label columns using iris_dataset.feature_names,
# create scatter matrix, color by y_train
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names) # why not [""]?
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins':20}, s=60,
                                 alpha=0.8, cmap=mglearn.cm3)
plt.show()

# Build model using k-Nearest Neighbors classifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Make prediction

X_new = np.array([[5, 2.9, 1, 0.2]]) # 2D np array
prediction = knn.predict(X_new)
print(f"Prediction: {prediction}")
print(f"Prediction target name: {iris_dataset['target_names'][prediction]}")

# Evaluate the model

y_prediction = knn.predict(X_test)
print(f"Test set prediction: {y_prediction}")
print(f"Test set score: {np.mean(y_prediction == y_test)}")
print(f"Test set score: {knn.score(X_test, y_test)}")