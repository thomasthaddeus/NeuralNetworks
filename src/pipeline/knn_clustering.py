from sklearn.neighbors import KNeighborsClassifier

# Let's say you have some training data in `X_train` and `y_train`
X_train = ...
y_train = ...

# Initialize the KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the data
knn.fit(X_train, y_train)
