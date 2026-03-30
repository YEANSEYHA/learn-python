# Module 5, Lesson 5: k-Nearest Neighbors (KNN)
# Book Reference: Chapter 11 - Classifiers

# Problem 1: Find the best k for KNN
# - Load Iris dataset, split 80/20, scale with StandardScaler
# - Try k = 1, 3, 5, 7, 9, 11, 13, 15
# - Print accuracy for each k
# - Print which k gives the best accuracy

#  Load Iris
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris = load_iris()
X, y = iris.data, iris.target

# Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


# Scale features (critical for KNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


n_neighbors = [1,3,5,7,9,11,13,15]
results = []

for k in n_neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    

    y_pred = knn.predict(X_test_scaled)
    data = {
        "k" : k,
        "accuracy" : accuracy_score(y_test, y_pred)
    }
    results.append(data)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    # Output: Accuracy: 1.00


best = max(results , key = lambda x:x['accuracy'])

print(best)

# Problem 2: Scaling matters
# - Use k=9 on Iris dataset (test_size=0.5, random_state=42)
# - Run KNN with StandardScaler → print accuracy
# - Run KNN without scaling → print accuracy
# - Compare the two results

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)




knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Problem 3: KNN for Regression
# - Load diabetes dataset: from sklearn.datasets import load_diabetes
# - Split 80/20 (random_state=42), scale with StandardScaler
# - Use KNeighborsRegressor(n_neighbors=5) from sklearn.neighbors
# - Print the R-squared score using knn.score(X_test_scaled, y_test)

from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

# print("Diabetes",diabetes)


X_train, X_test, y_train, y_test = train_test_split(X ,y ,test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


knn = KNeighborsRegressor(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test)

R_Squared = knn.score(X_test_scaled, y_test)

print(f"R Squared {R_Squared}")
