# Module 5, Lesson 1: Data Pipeline with Scikit-learn
# Book Reference: Ch 8 (Training & Testing), Ch 12 (Managing Data)
# Topics: train_test_split, scaling, encoding, cross-validation

# ============================================================
# PROBLEM 1: Train/Test Split
# ============================================================
# You have a dataset of 20 exam scores (X) and pass/fail labels (y).
# Split it into 80% training / 20% test using train_test_split.
# Use random_state=42 for reproducibility.
# Print the number of training and test samples.
#
# Expected output:
#   Training samples: 16
#   Test samples: 4
# ============================================================

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = np.random.RandomState(0).rand(20, 3)  # 20 samples, 3 features
y = np.array([1,0,1,1,0,0,1,0,1,1,0,1,0,0,1,1,0,1,0,1])

# Your code below:

X_train, X_test, y_train, y_test = train_test_split(
    X, y , test_size=0.2,random_state=42
)

# print(len(X_train))
# print(len(X_test))

# ============================================================
# PROBLEM 2: Feature Scaling
# ============================================================
# Using X_train and X_test from Problem 1:
# 1. Create a StandardScaler
# 2. fit_transform on X_train
# 3. transform only on X_test (no fitting!)
# 4. Print the mean of X_train_scaled (should be ~0)
#
# Expected output:
#   X_train_scaled mean: [-0.0, -0.0, 0.0]  (very close to zero)
# ============================================================

# Your code below:

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.mean(axis=0))

# ============================================================
# PROBLEM 3: Cross-Validation
# ============================================================
# Instead of trusting ONE train/test split, test on 5 different splits.
# 1. Create a KNeighborsClassifier(n_neighbors=3)
# 2. Run cross_val_score with cv=5 on the ORIGINAL X and y
# 3. Print each fold's score and the average
#
# Expected output (values may vary):
#   Fold scores: [0.75  0.5   0.5   0.75  0.75]
#   Average: 0.650 ± 0.122
# ============================================================
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Your code below:

model = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(model, X, y, cv=5)

print(f"Score : {scores}")