"""
LOGISTIC REGRESSION
====================
Linear regression predicts NUMBERS (price, temperature).
Logistic regression predicts CLASSES (yes/no, spam/not spam, cat/dog).

WHY IT MATTERS FOR ML:
- Most common classification algorithm in industry
- The OUTPUT LAYER of neural networks IS logistic regression
- Every ML interview asks about it

FROM LINEAR TO LOGISTIC:
- Linear regression:    y = mx + b           → outputs any number (-∞ to +∞)
- Logistic regression:  y = sigmoid(mx + b)  → outputs probability (0 to 1)

    The SIGMOID function squashes any number into 0-1 range:

         sigmoid(z) = 1 / (1 + e^(-z))

    Input z:   -10    -2     0     2     10
    Output:    0.00   0.12  0.50  0.88  1.00


VISUAL — How sigmoid works:
============================

    1.0 |                  ___________
        |                /
    0.5 |- - - - - - - /- - - - - - - -   ← decision boundary (0.5)
        |            /
    0.0 |___________/
        └──────────┬──────────────────
                   0
                 input (z = mx + b)

    If sigmoid output >= 0.5  →  predict class 1 (yes, spam, cat)
    If sigmoid output <  0.5  →  predict class 0 (no, not spam, dog)


VISUAL — Linear vs Logistic:
==============================

    Linear Regression:           Logistic Regression:
    y                            y
    |       /                    1.0|           ●●●●●
    |     /                         |         /
    |   / ●                     0.5|- - - /- - - - -
    | / ●                           |   /
    |/●___________x              0.0|●●●●●___________x
    predicts a NUMBER              predicts a PROBABILITY


KEY FORMULAS:
- Step 1:  z = w·x + b              (linear combination, same as linear regression)
- Step 2:  p = sigmoid(z)           (squeeze into 0-1)
- Step 3:  class = 1 if p >= 0.5, else 0

- Loss function: Log Loss (Binary Cross-Entropy)
    L = -(1/n) * sum[ y*log(p) + (1-y)*log(1-p) ]
    (NOT MSE — because sigmoid + MSE creates bad gradients)

SCIKIT-LEARN USAGE:
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)        # class labels (0 or 1)
    probabilities = model.predict_proba(X_test) # probabilities


EXAMPLE 1 — The sigmoid function:
===================================
"""
import numpy as np
import matplotlib.pyplot as plt

# The sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
plt.figure(figsize=(8, 4))
plt.plot(z, sigmoid(z), 'b-', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', label='decision boundary (0.5)')
plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
plt.xlabel('z (input)', fontsize=12)
plt.ylabel('sigmoid(z)', fontsize=12)
plt.title('Sigmoid Function', fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("sigmoid.png", dpi=100)
# plt.show()

print("sigmoid(-10) =", sigmoid(-10))   # ≈ 0.00
print("sigmoid(0)   =", sigmoid(0))     # = 0.50
print("sigmoid(10)  =", sigmoid(10))    # ≈ 1.00


"""
EXAMPLE 2 — Logistic Regression with Scikit-learn:
====================================================
Task: Predict if a student PASSES (1) or FAILS (0) based on hours studied.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Data: hours studied → pass (1) or fail (0)
np.random.seed(42)
hours = np.concatenate([
    np.random.normal(3, 1, 30),    # students who failed (avg 3 hrs)
    np.random.normal(7, 1, 30),    # students who passed (avg 7 hrs)
])
passed = np.array([0]*30 + [1]*30)

X = hours.reshape(-1, 1)    # sklearn needs 2D input
y = passed

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Results
print(f"\nWeight (w):  {model.coef_[0][0]:.4f}")
print(f"Bias (b):    {model.intercept_[0]:.4f}")
print(f"Accuracy:    {model.score(X_test, y_test):.2%}")

# Predict for a new student
new_student = np.array([[5]])
prob = model.predict_proba(new_student)
print(f"\nStudent who studied 5 hours:")
print(f"  Probability of FAIL: {prob[0][0]:.2%}")
print(f"  Probability of PASS: {prob[0][1]:.2%}")
print(f"  Prediction: {'PASS' if model.predict(new_student)[0] == 1 else 'FAIL'}")

# Visualize decision boundary
plt.figure(figsize=(8, 4))
X_range = np.linspace(0, 10, 200).reshape(-1, 1)
probs = model.predict_proba(X_range)[:, 1]

plt.scatter(X_test[y_test == 0], y_test[y_test == 0], c='red', label='Fail', s=60, zorder=3)
plt.scatter(X_test[y_test == 1], y_test[y_test == 1], c='green', label='Pass', s=60, zorder=3)
plt.plot(X_range, probs, 'b-', linewidth=2, label='P(pass)')
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Hours Studied', fontsize=12)
plt.ylabel('Probability of Passing', fontsize=12)
plt.title('Logistic Regression — Pass/Fail Prediction', fontsize=14)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("logistic_regression_example.png", dpi=100)
# plt.show()


# ============================================================
# EXERCISE 1: Sigmoid by hand
# ============================================================
# A logistic regression model has: w = 2.0, b = -10.0
#
# For a student who studied 6 hours:
#   (a) Calculate z = w * hours + b
#   (b) Calculate sigmoid(z) using the formula: 1 / (1 + e^(-z))
#   (c) What class does the model predict? (>= 0.5 → pass)
#
# Use numpy, NOT sklearn. Print z, probability, and prediction.

# --- your code here ---

w = 2.0 
b = -10.0
# (a)
hours = 6
z = w * hours + b
print(f"Z : {z}")

# (b)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print(sigmoid(z))

# c 

prob = sigmoid(z)

prediction = 1 if prob>=0.5 else 0

print(f"Probability: {prob:.2f}, Prediction: {'PASS' if prediction == 1 else 'FAIL'}")


# ============================================================
# EXERCISE 2: Tumor classification (benign vs malignant)
# ============================================================
# The breast cancer dataset has 30 features (tumor measurements)
# and 2 classes: malignant (0) or benign (1).
#
# Your task:
#   (a) Load the dataset using sklearn.datasets.load_breast_cancer()
#   (b) Split into train/test (test_size=0.2, random_state=42)
#   (c) Train a LogisticRegression(max_iter=10000) model
#   (d) Print the accuracy on the test set using model.score()
#
# hint: data = load_breast_cancer()
#       X = data.data     (features)
#       y = data.target   (labels)

# --- your code here ---

#(a)
from sklearn.datasets import load_breast_cancer

# load dataset
data = load_breast_cancer()
# print(data)
X = data.data
y = data.target

# (b) Split data
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# (C)
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# D

print(f" Accuracy : {model.score(X_test,y_test):.2%}")


# ============================================================
# EXERCISE 3: Confusion matrix + predict_proba
# ============================================================
# Using your trained breast cancer model from Exercise 2:
#
#   (a) Get predicted probabilities: model.predict_proba(X_test)
#       Print the first 5 rows — each row shows [P(malignant), P(benign)]
#
#   (b) Build a confusion matrix using sklearn:
#       from sklearn.metrics import confusion_matrix
#       y_pred = model.predict(X_test)
#       cm = confusion_matrix(y_test, y_pred)
#       Print the matrix.
#
#   (c) From the confusion matrix, calculate and print:
#       - How many malignant tumors were correctly identified?
#       - How many benign tumors were misclassified as malignant?
#
#   hint: confusion matrix layout:
#                    Predicted
#                   Mal   Ben
#   Actual  Mal  [  TN  | FP  ]
#           Ben  [  FN  | TP  ]

# --- your code here ---
# (a)

probability = model.predict_proba(X_test)

print(f"Probability {probability[:5]}")

#b
from sklearn.metrics import confusion_matrix
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(f" confusion matrix : {cm}")

#c 
print(" malignant tumors were correctly : 70")
print("many benign tumors were misclassified as malignan 1")


