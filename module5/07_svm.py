# ============================================
# MODULE 5: Support Vector Machines (SVM)
# Book: Ch 11 (Classifiers)
# ============================================

# --- WHAT IS SVM? ---
# SVM finds the BEST line (or boundary) that separates two classes.
# "Best" means: the line with the BIGGEST gap (margin) between classes.
#
# Think of it like this:
# - You have red dots and blue dots on a table
# - You want to draw a line between them
# - Many lines could work, but SVM picks the one with the WIDEST road between the groups

# --- VISUAL ---
"""
    Many lines can separate the data, but SVM picks the BEST one:

    BAD (narrow margin):              GOOD (wide margin - SVM picks this):

    o o o  |x x x                     o o o   |         |  x x x
    o o    |  x x                     o o     | MARGIN  |    x x
    o o o  |x x                       o o o   |         |  x x
           |                                  |         |
        (line)                          (support vectors on the edges)

    The dots closest to the line are called SUPPORT VECTORS.
    They define where the margin boundaries are.

    KEY IDEA:
    - Linear SVM  → draws a straight line
    - Kernel SVM  → bends the boundary into curves (for non-linear data)

    Common kernels:
    - 'linear'  → straight line       (simple data)
    - 'rbf'     → curved boundary     (complex data, most popular)
    - 'poly'    → polynomial curves   (in between)
"""

# --- EXAMPLE: SVM on Iris Dataset ---
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data (only 2 classes to keep it simple)
iris = load_iris()
X = iris.data[iris.target != 2]   # only class 0 and 1
y = iris.target[iris.target != 2]

# SVM works best with SCALED data (remember StandardScaler?)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train SVM with different kernels
for kernel in ['linear', 'rbf', 'poly']:
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_train, y_train)
    acc = accuracy_score(y_test, svm.predict(X_test))
    print(f"SVM (kernel={kernel}): Accuracy = {acc:.3f}")

# How many support vectors?
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train)
print(f"\nSupport vectors per class: {svm_linear.n_support_}")
print(f"Total support vectors: {sum(svm_linear.n_support_)} out of {len(X_train)} training samples")


# ============================================
# PRACTICE: Your exercises go below
# ============================================

# --- Problem 1: SVM with Scaling vs Without ---
# Using the wine dataset:
#   1. Load wine data (load_wine), split 70/30
#   2. Train SVM (kernel='rbf') on RAW data (no scaling) → print test accuracy
#   3. Train SVM (kernel='rbf') on SCALED data (StandardScaler) → print test accuracy
#   4. Print which is better
#
# This shows WHY scaling matters for SVM!
#
# Hint: from sklearn.datasets import load_wine

# 1 load wine data
from sklearn.datasets import load_wine

data = load_wine()

X = data.data
y = data.target

X_train , X_test, y_train , y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2 Train SVM

svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)

acc_without = accuracy_score(y_test, svm.predict(X_test))
print("Acc :",acc_without)

# 3 Train SVM

# SVM works best with SCALED data (remember StandardScaler?)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on train, transform train
X_test_scaled = scaler.transform(X_test)          # only transform test (no fit!)

svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)

acc_with = accuracy_score(y_test, svm.predict(X_test_scaled))
print("Acc with scaling:", acc_with)

# 4 Which is better?
if acc_with > acc_without:
    print(f"Scaled is better! ({acc_with:.3f} vs {acc_without:.3f})")
else:
    print(f"Raw is better! ({acc_without:.3f} vs {acc_with:.3f})")


# --- Problem 2: The C Parameter (Regularization) ---
# SVM has a key parameter: C (default=1.0)
#
#   Small C (e.g. 0.01) → wide margin, allows some misclassifications (simpler, less overfit)
#   Large C (e.g. 100)  → narrow margin, tries to classify EVERYTHING right (complex, may overfit)
#
#   Think of C as "how much do I punish mistakes?"
#
#        Small C (forgiving):          Large C (strict):
#        o o  x  |       x x           o o    |  x    x x
#        o    o  |  x    x             o   o  |    x  x
#        o o     |    x x              o o    |  x  x x
#           (wider margin,               (tight margin,
#            tolerates errors)            fits every point)
#
# Using the wine dataset with SCALED data:
#   1. Train SVM (kernel='rbf') with C=0.01, C=1, and C=100
#   2. Print train accuracy AND test accuracy for each
#   3. Print which C value has the best test accuracy


results = []
for c in [0.01, 1, 100]:
    svm = SVC(kernel='rbf', C=c)
    svm.fit(X_train_scaled, y_train)
    train_acc = accuracy_score(y_train, svm.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, svm.predict(X_test_scaled))
    print(f"C={c:5} → Train: {train_acc:.3f}, Test: {test_acc:.3f}")
    results.append({"C": c, "test_acc": test_acc})

# 3
best = max(results, key=lambda x: x['test_acc'])
print(f"Best C={best['C']} (test accuracy: {best['test_acc']:.3f})")


# --- Problem 3: Count Support Vectors ---
# Support vectors are the FEW points that actually define the decision boundary.
# More support vectors = model relies on more points = more complex boundary.
#
# Using the wine dataset with SCALED data:
#   1. Train SVM (kernel='rbf') with C=0.01, C=1, and C=100
#   2. After fitting, access svm.n_support_ to get support vector counts per class
#   3. Print total support vectors and total training samples for each C
#   4. Print which C uses the FEWEST support vectors (simplest model)
#
# What to notice: small C → many support vectors, large C → fewer


# 1
results = []

for c in [0.01, 1 , 100]:
    svm = SVC(kernel='rbf', C=c, random_state=42)
    svm.fit(X_train_scaled, y_train)

    

    print(f"C={c:5} → Support vectors per class: {svm.n_support_}, Total: {sum(svm.n_support_)} / {len(X_train)}")
    results.append({
        "C": c,
        "total_sv": sum(svm.n_support_)
    })

best = min(results, key=lambda x: x['total_sv'])
print(f"Fewest support vectors: C={best['C']} ({best['total_sv']} / {len(X_train)})")

