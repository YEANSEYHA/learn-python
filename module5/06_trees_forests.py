# ============================================
# MODULE 5: Decision Trees & Random Forests
# Book: Ch 10-11 (Ensemble Methods)
# ============================================

# --- EXAMPLE: Decision Tree on Iris Dataset ---
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
iris = load_iris()
X, y = iris.data, iris.target  # 4 features, 3 classes

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Predict
y_pred = tree.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred):.3f}")


# See what the tree learned — feature importances
# Higher = that feature matters more for splitting
for name, importance in zip(iris.feature_names, tree.feature_importances_):
    print(f"  {name}: {importance:.3f}")

# --- Gini Impurity by hand ---
# A group with 30 class-A and 10 class-B
p_a = 30 / 40
p_b = 10 / 40
gini = 1 - (p_a**2 + p_b**2)
print(f"\nGini of [30A, 10B]: {gini:.4f}")  # not pure

# A perfectly pure group (all class-A)
gini_pure = 1 - (1.0**2)
print(f"Gini of [40A, 0B]:  {gini_pure:.4f}")  # perfectly pure


# ============================================
# PRACTICE: Your exercises go below
# ============================================

# --- Problem 1: Calculate Gini Impurity ---
# Dataset: 100 samples (60 cats, 40 dogs)
#
# Split A → Left: [45 cats, 5 dogs], Right: [15 cats, 35 dogs]
# Split B → Left: [30 cats, 20 dogs], Right: [30 cats, 20 dogs]
#
# For each split:
#   1. Compute gini_left and gini_right  (Gini = 1 - p_cat² - p_dog²)
#   2. Compute weighted Gini = (n_left/100)*gini_left + (n_right/100)*gini_right
#   3. Print which split is better (lower weighted Gini = better)

def gini(p_a, p_b):
    return 1 - (p_a**2 + p_b**2)

# Split A → Left: [45 cats, 5 dogs], Right: [15 cats, 35 dogs]
gini_left_a = gini(45/50, 5/50)
gini_right_a = gini(15/50, 35/50)
weighted_a = (50/100) * gini_left_a + (50/100) * gini_right_a

print(f"Split A - Left Gini: {gini_left_a:.4f}, Right Gini: {gini_right_a:.4f}")
print(f"Split A - Weighted Gini: {weighted_a:.4f}")

# Split B → Left: [30 cats, 20 dogs], Right: [30 cats, 20 dogs]
gini_left_b = gini(30/50, 20/50)
gini_right_b = gini(30/50, 20/50)
weighted_b = (50/100) * gini_left_b + (50/100) * gini_right_b

print(f"Split B - Left Gini: {gini_left_b:.4f}, Right Gini: {gini_right_b:.4f}")
print(f"Split B - Weighted Gini: {weighted_b:.4f}")

# Which is better?
if weighted_a < weighted_b:
    print("Split A is better (lower Gini)")
else:
    print("Split B is better (lower Gini)")


# --- Problem 2: Overfitting with Decision Trees ---
# A deep tree memorizes the training data (overfits).
# A shallow tree generalizes better.
#
# Using the wine dataset from sklearn:
#   1. Load data with load_wine(), split 70/30
#   2. Train TWO trees: one with max_depth=2, one with max_depth=None (unlimited)
#   3. Print train accuracy AND test accuracy for both
#   4. Print which tree generalizes better (smaller train-test gap)
#
# Hint: from sklearn.datasets import load_wine

from sklearn.datasets import load_wine

# load data
data = load_wine()
# print("data :",data)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train
tresses = [{
    "name" : "tree1",
    "max_depth": 2
},
{
    "name" : "tree2",
    "max_depth": None
}]

results = []

for t in tresses:
    tree = DecisionTreeClassifier(max_depth=t['max_depth'], random_state=42)
    tree.fit(X_train, y_train)

    # Predict on BOTH train and test
    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc = accuracy_score(y_test, tree.predict(X_test))
    gap = train_acc - test_acc

    print(f"{t['name']} (max_depth={t['max_depth']}): Train={train_acc:.3f}, Test={test_acc:.3f}, Gap={gap:.3f}")
    results.append({
        "tree": t['name'],
        "train_acc": train_acc,
        "test_acc": test_acc,
        "gap": gap
    })

best = min(results, key=lambda x: x['gap'])
print(f"\nBest generalization: {best['tree']} (smallest gap: {best['gap']:.3f})")


# --- Problem 3: Random Forest vs Single Decision Tree ---
# Using the same wine dataset (X_train, X_test, y_train, y_test from above):
#   1. Train a single DecisionTreeClassifier (max_depth=None)
#   2. Train a RandomForestClassifier with n_estimators=100 (100 trees)
#   3. Print test accuracy for both
#   4. Print which model is better
#
# Hint: from sklearn.ensemble import RandomForestClassifier


# 1 . Train 
t1 = DecisionTreeClassifier(max_depth=None, random_state=42)
t1.fit(X_train, y_train)

test_acc = accuracy_score(y_test, t1.predict(X_test))

print("T1 Test Accuracy:",test_acc)

from sklearn.ensemble import RandomForestClassifier

#2  . Train 
t2 = RandomForestClassifier(max_depth=None, random_state=42, n_estimators=100)
t2.fit(X_train, y_train)

test_acc = accuracy_score(y_test, t2.predict(X_test))

print("T2 Test Accuracy:",test_acc)


print("T2 is Better than T1 ( 1.0 vs 0.96)")
