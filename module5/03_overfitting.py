# ============================================================
# MODULE 5.3 — OVERFITTING & UNDERFITTING
# Book Reference: Chapter 9 - Overfitting and Underfitting
# ============================================================

# ============================================================
# WHAT IS OVERFITTING & UNDERFITTING?
# ============================================================
#
# When training a model, we want it to learn GENERAL patterns
# from data — not memorize specific examples.
#
# UNDERFITTING: Model is too simple → misses real patterns
#   - High error on BOTH training and test data
#   - Like drawing a straight line through curved data
#
# OVERFITTING: Model is too complex → memorizes noise
#   - Low error on training, HIGH error on test data
#   - Like drawing a wiggly line that hits every training point
#
# JUST RIGHT (Good Fit): Model captures the trend, ignores noise
#   - Low error on both training and test data
#
# ============================================================
# VISUAL: The Bias-Variance Tradeoff
# ============================================================
#
#   Error
#    |
#    |  \                          ╱
#    |   \    Underfitting        ╱ Overfitting
#    |    \       zone          ╱    zone
#    |     \                  ╱
#    |      ╲    ┌──────┐   ╱   ← Validation Error
#    |       ╲───┤SWEET ├──╱
#    |        ╲  │ SPOT │╱
#    |         ──┴──────┘────────  ← Training Error
#    |
#    └──────────────────────────→  Model Complexity
#
#   KEY INSIGHT from Ch 9:
#   - Simple model = High Bias, Low Variance  (underfitting)
#   - Complex model = Low Bias, High Variance (overfitting)
#   - We want the sweet spot in between!
#
# ============================================================
# VISUAL: What the predictions look like
# ============================================================
#
#   True pattern: gentle curve
#   Data points:  scattered around the curve (noise)
#
#   UNDERFIT (too simple):       OVERFIT (too complex):
#         .  .                        .  .
#     ──────────── straight      ╱╲╱╲╱╲╱╲ wiggly
#       .    .   . line            .    .   hits every point
#
#   GOOD FIT:
#         .  .
#     ───╱──╲──── smooth curve
#       .    .   . captures trend
#
# ============================================================
# HOW TO DETECT: Training vs Validation Error
# ============================================================
#
#   Underfitting:  train_error = HIGH,  val_error = HIGH
#   Good fit:      train_error = LOW,   val_error = LOW
#   Overfitting:   train_error = LOW,   val_error = HIGH
#                                       ↑ big gap = overfitting!
#
# ============================================================


# ============================================================
# EXAMPLE: See overfitting & underfitting in action
# ============================================================
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create data with a true pattern: y = sin(x) + noise
np.random.seed(42)
X = np.sort(np.random.uniform(0, 6, 40)).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, 40)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Try 3 models with different complexity
degrees = [1, 4, 15]       # 1=underfit, 4=good, 15=overfit
labels = ["UNDERFIT (degree=1)", "GOOD FIT (degree=4)", "OVERFIT (degree=15)"]

print("=" * 55)
print("OVERFITTING vs UNDERFITTING DEMO")
print("True pattern: y = sin(x) + noise")
print("=" * 55)

for degree, label in zip(degrees, labels):
    # Polynomial regression with given degree
    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )
    model.fit(X_train, y_train)

    # Measure error on train and test
    train_err = mean_squared_error(y_train, model.predict(X_train))
    test_err = mean_squared_error(y_test, model.predict(X_test))
    gap = test_err - train_err

    print(f"\n{label}")
    print(f"  Train MSE: {train_err:.4f}")
    print(f"  Test MSE:  {test_err:.4f}")
    print(f"  Gap:       {gap:.4f}  {'<-- big gap = overfitting!' if gap > 0.5 else ''}")

# Output:
# =======================================================
# OVERFITTING vs UNDERFITTING DEMO
# True pattern: y = sin(x) + noise
# =======================================================
#
# UNDERFIT (degree=1)
#   Train MSE: 0.1757
#   Test MSE:  0.2640
#   Gap:       0.0883
#
# GOOD FIT (degree=4)
#   Train MSE: 0.0340
#   Test MSE:  0.0596
#   Gap:       0.0256
#
# OVERFIT (degree=15)
#   Train MSE: 0.0073
#   Test MSE:  7.2483
#   Gap:       7.2410  <-- big gap = overfitting!


# ============================================================
# KEY TAKEAWAYS:
# ============================================================
# 1. Underfit (degree=1): Both errors are HIGH → too simple
# 2. Good fit (degree=4): Both errors are LOW → captures pattern
# 3. Overfit (degree=15): Train is tiny, test is HUGE → memorized noise
#
# The GAP between train and test error is the overfitting signal!
# Ch 9 calls this: "when validation error starts rising, stop training"
# (this is called EARLY STOPPING)
# ============================================================


# ============================================================
# PRACTICE PROBLEMS
# ============================================================

# PROBLEM 1:
#
# You have train and test MSE values for 3 models.
# Write code to detect which model is: underfit, good fit, or overfit.
#
# Rules:
#   - If train_mse > 0.3           → "underfit"
#   - If test_mse - train_mse > 0.5 → "overfit"
#   - Otherwise                     → "good fit"
#
# Given data:
# models = {
#     "Model A": {"train_mse": 0.45, "test_mse": 0.50},
#     "Model B": {"train_mse": 0.05, "test_mse": 0.08},
#     "Model C": {"train_mse": 0.02, "test_mse": 1.85},
# }
#
# Loop through models and print: "Model A: underfit"
#
# YOUR CODE BELOW:

models = {
    "Model A": {"train_mse": 0.45, "test_mse": 0.50},
    "Model B": {"train_mse": 0.05, "test_mse": 0.08},
    "Model C": {"train_mse": 0.02, "test_mse": 1.85},
}



for key ,value in models.items():
    print(key, value)
    if(( value['test_mse'] - value['train_mse']) > 0.5 ):
        print(f"{key} overfit")
    elif(value['train_mse'] > 0.3):
        print(f"{key} underfit")
    else:
        print("Good Fit")


# PROBLEM 2:
#
# Given these train/test accuracy pairs for 4 models,
# find and print which model has the WORST overfitting
# (biggest gap between train and test accuracy).
#
# results = {
#     "Model A": (0.99, 0.62),   # (train_acc, test_acc)
#     "Model B": (0.85, 0.83),
#     "Model C": (1.00, 0.55),
#     "Model D": (0.92, 0.88),
# }
#
# Expected output: "Worst overfitting: Model C (gap: 0.45)"
#
# YOUR CODE BELOW:

results = {
    "Model A": (0.99, 0.62),   # (train_acc, test_acc)
    "Model B": (0.85, 0.83),
    "Model C": (1.00, 0.55),
    "Model D": (0.92, 0.88),
}

worst_model = []

for key, value in results.items():
    print(key, value)
    gap = value[0] - value[1]
    result = {
        "Model" : key,
        "gap": gap
    }
    worst_model.append(result)

print(worst_model)

worst = max(worst_model,key = lambda x:x['gap'])

print(f"THe worst model is {worst}")


# PROBLEM 3:
#
# A model was trained at different epochs. Given the errors below,
# find the best epoch to STOP training (early stopping).
#
# Rule: Stop at the epoch where validation error is LOWEST
#       (before it starts rising = overfitting begins)
#
# epochs =     [1,    2,    3,    4,    5,    6,    7,    8]
# train_err =  [0.90, 0.60, 0.40, 0.25, 0.15, 0.08, 0.04, 0.02]
# val_err =    [0.88, 0.58, 0.38, 0.30, 0.28, 0.35, 0.50, 0.72]
#
# Print: "Stop at epoch X (val_error: Y)"
#
# Hint: np.argmin() returns the INDEX of the smallest value
#
# YOUR CODE BELOW:

epochs =     [1,    2,    3,    4,    5,    6,    7,    8]
train_err =  [0.90, 0.60, 0.40, 0.25, 0.15, 0.08, 0.04, 0.02]
val_err =    [0.88, 0.58, 0.38, 0.30, 0.28, 0.35, 0.50, 0.72]

smallest_index = np.argmin(val_err)
print("Smallest Index",smallest_index)

print(f"Stop at epoch {smallest_index+1} (val_error: {val_err[smallest_index]})")

