# ============================================
# MODULE 6: Loss Functions
# Book: Ch 14 (Measuring Error)
# ============================================

# --- WHAT IS A LOSS FUNCTION? ---
# A loss function measures HOW WRONG the model's predictions are.
# The goal of training is to MINIMIZE the loss.
# Different problems use different loss functions:
#   - Regression   → MSE (Mean Squared Error)
#   - Binary class → Binary Cross-Entropy
#   - Multi-class  → Categorical Cross-Entropy

# --- VISUAL ---
"""
    MSE (regression):                  Cross-Entropy (classification):

    loss                               loss
    |  *                               |
    |    *                             |*
    |      *   *                       |  *
    |          *  *                    |    *  *
    |                *  *              |          *   *   *
    +──────────────────→ prediction    +──────────────────→ probability

    MSE penalizes large errors more    Cross-entropy penalizes CONFIDENT wrong
    (squared means big gaps = big loss) predictions very heavily

    KEY FORMULAS:
    ┌─────────────────────────────────────────────────────┐
    │  MSE        = mean( (y_pred - y_true)² )            │
    │  Binary CE  = -mean( y*log(p) + (1-y)*log(1-p) )   │
    │  Categorical CE = -mean( Σ y_true * log(y_pred) )  │
    └─────────────────────────────────────────────────────┘

    Where p = predicted probability, y = true label (0 or 1)
"""

import numpy as np

# --- MSE Loss ---
def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

# --- Binary Cross-Entropy Loss ---
def binary_cross_entropy(y_true, y_pred):
    # clip to avoid log(0) which is -infinity
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# --- Categorical Cross-Entropy Loss ---
def categorical_cross_entropy(y_true, y_pred):
    # y_true is one-hot, y_pred is softmax probabilities
    y_pred = np.clip(y_pred, 1e-7, 1.0)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# --- EXAMPLE: MSE for regression ---
y_true = np.array([3.0, 5.0, 2.5, 7.0])
y_pred = np.array([2.8, 5.2, 2.0, 6.5])
print(f"MSE Loss: {mse(y_true, y_pred):.4f}")

# --- EXAMPLE: Binary cross-entropy for classification ---
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([0.9, 0.1, 0.8, 0.6, 0.3])   # predicted probabilities
print(f"Binary CE Loss: {binary_cross_entropy(y_true, y_pred):.4f}")

# A WRONG confident prediction → high loss
y_pred_bad = np.array([0.1, 0.9, 0.2, 0.3, 0.8])
print(f"Binary CE (wrong): {binary_cross_entropy(y_true, y_pred_bad):.4f}")

# --- EXAMPLE: Categorical cross-entropy ---
# 3 samples, 3 classes (one-hot encoded)
y_true = np.array([[1, 0, 0],   # true class = 0
                   [0, 1, 0],   # true class = 1
                   [0, 0, 1]])  # true class = 2

y_pred = np.array([[0.8, 0.1, 0.1],   # confident and correct
                   [0.1, 0.7, 0.2],   # confident and correct
                   [0.2, 0.2, 0.6]])  # confident and correct

print(f"Categorical CE (good): {categorical_cross_entropy(y_true, y_pred):.4f}")

y_pred_bad = np.array([[0.1, 0.8, 0.1],   # wrong!
                       [0.7, 0.1, 0.2],   # wrong!
                       [0.2, 0.6, 0.2]])  # wrong!

print(f"Categorical CE (bad):  {categorical_cross_entropy(y_true, y_pred_bad):.4f}")


# ============================================
# PRACTICE: Your exercises go below
# ============================================

# --- Problem 1: MSE vs good/bad predictions ---
# MSE loss increases when predictions are far from the truth.
#
# y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
#
# Compute and print MSE for:
#   - good_pred  = [1.1, 1.9, 3.2, 3.8, 5.1]   (close to truth)
#   - bad_pred   = [2.0, 4.0, 1.0, 6.0, 3.0]   (far from truth)
#
# Print both losses and which prediction is better


y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

good_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
bad_pred = np.array([2.0, 4.0, 1.0, 6.0, 3.0])


print(f"MSE Good: {mse(y_true, good_pred):.4f}")
print(f"MSE Bad:  {mse(y_true, bad_pred):.4f}")
print("Better: good_pred" if mse(y_true, good_pred) < mse(y_true, bad_pred) else "Better: bad_pred")


# --- Problem 2: Binary Cross-Entropy ---
# Cross-entropy heavily penalizes CONFIDENT wrong predictions.
#
# y_true = [1, 1, 0, 0, 1]
#
# Compute and print binary cross-entropy for:
#   - confident_correct = [0.95, 0.88, 0.05, 0.10, 0.92]  (sure and right)
#   - confident_wrong   = [0.05, 0.10, 0.95, 0.88, 0.08]  (sure but wrong)
#   - unsure            = [0.55, 0.60, 0.45, 0.40, 0.58]  (not confident)
#
# Print all 3 losses — notice confident_wrong is the highest

y_true              = np.array([1, 1, 0, 0, 1])
confident_correct   = np.array([0.95, 0.88, 0.05, 0.10, 0.92])
confident_wrong     = np.array([0.05, 0.10, 0.95, 0.88, 0.08])
unsure              = np.array([0.55, 0.60, 0.45, 0.40, 0.58])

# your solution here:

print(f"BCE Confident correct: {binary_cross_entropy(y_true, confident_correct):.4f}")
print(f"BCE Confident wrong:  {binary_cross_entropy(y_true, confident_wrong):.4f}")
print(f"BCE Unsure:           {binary_cross_entropy(y_true, unsure):.4f}")


# --- Problem 3: Categorical Cross-Entropy ---
# Used when there are 3+ classes. y_true is one-hot encoded.
# The model's output is softmax probabilities (sum = 1 per sample).
#
# 4 samples, 3 classes:
y_true = np.array([[1,0,0],   # sample 0: cat
                   [0,1,0],   # sample 1: dog
                   [0,0,1],   # sample 2: bird
                   [1,0,0]])  # sample 3: cat

y_pred = np.array([[0.7, 0.2, 0.1],   # sample 0: predicts cat ✓
                   [0.1, 0.8, 0.1],   # sample 1: predicts dog ✓
                   [0.3, 0.3, 0.4],   # sample 2: predicts bird ✓ (barely)
                   [0.1, 0.1, 0.8]])  # sample 3: predicts bird ✗ (wrong!)

y_pred_fixed = np.array([[0.7, 0.2, 0.1],
                         [0.1, 0.8, 0.1],
                         [0.3, 0.3, 0.4],
                         [0.8, 0.1, 0.1]])  # sample 3 fixed → cat ✓

# Your task:
#   1. Compute and print categorical CE for y_pred
#   2. Compute and print categorical CE for y_pred_fixed
#   3. Print which has lower loss

# 1. 
print(f" Compute CE : {categorical_cross_entropy(y_true, y_pred):.4f}")

print(f" Compute CE  fixed: {categorical_cross_entropy(y_true, y_pred_fixed):.4f}")


print("y_pred_fixed is better" if categorical_cross_entropy(y_true, y_pred_fixed) < categorical_cross_entropy(y_true, y_pred) else "y_pred is better")


