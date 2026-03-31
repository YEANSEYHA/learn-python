# ============================================
# MODULE 6: Activation Functions
# Book: Ch 13 (Neural Networks)
# ============================================

# --- WHAT ARE ACTIVATION FUNCTIONS? ---
# After computing z = wx + b, we pass it through an activation function.
# Without activation functions, a neural network is just linear regression —
# no matter how many layers, it can only learn straight lines.
# Activation functions add NON-LINEARITY, letting networks learn complex patterns.

# --- VISUAL ---
"""
    The 4 main activation functions:

    STEP (Perceptron)       SIGMOID                  TANH                    ReLU
    output                  output                   output                  output
    1 |----               1 |   ___---               1 |  ___---             |    /
      |                    |  /                       |  /                   |   /
    0 |    ----           0.5|/                      0| /                   0|__/
      +------→ z           0|                        -1|                     +------→ z
                            +------→ z                 +------→ z
    0 or 1 only          0 to 1                     -1 to 1               0 or z (no negatives)

    USE CASES:
    - Sigmoid  → output layer for binary classification (0 to 1 = probability)
    - Tanh     → hidden layers (centered at 0, stronger gradients than sigmoid)
    - ReLU     → hidden layers in deep networks (most popular, fast to compute)
    - Softmax  → output layer for multi-class classification (probabilities that sum to 1)
"""

import numpy as np

# --- SIGMOID ---
# Squashes any value to (0, 1) — useful for probabilities
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# --- TANH ---
# Squashes any value to (-1, 1) — zero-centered
def tanh(z):
    return np.tanh(z)

# --- ReLU ---
# Keeps positives, kills negatives — simple and fast
def relu(z):
    return np.maximum(0, z)

# --- SOFTMAX ---
# Converts a vector of scores to probabilities that sum to 1
def softmax(z):
    exp_z = np.exp(z - np.max(z))   # subtract max for numerical stability
    return exp_z / exp_z.sum()

# --- EXAMPLE: Apply all activations to a range of z values ---
z_values = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])

print("z        sigmoid   tanh      relu")
print("-" * 40)
for z in z_values:
    print(f"{z:6.1f}  {sigmoid(z):8.4f}  {tanh(z):8.4f}  {relu(z):8.4f}")

# Softmax on a vector (e.g. 3-class scores)
scores = np.array([2.0, 1.0, 0.5])
probs  = softmax(scores)
print(f"\nSoftmax input:  {scores}")
print(f"Softmax output: {np.round(probs, 4)}")
print(f"Sum of probs:   {probs.sum():.4f}  ← always 1.0")


# ============================================
# PRACTICE: Your exercises go below
# ============================================

# --- Problem 1: ReLU from scratch ---
# ReLU = max(0, z). It zeroes out all negatives, keeps positives unchanged.
#
# Given:
#   z = [-2.0, -0.5, 0.0, 1.5, 3.0]
#
# Your task:
#   1. Apply ReLU using np.maximum(0, z)
#   2. Print the result
#   3. Count how many values are DEAD (equal to 0) and print the count
#
# "Dead neurons" is a real problem in deep learning — neurons stuck at 0 stop learning


z = [-2.0, -0.5, 0.0, 1.5, 3.0]
from collections import Counter

#1
def relu(z):
    return np.maximum(0, z)
print(relu(z))

counts = Counter(relu(z))
print(f"Counts {counts}")


# --- Problem 2: Sigmoid as probability ---
# Sigmoid output is always between 0 and 1 — perfect for probabilities.
# In binary classification:
#   - output >= 0.5 → predict class 1
#   - output <  0.5 → predict class 0
#
# Given these raw model scores (z values):
#   z = [-2.5, -0.5, 0.0, 0.8, 2.1]
#
# Your task:
#   1. Apply sigmoid to all z values
#   2. Convert to class predictions (1 if prob >= 0.5 else 0)
#   3. Print each z, its probability, and its predicted class

z = np.array([-2.5, -0.5, 0.0, 0.8, 2.1])

#
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print()
data = sigmoid(z)


probs = sigmoid(z)
predictions = [1 if p >= 0.5 else 0 for p in probs]

print(f"{'z':>6}  {'prob':>8}  {'class':>6}")
print("-" * 26)
for i in range(len(z)):
    print(f"{z[i]:6.1f}  {probs[i]:8.4f}  {predictions[i]:6}")


# --- Problem 3: Softmax for multi-class ---
# Softmax converts raw scores into probabilities for MULTIPLE classes.
# All outputs are between 0 and 1, and they always sum to 1.0.
# The class with the HIGHEST probability is the prediction.
#
# A model outputs these raw scores for 4 classes:
#   scores = [1.2, 3.1, 0.5, 2.0]
#             cat  dog  bird fish
#
# Your task:
#   1. Apply softmax to the scores (already defined at the top of the file)
#   2. Print each class name with its probability
#   3. Print the predicted class (highest probability)
#
# Hint: np.argmax(probs) gives the index of the highest value

scores = np.array([1.2, 3.1, 0.5, 2.0])

# 1. Apply Softmax
print(f"SoftMax {softmax(scores)}")

# 2 
classes = ['cat', 'dog', 'bird', 'fish']
probs = softmax(scores)
for name, prob in zip(classes, probs):
    print(f"{name}: {prob:.4f}")

best = np.argmax(probs)
print(f"\nPredicted class: {classes[best]} ({probs[best]:.4f})")
