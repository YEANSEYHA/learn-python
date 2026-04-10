# ============================================
# Full Neural Network — Step-by-Step Practice
# Build it piece by piece until it's automatic!
# ============================================

import numpy as np
np.random.seed(42)

# ── DATASET (same as before) ──
X = np.array([
    [0.1, 0.2],   # class 0
    [0.2, 0.1],   # class 0
    [0.3, 0.3],   # class 0
    [0.8, 0.7],   # class 1
    [0.9, 0.8],   # class 1
    [0.7, 0.9],   # class 1
])
y = np.array([0, 0, 0, 1, 1, 1])

# ── HELPER FUNCTIONS (given) ──
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)


# ============================================
# STEP 1: Initialize weights
# ============================================
"""
Create weights and biases for a network: 2 inputs → 3 hidden → 1 output

You need 4 things:
  W1 = random (2, 3) * 0.5
  b1 = zeros (3,)
  W2 = random (3, 1) * 0.5
  b2 = zeros (1,)

Print all 4 shapes.
"""
W1 = np.random.randn(2, 3) * 0.5
b1 = np.zeros(3)
W2 = np.random.randn(3, 1) * 0.5
b2 = np.zeros(1)

print(f"W1 shape: {W1.shape}")
print(f"b1 shape: {b1.shape}")
print(f"W2 shape: {W2.shape}")
print(f"b2 shape: {b2.shape}")


# ============================================
# STEP 2: Forward pass (1 sample)
# ============================================
"""
Take the first sample: x = X[0]

Compute:
  z1 = x @ W1 + b1          (hidden weighted sum)
  a1 = relu(z1)              (hidden activation)
  z2 = a1 @ W2 + b2          (output weighted sum)
  pred = sigmoid(z2)          (output probability)

Print z1, a1, and pred.
"""
x = X[0]
z1 = x @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
pred = sigmoid(z2)

print(f"z1: {np.round(z1, 4)}")
print(f"a1: {np.round(a1, 4)}")
print(f"pred: {pred[0]:.4f}  (true label: {y[0]})")


# ============================================
# STEP 3: Compute loss (1 sample)
# ============================================
"""
Using pred and label from Step 2, compute binary cross-entropy loss:

  label = y[0]
  loss = -(label * np.log(pred + 1e-8) + (1 - label) * np.log(1 - pred + 1e-8))

Print the loss value.

(The 1e-8 prevents log(0) which would be -infinity)
"""

label = y[0]
loss = -(label * np.log(pred + 1e-8) + (1 - label) * np.log(1 - pred + 1e-8))

print(f"Label: {label}")
print(f"Loss:  {loss[0]:.4f}")


# ============================================
# STEP 4: Backprop (compute gradients)
# ============================================
"""
Using pred, label, a1, z1, x, W2 from above, compute all gradients:

  dz2 = pred - label                                  # output error
  dW2 = a1.reshape(-1, 1) * dz2                       # gradient for W2
  db2 = dz2                                            # gradient for b2
  da1 = (dz2 * W2.T).flatten()                        # error back to hidden
  dz1 = da1 * relu_derivative(z1)                     # through ReLU
  dW1 = x.reshape(-1, 1) @ dz1.reshape(1, -1)        # gradient for W1
  db1 = dz1                                            # gradient for b1

Print dW1 shape and dW2 shape (should match W1 and W2).
"""


