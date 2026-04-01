# ============================================
# MODULE 6: Backpropagation
# Book: Ch 14 (How Networks Learn)
# ============================================

# --- WHAT IS BACKPROPAGATION? ---
# After the forward pass gives a prediction and we compute the loss,
# backprop sends the ERROR BACKWARDS through the network to compute
# how much each weight contributed to the mistake.
# Then gradient descent updates the weights to reduce the loss.

# --- VISUAL ---
"""
    FORWARD PASS (left → right):
    x ──→ [Layer 1] ──→ [Layer 2] ──→ prediction ──→ LOSS

    BACKWARD PASS (right → left):
    x ←── [update W1] ←── [update W2] ←── gradient of loss

    The key tool: CHAIN RULE
    If loss depends on a2, and a2 depends on z2, and z2 depends on W2:
      dLoss/dW2 = dLoss/da2 × da2/dz2 × dz2/dW2

    Each layer passes its gradient to the layer before it.

    THE FULL TRAINING LOOP:
    ┌─────────────────────────────────────────────────────┐
    │  1. Forward pass  → get prediction                  │
    │  2. Compute loss  → how wrong are we?               │
    │  3. Backward pass → compute gradients (chain rule)  │
    │  4. Update weights → W = W - lr * gradient          │
    │  Repeat until loss is small                         │
    └─────────────────────────────────────────────────────┘
"""

import numpy as np

# Activation functions + their DERIVATIVES (needed for backprop)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)      # derivative of sigmoid

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)   # 1 where z>0, 0 where z<=0

# MSE loss
def mse(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

# --- EXAMPLE: Full training loop for a 2-layer network ---
np.random.seed(42)

# Data: y = 1 if sum(x) > 0 else 0
X = np.array([[1.0, 2.0],
              [-1.0, -2.0],
              [0.5, -0.5],
              [-0.5, 1.0]])
y = np.array([[1], [0], [0], [1]])   # shape (4, 1)

# Network: 2 inputs → 3 hidden → 1 output
W1 = np.random.randn(3, 2) * 0.1
b1 = np.zeros((1, 3))
W2 = np.random.randn(1, 3) * 0.1
b2 = np.zeros((1, 1))

lr = 0.1
losses = []

for epoch in range(1000):
    # ── FORWARD PASS ──
    z1 = X @ W1.T + b1          # (4,3)
    a1 = sigmoid(z1)            # (4,3)
    z2 = a1 @ W2.T + b2         # (4,1)
    a2 = sigmoid(z2)            # (4,1) ← predictions

    loss = mse(y, a2)
    losses.append(loss)

    # ── BACKWARD PASS ──
    # Gradient of loss w.r.t a2
    dL_da2 = 2 * (a2 - y) / len(y)           # (4,1)

    # Output layer gradients
    da2_dz2 = sigmoid_derivative(z2)          # (4,1)
    dL_dz2  = dL_da2 * da2_dz2               # (4,1)
    dL_dW2  = dL_dz2.T @ a1                  # (1,3)
    dL_db2  = dL_dz2.sum(axis=0, keepdims=True)

    # Hidden layer gradients (chain rule continues backwards)
    dL_da1  = dL_dz2 @ W2                    # (4,3)
    da1_dz1 = sigmoid_derivative(z1)          # (4,3)
    dL_dz1  = dL_da1 * da1_dz1              # (4,3)
    dL_dW1  = dL_dz1.T @ X                   # (3,2)
    dL_db1  = dL_dz1.sum(axis=0, keepdims=True)

    # ── UPDATE WEIGHTS ──
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2
    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1

    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d}  Loss: {loss:.4f}")

print(f"\nFinal predictions:\n{np.round(a2, 3)}")
print(f"True labels:\n{y}")


# ============================================
# PRACTICE: Your exercises go below
# ============================================

# --- Problem 1: Derivatives by hand ---
# Backprop needs derivatives of activation functions.
# Before writing full backprop, practice computing them.
#
# Given z values:
z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

# Your task:
#   1. Compute sigmoid(z) and sigmoid_derivative(z) — print both
#   2. Compute relu(z) and relu_derivative(z) — print both
#   3. Notice: relu_derivative is just 0s and 1s (much simpler than sigmoid)


