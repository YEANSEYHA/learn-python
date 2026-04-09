# ============================================
# MODULE 6: Full Neural Network from Scratch
# Putting it all together!
# ============================================

import numpy as np
np.random.seed(42)

# ── OUR DATASET ──
# Simple 2D points: class 0 (bottom-left) vs class 1 (top-right)
X = np.array([
    [0.1, 0.2],   # class 0
    [0.2, 0.1],   # class 0
    [0.3, 0.3],   # class 0
    [0.8, 0.7],   # class 1
    [0.9, 0.8],   # class 1
    [0.7, 0.9],   # class 1
])
y = np.array([0, 0, 0, 1, 1, 1])  # labels

print("Dataset:")
for i in range(len(X)):
    print(f"  {X[i]} → class {y[i]}")


# ── HELPER FUNCTIONS (you've already built these!) ──
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)


# ── EXAMPLE: The full network, all parts labeled ──

# STEP 1: Initialize weights randomly (small numbers)
# Hidden layer: 2 inputs → 3 neurons
W1 = np.random.randn(2, 3) * 0.5   # shape (2, 3)
b1 = np.zeros(3)                     # shape (3,)

# Output layer: 3 inputs → 1 neuron
W2 = np.random.randn(3, 1) * 0.5   # shape (3, 1)
b2 = np.zeros(1)                     # shape (1,)

print(f"\nW1 shape: {W1.shape}  (2 inputs → 3 hidden neurons)")
print(f"W2 shape: {W2.shape}  (3 hidden → 1 output neuron)")

# STEP 2: Forward pass (1 sample)
x = X[0]  # first sample: [0.1, 0.2]

z1 = x @ W1 + b1           # hidden layer: weighted sum
a1 = relu(z1)               # hidden layer: activation
z2 = a1 @ W2 + b2           # output layer: weighted sum
prediction = sigmoid(z2)    # output layer: activation (0-1)

print(f"\nForward pass for {x}:")
print(f"  Hidden (z1): {np.round(z1, 3)}")
print(f"  Hidden (a1): {np.round(a1, 3)}  ← after ReLU")
print(f"  Output:      {prediction[0]:.4f}  ← probability of class 1")
print(f"  True label:  {y[0]}")


# ============================================
# PRACTICE: Your exercises go below
# ============================================

"""
Problem 1: Forward pass for ALL samples

Loop through every sample in X and compute the prediction.

For each sample x in X:
  z1 = x @ W1 + b1
  a1 = relu(z1)
  z2 = a1 @ W2 + b2
  pred = sigmoid(z2)

Print each sample's prediction and true label.

(Use the SAME W1, b1, W2, b2 from above — don't re-initialize!)
"""


for x, label in zip(X, y):
    z1 = x @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    pred = sigmoid(z2)
    predicted_class = 1 if pred >= 0.5 else 0
    print(f"x: {x}, pred: {pred[0]:.4f}, class: {predicted_class}, true: {label}")

"""
NETWORK ARCHITECTURE:

INPUT (2)          HIDDEN LAYER (3 neurons)         OUTPUT (1 neuron)
─────────          ────────────────────────         ─────────────────

              W1 (2×3)                    W2 (3×1)
           ┌──────────┐              ┌──────────┐
           │          │              │          │
x[0] ─────┼───→ n₁ ──┼──ReLU──→ a₁─┼───→      │
  (0.1)    │  ╱       │              │    ╲     │
           │ ╱        │              │     ╲    │
           │╱         │              │      ╲   │
x[1] ─────┼───→ n₂ ──┼──ReLU──→ a₂─┼───→ OUT ─┼─Sigmoid─→ pred (0-1)
  (0.2)    │╲         │              │      ╱   │
           │ ╲        │              │     ╱    │
           │  ╲       │              │    ╱     │
           │   → n₃ ──┼──ReLU──→ a₃─┼───→      │
           │          │              │          │
           └──────────┘              └──────────┘
             + b1 (3)                  + b2 (1)

WEIGHT SHAPES:
  W1: (2, 3) ← every input connects to every hidden neuron (2×3 = 6 connections)
  b1: (3,)   ← one bias per hidden neuron
  W2: (3, 1) ← every hidden neuron connects to output (3×1 = 3 connections)
  b2: (1,)   ← one bias for output neuron
  Total: 6 + 3 + 3 + 1 = 13 learnable parameters
"""


"""
Problem 2: Training loop — teach the network!

Re-initialize weights fresh, then train for 500 epochs with learning rate 0.5.

Each epoch does this for EVERY sample:
  1. Forward pass:
     z1 = x @ W1 + b1
     a1 = relu(z1)
     z2 = a1 @ W2 + b2
     pred = sigmoid(z2)

  2. Loss (binary cross-entropy for one sample):
     loss = -(label * np.log(pred + 1e-8) + (1 - label) * np.log(1 - pred + 1e-8))

  3. Backprop (gradients):
     dz2 = pred - label              # output error
     dW2 = a1.reshape(-1, 1) * dz2   # gradient for W2
     db2 = dz2                        # gradient for b2
     da1 = (dz2 * W2.T).flatten()    # error passed back to hidden
     dz1 = da1 * relu_derivative(z1) # through ReLU
     dW1 = x.reshape(-1, 1) @ dz1.reshape(1, -1)  # gradient for W1
     db1 = dz1                        # gradient for b1

  4. Update weights:
     W1 -= lr * dW1
     b1 -= lr * db1
     W2 -= lr * dW2
     b2 -= lr * db2

After training, print predictions for all samples.
Print the loss every 100 epochs to see it decrease.
"""


# Re-initialize weights
W1 = np.random.randn(2, 3) * 0.5
b1 = np.zeros(3)
W2 = np.random.randn(3, 1) * 0.5
b2 = np.zeros(1)
lr = 0.5

for epoch in range(500):
    total_loss = 0
    for x, label in zip(X, y):    # ← loop over EVERY sample
        # 1. Forward pass
        z1 = x @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        pred = sigmoid(z2)
        
        # 2. Loss (next step)
        loss = -(label * np.log(pred + 1e-8) + (1 - label) * np.log(1 - pred + 1e-8))
        total_loss += loss[0]
        # 3. Backprop (next step)
        dz2 = pred - label              # output error
        dW2 = a1.reshape(-1, 1) * dz2   # gradient for W2
        db2 = dz2                        # gradient for b2
        da1 = (dz2 * W2.T).flatten()    # error passed back to hidden
        dz1 = da1 * relu_derivative(z1) # through ReLU
        dW1 = x.reshape(-1, 1) @ dz1.reshape(1, -1)  # gradient for W1
        db1 = dz1  
        # 4. Update (next step)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, loss: {total_loss:.4f}")

# After training — print final predictions
print("\nFinal predictions:")
for x, label in zip(X, y):
    z1 = x @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    pred = sigmoid(z2)
    predicted_class = 1 if pred >= 0.5 else 0
    print(f"x: {x}, pred: {pred[0]:.4f}, class: {predicted_class}, true: {label}")