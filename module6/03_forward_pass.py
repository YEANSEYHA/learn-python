# ============================================
# MODULE 6: Forward Pass
# Book: Ch 13 (Neural Networks)
# ============================================

# --- WHAT IS THE FORWARD PASS? ---
# The forward pass is how data flows THROUGH a neural network to make a prediction.
# Each layer takes the previous layer's output, applies weights + bias, then activation.
# The final layer produces the prediction.

# --- VISUAL ---
"""
    A 3-layer neural network (input → hidden → output):

    INPUT LAYER      HIDDEN LAYER       OUTPUT LAYER
    (3 features)     (4 neurons)        (1 neuron)

       x1 ──┐                              ┌──→ output
       x2 ──┤──→ [z = Wx + b] ──→ [ReLU] ──┤
       x3 ──┘                              └── sigmoid (for binary classification)

    Step by step:
    1. Input  → Hidden:  z1 = W1 · x  + b1  → a1 = relu(z1)
    2. Hidden → Output:  z2 = W2 · a1 + b2  → a2 = sigmoid(z2)

    Each arrow has a WEIGHT.
    Each layer has a BIAS.
    Each layer applies an ACTIVATION FUNCTION.

    The output of one layer becomes the INPUT of the next.
"""

import numpy as np

# Activation functions
def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# --- EXAMPLE: Forward pass through a 2-layer network ---
np.random.seed(42)

# Input: 1 sample with 3 features
x = np.array([1.5, 2.0, -1.0])   # shape: (3,)

# Layer 1: 3 inputs → 4 neurons
W1 = np.random.randn(4, 3) * 0.1  # shape: (4, 3)
b1 = np.zeros(4)                   # shape: (4,)

# Layer 2: 4 inputs → 1 neuron (output)
W2 = np.random.randn(1, 4) * 0.1  # shape: (1, 4)
b2 = np.zeros(1)                   # shape: (1,)

# Forward pass
z1 = np.dot(W1, x) + b1       # hidden layer weighted sum
a1 = relu(z1)                  # hidden layer activation
z2 = np.dot(W2, a1) + b2      # output layer weighted sum
a2 = sigmoid(z2)               # output layer activation (probability)

print("=== Forward Pass ===")
print(f"Input x:        {x}")
print(f"Hidden z1:      {np.round(z1, 4)}")
print(f"Hidden a1:      {np.round(a1, 4)}  ← after ReLU")
print(f"Output z2:      {np.round(z2, 4)}")
print(f"Output a2:      {np.round(a2, 4)}  ← final prediction (probability)")
print(f"Predicted class: {1 if a2 >= 0.5 else 0}")


# ============================================
# PRACTICE: Your exercises go below
# ============================================

# --- Problem 1: Manual forward pass ---
# Build a forward pass for a network with:
#   - Input: x = [1.0, 0.5, -1.5, 2.0]  (4 features)
#   - Layer 1: 3 neurons, ReLU activation
#     W1 = [[ 0.1, -0.2,  0.3,  0.5],
#            [ 0.4,  0.1, -0.1,  0.2],
#            [-0.3,  0.5,  0.2, -0.4]]
#     b1 = [0.1, 0.0, -0.1]
#   - Layer 2: 1 neuron, Sigmoid activation
#     W2 = [[0.6, -0.3, 0.8]]
#     b2 = [0.2]
#
# Your task:
#   1. Compute z1, a1 (hidden layer)
#   2. Compute z2, a2 (output layer)
#   3. Print a1 and the final prediction probability a2


x = np.array([1.0, 0.5, -1.5, 2.0])

w1 = np.array([[ 0.1, -0.2,  0.3,  0.5],
            [ 0.4,  0.1, -0.1,  0.2],
            [-0.3,  0.5,  0.2, -0.4]])

b1 = np.array([0.1, 0.0, -0.1])

# 1 . z1, a1

z1 = np.dot(w1, x) + b1   # (3,4) · (4,) = (3,)

a1 = relu(z1)

print("Z1 :",z1)
print("A1: ",a1)

# 2
W2 = np.array([[0.6, -0.3, 0.8]])
b2 = np.array([0.2])

z2 = np.dot(W2, a1) + b2
a2 = sigmoid(z2)

print(f"A1 (hidden): {a1}")
print(f"A2 (output probability): {a2}")


# --- Problem 2: Multi-class forward pass ---
# So far we used sigmoid for binary (0 or 1).
# For multi-class problems, the output layer uses SOFTMAX instead.
#
# Build a forward pass for a 3-class classifier:
#   - Input: x = [0.5, 1.2, -0.8]  (3 features)
#   - Layer 1: 4 neurons, ReLU
#     W1 = np.random.randn(4, 3) * 0.1  (use np.random.seed(0) before this)
#     b1 = np.zeros(4)
#   - Layer 2: 3 neurons (one per class), Softmax
#     W2 = np.random.randn(3, 4) * 0.1
#     b2 = np.zeros(3)
#
# Your task:
#   1. Compute z1, a1 (hidden layer with ReLU)
#   2. Compute z2, a2 (output layer with Softmax)
#   3. Print the 3 class probabilities and which class wins
#
# Hint: softmax() is already defined at the top of the file

def softmax(z):
    exp_z = np.exp(z - np.max(z))   # subtract max for numerical stability
    return exp_z / exp_z.sum()

x = np.array([0.5, 1.2, -0.8])
# z1 = w.x +b
# Compute z1, a1
W1 = np.random.randn(4, 3) * 0.1
b1 = np.zeros(4)

z1 = np.dot(W1, x) + b1
a1 = relu(z1)

print("z1 : ",z1)
print("a1 : ",a1)

# Compute z2
# z2 = w.x1 + b2
W2 = np.random.randn(3, 4) * 0.1
b2 = np.zeros(3)

z2 = np.dot(W2, a1) + b2   # feed a1 (after ReLU), not z1
a2 = softmax(z2)

# 3 Print class probabilities and winner
classes = ['class 0', 'class 1', 'class 2']
for name, prob in zip(classes, a2):
    print(f"{name}: {prob:.4f}")

winner = np.argmax(a2)
print(f"\nPredicted: {classes[winner]} ({a2[winner]:.4f})")


# --- Problem 3: Forward pass on a BATCH of samples ---
# In real training we process many samples at once (a "batch"), not one at a time.
# NumPy handles this automatically — just change x from shape (3,) to (N, 3).
#
# Use the same W1, b1, W2, b2 from Problem 2 (already defined above).
# Input batch: 4 samples, each with 3 features
#   X = [[ 0.5,  1.2, -0.8],
#        [-1.0,  0.3,  0.5],
#        [ 0.2, -0.5,  1.1],
#        [ 1.5,  0.0, -0.3]]
#
# Your task:
#   1. Compute z1 = X @ W1.T + b1   (use @ for matrix multiply, .T to transpose W1)
#   2. Apply ReLU → a1
#   3. Compute z2 = a1 @ W2.T + b2
#   4. Apply softmax row-wise to get a2 — shape should be (4, 3)
#   5. Print the predicted class for each of the 4 samples
#
# Note: for batch softmax use: np.array([softmax(row) for row in z2])


X = np.array([[ 0.5,  1.2, -0.8],
        [-1.0,  0.3,  0.5],
        [ 0.2, -0.5,  1.1],
        [ 1.5,  0.0, -0.3]]
)

# 1. Compute z1

z1 = X @ W1.T + b1
print("z1",z1)

# 2. Apply Relu
a1 = relu(z1)

print("a1 ",a1)

# 3. Compute z2
z2 = a1 @ W2.T + b2
print("z2 ",z2)

# apply softmax row-by-row
a2 = np.array([softmax(row) for row in z2])  # shape: (4, 3)

# 5. Print predicted class for each sample
classes = ['class 0', 'class 1', 'class 2']
for i, probs in enumerate(a2):
    winner = np.argmax(probs)
    print(f"Sample {i}: {classes[winner]} ({probs[winner]:.4f})")





