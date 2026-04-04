# ============================================
# MODULE 6: Regularization
# Book: Ch 9 (Overfitting), Ch 15 (Optimizers)
# ============================================

# Regularization = tricks to prevent overfitting
# Two big ones: L2 (weight decay) and Dropout

import numpy as np

# ── EXAMPLE 1: L2 Regularization ──
# Add a penalty for large weights to the loss
# loss = error + lambda * sum(weights²)

print("=== L2 Regularization ===")

weights = np.array([10.0, 20.0, 5.0])   # big weights
error = 3.0                              # prediction error

lam = 0.01   # lambda = how much to penalize (small number)

# Without L2
loss_no_reg = error
print(f"Loss without L2: {loss_no_reg}")

# With L2
penalty = lam * np.sum(weights ** 2)
loss_with_reg = error + penalty
print(f"L2 penalty:      {penalty:.2f}")
print(f"Loss with L2:    {loss_with_reg:.2f}")

# The penalty pushes the optimizer to shrink weights!
small_weights = np.array([1.0, 2.0, 0.5])
small_penalty = lam * np.sum(small_weights ** 2)
print(f"\nSmall weights penalty: {small_penalty:.4f}  ← much less!")


# ── EXAMPLE 2: Dropout ──
# Randomly set some neuron outputs to 0 during training

print("\n=== Dropout ===")

layer_output = np.array([0.5, 0.8, 0.3, 0.9, 0.2])
dropout_rate = 0.4   # kill 40% of neurons

# Create a random mask: 1 = keep, 0 = drop
np.random.seed(42)
mask = np.random.rand(5) > dropout_rate   # True/False array
print(f"Original output: {layer_output}")
print(f"Dropout mask:    {mask.astype(int)}")  # 1s and 0s

# Apply mask
dropped = layer_output * mask
print(f"After dropout:   {dropped}")

# Scale up survivors so total signal stays similar
# (divide by keep probability)
scaled = dropped / (1 - dropout_rate)
print(f"After scaling:   {scaled}")
print("(Scaling keeps the average signal strength the same)")


# ============================================
# PRACTICE: Your exercises go below
# ============================================

"""
Problem 1: Compute L2 penalty

Given these weights:
  weights = np.array([3.0, -4.0, 2.0, -1.0])
  lam = 0.01

Compute:
  penalty = lam * np.sum(weights ** 2)

Then compute the total loss:
  prediction_error = 5.0
  total_loss = prediction_error + penalty

Print the penalty and total_loss.
"""
weights = np.array([3.0, -4.0, 2.0, -1.0])
lam = 0.01
penalty = lam * np.sum(weights ** 2)

print("Penalty :",penalty)

prediction_error = 5.0

total_loss = prediction_error + penalty

print("Total Loss : ",total_loss)


"""
Problem 2: Apply dropout to a layer

Given a layer output:
  layer = np.array([0.7, 0.3, 0.9, 0.5, 0.1, 0.8])
  dropout_rate = 0.5   # kill 50% of neurons

Steps:
  1. Create a random mask:
     mask = np.random.rand(6) > dropout_rate
  2. Apply mask:
     dropped = layer * mask
  3. Scale survivors (so average signal stays the same):
     result = dropped / (1 - dropout_rate)

Print: original layer, mask (as 1s and 0s), and result.
Use np.random.seed(0) before creating the mask.
"""

layer = np.array([0.7, 0.3, 0.9, 0.5, 0.1, 0.8])
dropout_rate = 0.5   # kill 50% of neurons

# Create random mask
np.random.seed(42)
mask = np.random.rand(6) > dropout_rate

# Apply mask
dropped = layer * mask


result = dropped / (1 - dropout_rate)

print("REsults",result)


"""
Problem 3: Training with L2 regularization

Train a single weight to predict target, but WITH L2 penalty.

  x = 2.0, target = 6.0, w = 0.0, lr = 0.1, lam = 0.1

Loop 20 steps:
  prediction = w * x
  error = (prediction - target) ** 2
  l2_penalty = lam * w ** 2
  total_loss = error + l2_penalty

  # Gradient now includes the L2 term:
  gradient = 2 * (prediction - target) * x + 2 * lam * w

  w = w - lr * gradient

Print every 5 steps: step, w, total_loss.

Without L2, w would reach exactly 3.0 (since 2*3=6).
With L2, w will be slightly LESS than 3.0 — the penalty
pulls weights toward zero.
"""

print("\n\n\n")
x = 2.0 
target = 6.0
w = 0.0
lr = 0.1
lam = 0.1
for i in range(1,21,1):
    print(i)
    prediction = w * x
    error = (prediction - target) **2
    l2_penalty = lam * w ** 2
    total_loss = error + l2_penalty

    # Gradient
    gradient = 2 * (prediction - target) * x + 2 * lam * w
    w = w - lr * gradient
    if i % 5 == 0 :
        print(f"Step : {i}, Total Loss: {total_loss:.4f}")

