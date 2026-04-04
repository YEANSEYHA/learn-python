# ============================================
# MODULE 6: Batch Normalization
# Book: Ch 15 (Stabilizing Training)
# ============================================

# Batch norm = normalize layer inputs so training is faster & more stable
# Same idea as StandardScaler, but INSIDE the network

import numpy as np

# ── EXAMPLE: Batch Norm step by step ──

print("=== Batch Normalization ===")

# A batch of 5 values coming out of a layer
batch = np.array([100.0, -50.0, 200.0, 150.0, 0.0])
print(f"Raw layer output: {batch}")
print(f"  Mean: {batch.mean():.1f}, Std: {batch.std():.1f}")

# Step 1-2: Compute mean and variance
mean = np.mean(batch)
var = np.var(batch)

# Step 3: Normalize (mean=0, std≈1)
epsilon = 1e-8
normalized = (batch - mean) / np.sqrt(var + epsilon)
print(f"\nAfter normalizing: {np.round(normalized, 3)}")
print(f"  Mean: {normalized.mean():.4f}, Std: {normalized.std():.4f}")

# Step 4: Scale and shift (gamma and beta are LEARNABLE)
gamma = 1.0   # initially 1 (no scaling)
beta = 0.0    # initially 0 (no shifting)
output = gamma * normalized + beta
print(f"\nFinal output (γ=1, β=0): {np.round(output, 3)}")

# If the network learns gamma=2, beta=5:
gamma2, beta2 = 2.0, 5.0
output2 = gamma2 * normalized + beta2
print(f"With γ=2, β=5: {np.round(output2, 3)}")
print("(Network can learn to scale/shift however it wants)")


# ============================================
# PRACTICE: Your exercises go below
# ============================================

"""
Problem 1: Normalize a batch

Given:
  batch = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
  epsilon = 1e-8

Compute:
  mean = np.mean(batch)
  var = np.var(batch)
  normalized = (batch - mean) / np.sqrt(var + epsilon)

Print: mean, var, and normalized.
The normalized values should have mean ≈ 0 and std ≈ 1.
"""

batch = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
epsilon = 1e-8
# Compute mean
mean = np.mean(batch)
print(mean)

# Variance
var = np.var(batch)
print(var)

normalized = (batch - mean) / np.sqrt(var + epsilon)

print("Normalized",normalized)


"""
Problem 2: Add gamma and beta (scale & shift)

Using your normalized result from Problem 1, apply:
  output = gamma * normalized + beta

Try two cases:
  Case 1: gamma = 1.0, beta = 0.0  (no change)
  Case 2: gamma = 3.0, beta = 10.0 (scale up and shift)

Print both outputs.
"""

gamma = 1.0
beta = 0.0
output = gamma * normalized + beta

print("Output1:",output)

gamma = 3.0
beta = 10.0
output = gamma * normalized + beta

print("Output2:",output)


"""
Problem 3: Write a batch_norm function

def batch_norm(x, gamma, beta, epsilon=1e-8):
    # 1. mean = np.mean(x)
    # 2. var = np.var(x)
    # 3. normalized = (x - mean) / np.sqrt(var + epsilon)
    # 4. return gamma * normalized + beta

Test it with:
  x = np.array([5.0, 15.0, 25.0, 35.0])
  result = batch_norm(x, gamma=2.0, beta=1.0)
  print(result)
"""

x = np.array([5.0, 15.0, 25.0, 35.0])

def batch_norm(x, gamma, beta, epsilon=1e-8):
    mean = np.mean(x)
    var = np.var(x)
    normalized = (x - mean) / np.sqrt(var + epsilon)
    return gamma * normalized + beta

result = batch_norm(x, gamma=2.0, beta=1.0)

print(result)