# ============================================
# MODULE 5: Gradient Descent from Scratch
# Book: Ch 5, 14 (Core optimization algorithm)
# ============================================

# --- WHAT IS GRADIENT DESCENT? ---
# Every ML model has a LOSS (error). Gradient descent is how we MINIMIZE that loss.
# It's the engine behind ALL neural network training.
#
# Idea: Imagine you're blindfolded on a mountain. You want to reach the bottom (lowest loss).
# Strategy: Feel the slope under your feet, take a step DOWNHILL. Repeat.
#
# That slope = the GRADIENT (derivative of loss with respect to parameters)
# The step size = the LEARNING RATE

# --- VISUAL ---
"""
    Loss
    |
    |  *                          * = starting point
    |    *
    |      *                      Each step follows the slope downhill
    |        *
    |          *    *             Learning rate controls step size:
    |            * *               - Too big  → jumps over the minimum
    |             *  ← minimum     - Too small → takes forever
    |                              - Just right → converges smoothly
    +---------------------------→ parameter value

    The UPDATE RULE:
    ┌─────────────────────────────────────────┐
    │  parameter = parameter - lr * gradient  │
    └─────────────────────────────────────────┘

    lr = learning rate (e.g. 0.01)
    gradient = slope of the loss at current position
"""

# --- EXAMPLE: Gradient Descent to find minimum of f(x) = x² ---
# The minimum of x² is at x=0. Let's watch gradient descent find it.
# Derivative of x² = 2x (this is the gradient)

import numpy as np

x = 5.0           # start far from minimum
lr = 0.1           # learning rate
history = []

print("Finding minimum of f(x) = x²")
print(f"{'Step':>4}  {'x':>8}  {'f(x)':>8}  {'gradient':>8}")
print("-" * 38)

for step in range(20):
    fx = x ** 2           # the loss
    gradient = 2 * x      # derivative of x²
    history.append((x, fx))
    print(f"{step:4d}  {x:8.4f}  {fx:8.4f}  {gradient:8.4f}")
    x = x - lr * gradient  # THE UPDATE RULE

print(f"\nFinal x = {x:.6f} (should be close to 0)")


# --- EXAMPLE 2: Gradient Descent for Linear Regression ---
# Instead of using sklearn, let's train y = wx + b FROM SCRATCH
# Loss = MSE = mean((y_pred - y_true)²)

np.random.seed(42)

# Generate data: y = 3x + 7 + noise
X = np.random.rand(100) * 10
y_true = 3 * X + 7 + np.random.randn(100) * 2

# Initialize random parameters
w = 0.0   # weight (slope)
b = 0.0   # bias (intercept)
lr = 0.001
epochs = 500

print("\n\nTraining linear regression from scratch")
print(f"True values: w=3, b=7")
print(f"{'Epoch':>5}  {'w':>8}  {'b':>8}  {'MSE':>10}")
print("-" * 38)

for epoch in range(epochs):
    # Forward: predict
    y_pred = w * X + b

    # Loss: MSE
    mse = np.mean((y_pred - y_true) ** 2)

    # Gradients (derivatives of MSE with respect to w and b)
    dw = np.mean(2 * (y_pred - y_true) * X)   # d(MSE)/dw
    db = np.mean(2 * (y_pred - y_true))        # d(MSE)/db

    # Update
    w = w - lr * dw
    b = b - lr * db

    if epoch % 100 == 0:
        print(f"{epoch:5d}  {w:8.4f}  {b:8.4f}  {mse:10.4f}")

print(f"\nLearned: w={w:.3f}, b={b:.3f}")
print(f"Actual:  w=3.000, b=7.000")


# ============================================
# PRACTICE: Your exercises go below
# ============================================

# --- Problem 1: Learning Rate Effect ---
# The learning rate controls how big each step is.
#
# Using f(x) = x² (gradient = 2x), starting at x = 5.0:
#   1. Run gradient descent for 20 steps with lr=0.01 (slow)
#   2. Run gradient descent for 20 steps with lr=0.3  (fast)
#   3. Run gradient descent for 20 steps with lr=1.0  (too big!)
#   4. Print the final x value for each — which lr works best?
#
# What to notice: lr=1.0 will NEVER converge (x bounces around or explodes)


results = []
for lr in [0.01, 0.3, 1.0]:
    x = 5.0
    for step in range(20):
        x = x - lr * (2 * x)
    results.append({"lr": lr, "final_x": x})
    print(f"lr={lr:4} → Final x = {x:.6f}")

best = min(results, key=lambda r: abs(r['final_x']))
print(f"\nBest lr={best['lr']} (closest to 0: x={best['final_x']:.6f})")


# --- Problem 2: Gradient Descent for Linear Regression ---
# In Example 2 above, we trained y = wx + b from scratch.
# But the learned values (w=3.7, b=2.3) weren't perfect (true: w=3, b=7).
#
# Your task: improve it by tuning EPOCHS and LEARNING RATE.
#
# Using the same data from Example 2 (X and y_true are already defined above):
#   1. Set w=0, b=0
#   2. Try lr=0.002 and epochs=2000
#   3. Inside the loop: compute y_pred, mse, gradients dw and db, then update w and b
#   4. Print w, b, and MSE every 500 epochs
#   5. Print final w and b — they should be closer to w=3, b=7 than the example
#
# Reminder of the gradients:
#   dw = mean(2 * (y_pred - y_true) * X)
#   db = mean(2 * (y_pred - y_true))

w = 0
b = 0
lr = 0.002

print("\n\nProblem 2: Linear Regression from Scratch")
print(f"{'Epoch':>5}  {'w':>8}  {'b':>8}  {'MSE':>10}")
print("-" * 38)

for step in range(2000):
    y_pred = w * X + b

    # Loss
    mse = np.mean((y_pred - y_true) ** 2)

    # Gradients
    dw = np.mean(2 * (y_pred - y_true) * X)
    db = np.mean(2 * (y_pred - y_true))

    # Update
    w = w - lr * dw
    b = b - lr * db

    if step % 500 == 0:
        print(f"{step:5d}  {w:8.4f}  {b:8.4f}  {mse:10.4f}")

print(f"\nLearned: w={w:.3f}, b={b:.3f}")
print(f"Actual:  w=3.000, b=7.000")


# --- Problem 3: Track the Loss Curve ---
# In real ML, we always watch how loss changes over time.
# If loss goes DOWN → model is learning. If it goes UP or explodes → something is wrong.
#
# Using the same linear regression setup (y = wx + b):
#   1. Set w=0, b=0, lr=0.002, epochs=2000
#   2. Train with gradient descent (same as Problem 2)
#   3. Save the MSE at every step into a list called "losses"
#   4. After training, print:
#      - First loss (epoch 0)
#      - Last loss (epoch 1999)
#      - Whether loss decreased (last < first)

w = 0
b = 0
lr = 0.002
losses = []

for epoch in range(2000):
    y_pred = w * X + b
    mse = np.mean((y_pred - y_true) ** 2)
    losses.append(mse)

    dw = np.mean(2 * (y_pred - y_true) * X)
    db = np.mean(2 * (y_pred - y_true))

    w = w - lr * dw
    b = b - lr * db

print(f"First loss (epoch 0):    {losses[0]:.4f}")
print(f"Last loss (epoch 1999):  {losses[-1]:.4f}")
print(f"Loss decreased: {losses[-1] < losses[0]}")
