# ============================================
# MODULE 6: Backpropagation
# Book: Ch 14 (How Networks Learn)
# ============================================

# --- WHAT IS BACKPROPAGATION? ---
# Think of it like throwing darts:
#   1. You throw (forward pass) → miss the bullseye
#   2. You see how far you missed (loss)
#   3. You figure out what to adjust (backprop)
#   4. You adjust your aim (update weights)
#   5. Repeat until you hit the target!

# --- VISUAL ---
"""
    THE DART ANALOGY:

    Throw #1:  target = 20,  you got 24  →  off by 4  →  adjust!
    Throw #2:  target = 20,  you got 21  →  off by 1  →  adjust!
    Throw #3:  target = 20,  you got 20  →  perfect!  →  done!

    In a neural network:
    ┌─────────────────────────────────────────────────────┐
    │  1. Forward pass  → make a prediction               │
    │  2. Compute loss  → how far did we miss?            │
    │  3. Backward pass → who's to blame? (gradients)     │
    │  4. Update weights → nudge weights to fix the miss  │
    │  Repeat until loss is small enough                  │
    └─────────────────────────────────────────────────────┘

    CHAIN OF BLAME (chain rule):

    x = 2  →  [× w=3]  →  [× v=4]  →  24  →  LOSS (target was 20)

    Who caused the miss?
    ├── v multiplied by 4 → made it too big
    └── w multiplied by 3 → fed v a big number

    Backprop passes blame backwards, step by step.
"""

import numpy as np

# ── EXAMPLE 1: The simplest backprop (just 1 weight) ──
# A "network" with 1 input, 1 weight, no activation
# prediction = x * w
# We want to learn the right w so that prediction = target

x = 2.0           # input
target = 10.0      # we want prediction to be 10
w = 0.0            # start with w = 0 (knows nothing)
lr = 0.1           # learning rate (how big each nudge is)

print("=== Simplest backprop: learn w so that x * w = target ===")
for step in range(10):
    # FORWARD: make prediction
    prediction = x * w

    # LOSS: how far off?
    loss = (prediction - target) ** 2

    # BACKWARD: gradient = how does w affect the loss?
    # gradient tells us: "if w goes UP, does loss go UP or DOWN?"
    gradient = 2 * (prediction - target) * x

    # UPDATE: nudge w in the opposite direction of gradient
    w = w - lr * gradient

    print(f"  Step {step}: w = {w:.2f}, prediction = {x*w:.2f}, loss = {loss:.2f}")

print(f"\nLearned w = {w:.2f}  (correct answer: {target/x})")


# ── EXAMPLE 2: Two weights in a chain (chain rule) ──
# prediction = v * (w * x)
# Backprop must pass blame through BOTH weights

print("\n=== Chain rule: two weights ===")
x = 2.0
target = 20.0
w = 1.0    # first weight
v = 1.0    # second weight
lr = 0.01

for step in range(50):
    # FORWARD
    hidden = w * x       # first multiply
    prediction = v * hidden   # second multiply
    loss = (prediction - target) ** 2

    # BACKWARD (chain rule: blame flows right to left)
    dLoss_dpred = 2 * (prediction - target)  # how loss changes with prediction
    dLoss_dv = dLoss_dpred * hidden          # blame on v
    dLoss_dw = dLoss_dpred * v * x           # blame on w (passes through v)

    # UPDATE both weights
    v = v - lr * dLoss_dv
    w = w - lr * dLoss_dw

    if step % 10 == 0:
        print(f"  Step {step}: w={w:.2f}, v={v:.2f}, pred={prediction:.2f}, loss={loss:.1f}")

print(f"\nFinal: {w:.2f} * {x} * {v:.2f} = {w * x * v:.2f}  (target: {target})")


"""
Problem 3: Backprop with sigmoid (a real neuron!)

Until now we just multiplied weights. Real neurons have an ACTIVATION
function (like sigmoid) that squishes the output between 0 and 1.

Your network (1 input → 1 neuron → 1 output):
  z = w * x + b            ← weighted sum + bias
  prediction = sigmoid(z)  ← squish to 0-1
  loss = (prediction - target) ** 2

Use: x = 1.0, target = 0.8, w = 0.0, b = 0.0, lr = 1.0
Run 50 steps.

FORWARD:
  z = w * x + b
  prediction = sigmoid(z)
  loss = (prediction - target) ** 2

BACKWARD (chain rule, 3 steps back):
  dLoss_dpred = 2 * (prediction - target)
  dpred_dz = sigmoid_derivative(z)
  dLoss_dz = dLoss_dpred * dpred_dz
  gradient_w = dLoss_dz * x
  gradient_b = dLoss_dz

UPDATE:
  w = w - lr * gradient_w
  b = b - lr * gradient_b

Print every 10 steps. Define sigmoid functions yourself:
  def sigmoid(z): return 1 / (1 + np.exp(-z))
  def sigmoid_derivative(z): s = sigmoid(z); return s * (1 - s)
"""




# ============================================
# PRACTICE: Your exercises go below
# ============================================

# --- Problem 1: Learn a single weight ---
# Use the same pattern from Example 1.
#
# Your network: prediction = x * w
# Input x = 3.0, target = 15.0
# Start with w = 0.0, learning rate = 0.1
#
# Write a loop that runs 10 steps:
#   1. prediction = x * w
#   2. loss = (prediction - target) ** 2
#   3. gradient = 2 * (prediction - target) * x
#   4. w = w - lr * gradient
#   Print w and prediction each step.
#
# What should w converge to? (Think: 3 * ? = 15)


x = 3.0
target = 15.0
w = 0.0
lr = 0.1

for step in range(100):
    # forward
    prediction = x * w

    loss = (prediction - target) ** 2
    # backward
    gradient = 2 * (prediction - target) * x
    w = w - lr * gradient

    print(f"  Step {step}: w = {w:.2f}, prediction = {x*w:.2f}, loss = {loss:.2f}")

print(f"\nLearned w = {w:.2f}  (correct answer: {target/x})")


"""
Problem 2: Two weights (chain rule)

Now your network has TWO weights:
  hidden = w * x
  prediction = v * hidden    (same as: v * w * x)

Use: x = 2.0, target = 20.0, w = 1.0, v = 1.0, lr = 0.01

Write a loop that runs 50 steps:
  FORWARD:
    hidden = w * x
    prediction = v * hidden
    loss = (prediction - target) ** 2
  BACKWARD:
    dLoss_dpred = 2 * (prediction - target)
    gradient_v = dLoss_dpred * hidden
    gradient_w = dLoss_dpred * v * x
  UPDATE:
    v = v - lr * gradient_v
    w = w - lr * gradient_w
  Print every 10 steps.

This is the chain rule — blame flows backwards through both weights!
"""

x = 2.0
target = 20.0
w = 1.0
v = 1.0
lr = 0.01
for step in range(50):
    # Forward
    hidden = w * x
    prediction = v * hidden

    loss = (prediction - target)**2

    #backward
    dLoss_dpred = 2 * (prediction - target)
    dLoss_dv = dLoss_dpred * hidden
    dLoss_dw = dLoss_dpred * v * x

    # update both weights
    v = v- lr * dLoss_dv
    w = w-lr* dLoss_dw
    if step % 10 == 0:
        print(f"  Step {step}: w={w:.2f}, v={v:.2f}, pred={prediction:.2f}, loss={loss:.1f}")


print(f"\nFinal: {w:.2f} * {x} * {v:.2f} = {w * x * v:.2f}  (target: {target})")


print("\n\n\n\n\n\n")


"""
Problem 3: Backprop with sigmoid (a real neuron!)

Until now we just multiplied weights. Real neurons have an ACTIVATION
function (like sigmoid) that squishes the output between 0 and 1.

This changes the backward pass — we need the sigmoid DERIVATIVE too.

Reminder:
  sigmoid(z) = 1 / (1 + exp(-z))
  sigmoid_derivative(z) = sigmoid(z) * (1 - sigmoid(z))

Your network (1 input → 1 neuron → 1 output):
  z = w * x + b            ← weighted sum + bias
  prediction = sigmoid(z)  ← squish to 0-1
  loss = (prediction - target) ** 2

Use: x = 1.0, target = 0.8, w = 0.0, b = 0.0, lr = 1.0
Run 50 steps.

FORWARD:
  z = w * x + b
  prediction = sigmoid(z)
  loss = (prediction - target) ** 2

BACKWARD (chain rule, 3 steps back):
  dLoss_dpred = 2 * (prediction - target)
  dpred_dz = sigmoid_derivative(z)
  dLoss_dz = dLoss_dpred * dpred_dz
  gradient_w = dLoss_dz * x
  gradient_b = dLoss_dz

UPDATE:
  w = w - lr * gradient_w
  b = b - lr * gradient_b

Print every 10 steps. Define sigmoid functions yourself:
  def sigmoid(z): return 1 / (1 + np.exp(-z))
  def sigmoid_derivative(z): s = sigmoid(z); return s * (1 - s)
"""

from math import exp

def sigmoid(z):
    return 1 / (1 + exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))
   

x = 1.0
target = 0.8
w = 0.0
b = 0.0
lr = 1.0

for step in range(51):
    # Forward
    z = w * x + b
    prediction = sigmoid(z)
    loss = (prediction - target) ** 2
    # print(f"Loss {loss}")
    
    # backward
    dLoss_dpred = 2 * (prediction - target)
    dpred_dz = sigmoid_derivative(z)
    dLoss_dz = dLoss_dpred * dpred_dz

    gradient_w = dLoss_dz * x
    gradient_b = dLoss_dz

    w = w - lr * gradient_w
    b = b - lr * gradient_b

    if step % 10 == 0:
        print(f"Step : {step}, Loss: {loss}")
        print("B",b)
        print("W",w)


