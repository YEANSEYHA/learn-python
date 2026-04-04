# ============================================
# MODULE 6: Optimizers & Regularization
# Book: Ch 15 (Optimizers)
# ============================================

# --- WHY OPTIMIZERS? ---
# In backprop we did: w = w - lr * gradient
# That's "vanilla SGD" — it works but it's slow and zig-zags.
# Better optimizers = faster training + better results.

import numpy as np

# ── EXAMPLE 1: SGD vs Momentum ──
# We'll optimize a simple function: f(w) = (w - 5)^2
# Minimum is at w = 5. Let's watch how fast each gets there.

print("=== SGD vs Momentum ===")

# The function and its gradient
def loss_fn(w):
    return (w - 5) ** 2

def gradient_fn(w):
    return 2 * (w - 5)

# --- Vanilla SGD ---
w_sgd = 0.0
lr = 0.1

sgd_path = [w_sgd]
for step in range(20):
    grad = gradient_fn(w_sgd)
    w_sgd = w_sgd - lr * grad
    sgd_path.append(w_sgd)

# --- SGD with Momentum ---
w_mom = 0.0
velocity = 0.0
lr = 0.1
momentum = 0.9    # how much of previous velocity to keep

mom_path = [w_mom]
for step in range(20):
    grad = gradient_fn(w_mom)
    velocity = momentum * velocity + grad    # build up speed
    w_mom = w_mom - lr * velocity            # move with velocity
    mom_path.append(w_mom)

print(f"SGD after 20 steps:      w = {sgd_path[-1]:.6f}")
print(f"Momentum after 20 steps: w = {mom_path[-1]:.6f}")
print(f"Target: 5.0")

# SGD gets there smoothly but slowly
# Momentum overshoots then settles — but converges faster on hard problems!

print("\nFirst 8 steps comparison:")
print(f"{'Step':<6} {'SGD w':<12} {'Momentum w':<12}")
for i in range(8):
    print(f"{i:<6} {sgd_path[i]:<12.4f} {mom_path[i]:<12.4f}")


# ── EXAMPLE 2: Adam Optimizer ──
# Adam = Adaptive Moment Estimation
# It keeps a running average of gradients (m) and squared gradients (v)
# This lets it take big steps in smooth directions, small steps in bumpy ones

print("\n=== Adam Optimizer ===")

w_adam = 0.0
m = 0.0         # first moment (mean of gradients)
v = 0.0         # second moment (mean of squared gradients)
lr = 0.5
beta1 = 0.9     # decay rate for m
beta2 = 0.999   # decay rate for v
epsilon = 1e-8  # prevents division by zero

adam_path = [w_adam]
for t in range(1, 21):  # Adam uses t starting from 1
    grad = gradient_fn(w_adam)

    # Update moving averages
    m = beta1 * m + (1 - beta1) * grad          # average direction
    v = beta2 * v + (1 - beta2) * grad ** 2     # average magnitude

    # Bias correction (important in early steps when m,v are near 0)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # Update weight
    w_adam = w_adam - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    adam_path.append(w_adam)

print(f"Adam after 20 steps: w = {w_adam:.6f}  (target: 5.0)")

print("\nFirst 8 steps comparison (all three):")
print(f"{'Step':<6} {'SGD':<12} {'Momentum':<12} {'Adam':<12}")
for i in range(8):
    print(f"{i:<6} {sgd_path[i]:<12.4f} {mom_path[i]:<12.4f} {adam_path[i]:<12.4f}")


# ============================================
# PRACTICE: Your exercises go below
# ============================================

"""
Problem 1: Vanilla SGD — the foundation of everything

The idea is dead simple:
  - You have a value w, and a target you want to reach
  - The GRADIENT tells you which direction to move
  - You take a small step in that direction
  - Repeat

Function: loss = (w - 5) ** 2
Gradient: gradient = 2 * (w - 5)

   gradient is NEGATIVE when w is too small → pushes w UP
   gradient is POSITIVE when w is too big   → pushes w DOWN

Update rule (just ONE line!):
   w = w - lr * gradient

Use: w = 0.0, lr = 0.1, run 15 steps.
Print w and loss each step.
"""

print("\n\n\n")
w = 0.0
lr = 0.1
for i in range(1,16):
    loss = (w-5)**2

    gradient = 2 * (w - 5)

    w = w - lr * gradient

    print(f"Step {i}, W: {w:.4f} , Loss: {loss:.4f}")


"""
Problem 2: Add momentum to SGD

Same function: loss = (w - 5) ** 2, gradient = 2 * (w - 5)

The ONLY difference from Problem 1:
  Instead of:  w = w - lr * gradient
  You do:      velocity = 0.9 * velocity + gradient
               w = w - lr * velocity

Use: w = 0.0, lr = 0.1, velocity = 0.0
Run 15 steps. Print w and loss each step.

Compare your output to Problem 1 — momentum reaches 5.0 faster
but OVERSHOOTS (goes past 5.0 then comes back).
"""
print("\n\n\n\n")

w = 0.0
lr = 0.1
velocity = 0.0
for i in range(1,16):
    loss = (w-5)**2

    gradient = 2 * (w - 5)

    velocity = 0.9 * velocity + gradient
    w = w - lr * velocity

    print(f"Step {i},W: {w:.4f} , Loss: {loss:.4f}")


"""
Problem 3: Adam optimizer

Same function: loss = (w - 5) ** 2, gradient = 2 * (w - 5)

Adam keeps TWO running averages:
  m = average of gradients       (like momentum — which direction?)
  v = average of gradient²       (how bumpy is the path?)

Update rule (4 lines inside loop):
  m = 0.9 * m + 0.1 * gradient
  v = 0.999 * v + 0.001 * gradient ** 2
  m_hat = m / (1 - 0.9 ** t)       # fix early bias
  v_hat = v / (1 - 0.999 ** t)     # fix early bias
  w = w - lr * m_hat / (np.sqrt(v_hat) + 1e-8)

Use: w = 0.0, lr = 0.5, m = 0.0, v = 0.0
Run 30 steps (t from 1 to 30). Print every 5 steps.

Notice: Adam reaches 5.0 WITHOUT the wild overshooting!
"""

print("Adam \n\n\n")

w = 0.0
lr = 0.5
m = 0.0
v = 0.0
for i in range(1,31):
    loss = (w-5)**2

    gradient = 2 * (w - 5)

    m = 0.9 * m + 0.1 * gradient
    v = 0.999 * v + 0.001 * gradient ** 2
    m_hat = m / (1 - 0.9 ** i)       # fix early bias
    v_hat = v / (1 - 0.999 ** i)     # fix early bias
    w = w - lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    if i % 5 == 0:
        print(f"Step {i},W: {w:.4f} , Loss: {loss:.4f}")
    