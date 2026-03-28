"""
Chapter 1: Building and Training Your First Neural Network
Topic: Classifying 1D Data with a Single Neuron
Book: Deep Learning Crash Course (p.31-39)

WHAT IS AN ARTIFICIAL NEURON?
================================
An artificial neuron is the smallest building block of a neural network.
It mimics how a biological neuron works:

    Biological neuron:  receives signals → fires if threshold is reached
    Artificial neuron:  receives inputs  → outputs 0 or 1 based on weights

HOW IT WORKS (3 steps):
========================

    Step 1: Receive inputs          x = [x0, x1, ..., xN]
    Step 2: Calculate potential      p = w · x + b     (weighted sum + bias)
    Step 3: Apply activation         y = H(p)          (Heaviside step function)

    H(p) = 0 if p <= 0
    H(p) = 1 if p > 0


VISUAL — Single Neuron (1D classification):
============================================

    Input        Weight       Activation        Output
    ─────        ──────       ──────────        ──────
                              ┌─────────┐
     x0 ──── × w0 ────────▶  │  H(p)   │ ────▶  y
                              │ 0 or 1  │        (class 0 or 1)
                              └─────────┘

    The neuron decides:
    • If w0 * x0 > 0  →  output = 1
    • If w0 * x0 <= 0  →  output = 0


TRAINING — How the neuron learns:
===================================

    1. Start with random weight w0
    2. Pick a random data point (x, true_label)
    3. Predict: y_predicted = neuron(w0, x)
    4. If wrong: update weight →  w0 = w0 - η * (y_predicted - true_label) * x
    5. Repeat many times

    η (eta) = learning rate (controls how big each update step is)


EXAMPLE — Building and training a single neuron:
==================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

# === Step 1: Create simple 1D data ===
# 10 points: negative x → class 0, positive x → class 1
x = np.array([0.70, -0.67, -1.26, -0.40, -0.06, -0.55, -1.49, -0.08, 0.97, 0.43])
y_gt = np.array([1, 0, 0, 0, 1, 0, 0, 1, 1, 1])  # ground truth labels

# === Step 2: Define the neuron ===
def neuron_1d(w0, x):
    """Single neuron: returns 1 if w0*x > 0, else 0"""
    return (w0 * x > 0).astype(int)

# === Step 3: Random initialization ===
rng = default_rng()
w0 = rng.standard_normal()  # random weight
print(f"Initial weight: w0 = {w0:.4f}")
print(f"Predictions before training: {neuron_1d(w0, x)}")
print(f"Ground truth:                {y_gt}")

# === Step 4: Train the neuron ===
eta = 0.1              # learning rate
num_iterations = 100

for i in range(num_iterations):
    idx = rng.integers(0, len(x))        # pick random sample
    x_sel = x[idx]
    y_gt_sel = y_gt[idx]
    y_p_sel = neuron_1d(w0, x_sel)        # predict
    error = y_p_sel - y_gt_sel            # calculate error
    w0 = w0 - eta * error * x_sel         # update weight

print(f"\nTrained weight: w0 = {w0:.4f}")
print(f"Predictions after training:  {neuron_1d(w0, x)}")
print(f"Ground truth:                {y_gt}")

# === Step 5: Visualize ===
y_pred = neuron_1d(w0, x)
plt.figure(figsize=(8, 4))
plt.scatter(x, y_gt, s=50, c="black", label="ground truth", zorder=3)
plt.scatter(x, y_pred, s=120, c="orange", marker="x", label="predicted", zorder=2)
plt.xlabel("x", fontsize=14)
plt.ylabel("y (class)", fontsize=14)
plt.title("Single Neuron — 1D Classification", fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("crash_course/ch01/01_single_neuron_1d.png", dpi=100)
plt.show()


# ============================================================
# EXERCISE 1 (from the book — Exercise 1-1)
# ============================================================
# Run the code above several times (re-run the whole file).
# The initial random weight changes each time, but after training,
# the neuron always converges to the same result.
#
# Question: WHY does the trained neuron always end up the same?
#
# Write your answer as a comment below:
# answer_1 = "..."


# ============================================================
# EXERCISE 2 (from the book — Exercise 1-3)
# ============================================================
# The learning rate (eta) controls how fast the neuron learns.
# Try different learning rates and observe what happens.
#
# Your task:
# - Train the neuron with eta = 0.01 (very small)
# - Train the neuron with eta = 1.0  (very large)
# - Print the trained weight and predictions for each
#
# Which learning rate trains faster? Which one overshoots?

# --- your code here ---


# ============================================================
# EXERCISE 3 (from the book — Exercise 1-2)
# ============================================================
# Look at the data: x = -0.06 has label 1, and x = -0.08 has label 1
# But they are NEGATIVE numbers! Our neuron classifies all negatives as 0.
#
# The neuron can NEVER get these right with just w0 (no bias).
#
# Your task: Add a BIAS term to fix this.
# Modify the neuron so it computes: (w0 * x + b > 0)
# Then train BOTH w0 and b.
#
# Hint: the update rules are:
#   w0 = w0 - eta * error * x_selected
#   b  = b  - eta * error

# --- your code here ---
