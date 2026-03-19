import numpy as np

# ============================================================
# TOPIC: Probability Distributions
# ============================================================
#
# WHAT: A probability distribution describes how likely each
#       possible value is. Think of it as a "shape" that data follows.
#
# WHY FOR ML:
#   - Weight initialization: random values drawn from distributions
#   - Data understanding: is your data normal? skewed? uniform?
#   - Generative models (GANs, VAEs): learn to produce data
#     that matches a target distribution
#
# TWO MAIN TYPES:
#   Uniform  — every value equally likely (like a fair die)
#   Normal   — bell curve, most values near the mean (like heights)
#
# VISUAL:
#
#   Uniform Distribution          Normal (Gaussian) Distribution
#   (equal chance everywhere)     (bell curve — most data near center)
#
#    ▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌            ▌
#    ▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌           ▌▌▌
#    ▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌          ▌▌▌▌▌
#    ▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌         ▌▌▌▌▌▌▌
#    ▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌       ▌▌▌▌▌▌▌▌▌▌▌
#   ────────────────────    ────────────────────
#   low              high   -3σ  -2σ  -1σ  μ  1σ  2σ  3σ
#
#   68-95-99.7 RULE (Normal):
#     68% of data falls within 1 std of mean
#     95% of data falls within 2 std of mean
#     99.7% of data falls within 3 std of mean
#
# EXAMPLE:
#   # Generate random samples
#   uniform_data = np.random.uniform(low=0, high=10, size=1000)
#   normal_data  = np.random.normal(loc=5, scale=2, size=1000)
#                                    ↑ mean   ↑ std
#
#   print(np.mean(uniform_data))   # ≈ 5.0  (midpoint of 0-10)
#   print(np.std(uniform_data))    # ≈ 2.88
#
#   print(np.mean(normal_data))    # ≈ 5.0  (the loc we set)
#   print(np.std(normal_data))     # ≈ 2.0  (the scale we set)
# ============================================================

# PROBLEM 1: Distribution Explorer
# Generate 10,000 samples from each distribution:
#   - Uniform: between 0 and 100
#   - Normal: mean=50, std=15
#
# For EACH distribution, print:
#   - Mean, Median, Std
#   - Min and Max values
#   - What % of values fall between 35 and 65
#     (hint: use boolean mask to count, divide by total)
#
# Compare: which distribution has more values near the center?

np.random.seed(42)  # so results are reproducible

# Your code below:
uniform_data = np.random.uniform(low=0, high=100, size=10000)
normal_data = np.random.normal(loc=50, scale=15, size=10000)

for name, data in {'Uniform': uniform_data, 'Normal': normal_data}.items():
    print(f"\n{name} Distribution:")
    print(f"  Mean:   {np.mean(data):.2f}")
    print(f"  Median: {np.median(data):.2f}")
    print(f"  Std:    {np.std(data):.2f}")
    print(f"  Min:    {np.min(data):.2f}")
    print(f"  Max:    {np.max(data):.2f}")

    between = np.sum((data >= 35) & (data <= 65)) / len(data) * 100
    print(f"  % between 35-65: {between:.1f}%")

print("\n→ Normal has more values near the center (bell curve concentrates data around the mean)")


# ============================================================
# TOPIC: Weight Initialization with Distributions
# ============================================================
#
# WHAT: Neural networks start with random weights. The distribution
#       you pick to initialize them matters A LOT for training.
#
# WHY:
#   - Too large weights → outputs explode (gradient explosion)
#   - Too small weights → outputs shrink to zero (vanishing gradients)
#   - Right distribution → stable training from the start
#
# VISUAL:
#   Bad init (too wide):     Good init (scaled):     Bad init (too narrow):
#   weights = [-5, 8, -3]   weights = [-0.3, 0.5]   weights = [0.001, -0.001]
#         ↓                        ↓                        ↓
#   outputs EXPLODE          outputs STABLE            outputs → 0
#   [99999, -88888]          [0.7, -0.4, 0.2]         [0.000, 0.000]
#
# COMMON STRATEGIES:
#   np.random.normal(0, 1/√n, size)    ← "Xavier" init (n = input size)
#   np.random.normal(0, √(2/n), size)  ← "He" init (for ReLU networks)
#
# EXAMPLE:
#   n_inputs = 100
#   # Xavier: std = 1/√n
#   weights = np.random.normal(0, 1/np.sqrt(n_inputs), size=(100, 64))
#   print(np.std(weights))   # ≈ 0.1
#
#   # He: std = √(2/n)
#   weights = np.random.normal(0, np.sqrt(2/n_inputs), size=(100, 64))
#   print(np.std(weights))   # ≈ 0.14
# ============================================================

# PROBLEM 2: Weight Initialization Simulator
# Simulate a simple 3-layer network (no real computation, just random weights).
#
# Layer sizes: input=256, hidden=128, output=10
#
# For each layer, initialize weights using:
#   a) Random normal (mean=0, std=1)          ← naive
#   b) Xavier init  (mean=0, std=1/√n_inputs) ← smart
#
# For each method, print:
#   - Weight matrix shape
#   - Mean and Std of the weights
#   - Simulated output range: multiply a fake input (np.ones) through
#     all 3 layers and print min/max of final output
#
# Compare: which method keeps outputs in a reasonable range?

np.random.seed(99)

# Layer sizes
n_input = 256
n_hidden = 128
n_output = 10

# Your code below:
