# ============================================
# MODULE 6: Perceptron & Neurons
# Book: Ch 13 (Neural Networks)
# ============================================

# --- WHAT IS A NEURON? ---
# A neuron is the building block of every neural network.
# It takes inputs, multiplies each by a weight, adds a bias, then passes
# the result through an activation function to produce an output.

# --- VISUAL ---
"""
    A single neuron:

    x1 ──(w1)──┐
    x2 ──(w2)──┤──→ [ sum: w1x1 + w2x2 + b ] ──→ [ activation ] ──→ output
    x3 ──(w3)──┘

    Step by step:
    1. Each input x is multiplied by its weight w
    2. Add them all up + bias b  →  z = w1*x1 + w2*x2 + w3*x3 + b
    3. Pass z through activation function  →  output = f(z)

    The PERCEPTRON is the simplest neuron:
    - activation = step function (output is 0 or 1)
    - If z >= 0 → output 1  (fires)
    - If z <  0 → output 0  (silent)

    WEIGHTS control how important each input is.
    BIAS shifts the threshold — like "how hard is it to activate this neuron?"
"""

# --- EXAMPLE: A single neuron from scratch ---
import numpy as np

def neuron(inputs, weights, bias):
    z = np.dot(inputs, weights) + bias   # weighted sum
    return z                              # raw output (before activation)

# A neuron with 3 inputs
inputs  = np.array([1.5, 2.0, -1.0])
weights = np.array([0.4, 0.3,  0.8])
bias    = 0.5

z = neuron(inputs, weights, bias)
print(f"Inputs:  {inputs}")
print(f"Weights: {weights}")
print(f"Bias:    {bias}")
print(f"z (weighted sum + bias): {z:.4f}")

# Perceptron: step activation
output = 1 if z >= 0 else 0
print(f"Perceptron output: {output}  (1=fires, 0=silent)")


# --- EXAMPLE 2: A layer of neurons ---
# In a real network, we have MANY neurons at once.
# Each neuron has its own weights — stored as rows in a matrix.
"""
    Layer with 3 neurons, each taking 3 inputs:

    inputs (1x3) × weights (3x3) + bias (1x3) = output (1x3)

    Each ROW in the weight matrix = one neuron's weights
"""

inputs  = np.array([1.5, 2.0, -1.0])        # 1 sample, 3 features
weights = np.array([[ 0.4,  0.3,  0.8],      # neuron 1 weights
                    [-0.2,  0.9,  0.1],       # neuron 2 weights
                    [ 0.7, -0.4,  0.5]])      # neuron 3 weights
biases  = np.array([0.5, -0.1, 0.3])

z = np.dot(weights, inputs) + biases   # shape: (3,)
print(f"\nLayer output (3 neurons): {z}")


# ============================================
# PRACTICE: Your exercises go below
# ============================================

# --- Problem 1: Build a neuron by hand ---
# A neuron has 4 inputs:
#   inputs  = [2.0, -1.0, 0.5, 3.0]
#   weights = [0.3,  0.5, 0.8, -0.2]
#   bias    = 1.0
#
# Your task:
#   1. Compute z = weighted sum + bias (use np.dot)
#   2. Apply step activation: output = 1 if z >= 0 else 0
#   3. Print z and the output

inputs  = [2.0, -1.0, 0.5, 3.0]
weights = [0.3,  0.5, 0.8, -0.2]
bias    = 1.0
# 1 Compute Z
z = np.dot(inputs ,weights) + bias

print(f"z : {z}")

# Perceptron: step activation

output = 1 if z >= 0 else 0

print(f"Output {output}")


# --- Problem 2: Layer of neurons ---
# Build a layer of 4 neurons, each taking 3 inputs.
#
#   inputs  = [1.0, 2.0, 3.0]
#   weights = [[ 0.1,  0.2, -0.3],   ← neuron 1
#              [ 0.4, -0.1,  0.5],   ← neuron 2
#              [-0.2,  0.3,  0.1],   ← neuron 3
#              [ 0.6,  0.0, -0.4]]   ← neuron 4
#   biases  = [0.1, -0.2, 0.3, 0.0]
#
# Your task:
#   1. Compute z for all 4 neurons at once using np.dot(weights, inputs) + biases
#   2. Print z — it should have 4 values (one per neuron)

inputs  = [1.0, 2.0, 3.0]
weights = [[ 0.1,  0.2, -0.3],  
             [ 0.4, -0.1,  0.5],  
             [-0.2,  0.3,  0.1],   
             [ 0.6,  0.0, -0.4]]   
biases  = [0.1, -0.2, 0.3, 0.0]


z = np.dot(weights, inputs) + biases
print("Z2 :",z)


# --- Problem 3: Perceptron as a logic gate (AND gate) ---
# A perceptron can learn logical rules. An AND gate outputs 1 only when BOTH inputs are 1.
#
# Truth table:
#   x1=0, x2=0 → 0
#   x1=0, x2=1 → 0
#   x1=1, x2=0 → 0
#   x1=1, x2=1 → 1
#
# Use these fixed weights and bias that implement AND:
#   weights = [1.0, 1.0]
#   bias    = -1.5
#
# Your task:
#   1. Loop through all 4 input combinations above
#   2. Compute z = np.dot(inputs, weights) + bias
#   3. Apply step activation: output = 1 if z >= 0 else 0
#   4. Print each input pair and its output — should match the truth table

inputs_all = [
    [0, 0],   # → 0
    [0, 1],   # → 0
    [1, 0],   # → 0
    [1, 1],   # → 1
]
weights = [1.0, 1.0]
bias    = -1.5
for inputs in inputs_all:
    z = np.dot(inputs, weights) + bias
    output = 1 if z >= 0 else 0
    print(f"x1={inputs[0]}, x2={inputs[1]} → output={output}")
