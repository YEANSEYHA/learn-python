import numpy as np

# ============================================================
# MODULE 2, TOPIC 4: Math Operations
# ============================================================

# --- ELEMENT-WISE OPERATIONS ---
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("a + b:", a + b)       # [5, 7, 9]
print("a * b:", a * b)       # [4, 10, 18]  (NOT dot product)
print("a ** 2:", a ** 2)     # [1, 4, 9]    (squared)
print("np.sqrt(a):", np.sqrt(a))  # [1.0, 1.41, 1.73]

# --- DOT PRODUCT ---
print("\n--- Dot Product ---")
print("np.dot(a, b):", np.dot(a, b))  # 1*4 + 2*5 + 3*6 = 32

# --- HOW A NEURON WORKS ---
print("\n--- Single Neuron ---")
inputs = np.array([0.5, 0.3, 0.8])
weights = np.array([0.2, 0.7, 0.1])
bias = 0.1

output = np.dot(inputs, weights) + bias
print("Neuron output:", output)  # 0.49

# --- USEFUL MATH FUNCTIONS ---
print("\n--- Useful Functions ---")
data = np.array([1, -2, 3, -4, 5])
print("Sum:", np.sum(data))       # 3
print("Mean:", np.mean(data))     # 0.6
print("Abs:", np.abs(data))       # [1, 2, 3, 4, 5]
print("Max:", np.max(data))       # 5
print("Argmax:", np.argmax(data)) # 4 (INDEX of max value)

# argmax is how you get a model's prediction:
# e.g., output = [0.1, 0.7, 0.2] → argmax = 1 → class 1


# ============================================================
# PROBLEMS — solve below
# ============================================================

# Problem 1: Simulate a neuron with 4 inputs.
#             Create your own inputs (4 values between 0 and 1)
#             and weights (4 values). Pick a bias.
#             Calculate the output using dot product + bias. Print it.

inputs = np.array([0,0.2,0.5,0.8])
weights = np.array([0.2,0.3,0.6,0.1])
bias = 0.2

output = np.dot(inputs, weights) + bias
print(output)

# Problem 2: Given these model outputs for 5 samples across 3 classes:
outputs = np.array([[0.2, 0.7, 0.1],
                    [0.8, 0.1, 0.1],
                    [0.3, 0.3, 0.4],
                    [0.1, 0.2, 0.7],
                    [0.6, 0.3, 0.1]])
# a) Use argmax to get the predicted class for EACH sample (hint: axis=1)
# print(outputs)
# for output in outputs:
#     print(np.argmax(output))
print(np.argmax(outputs, axis=1))
# b) Print the max confidence for each sample
print(np.max(outputs,axis=1))

# Problem 3: Normalize this data so values range from 0 to 1.
#             Formula: normalized = (data - min) / (max - min)
#             Use np.min() and np.max(). Print original and normalized.
data = np.array([20, 45, 10, 80, 55, 30])

normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
print("Original:", data)
print("Normalized:", normalized)

