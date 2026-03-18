import numpy as np

# ============================================================
# MODULE 2, TOPIC 3: Reshaping & Broadcasting
# ============================================================

# --- RESHAPING EXAMPLES ---

a = np.arange(12)  # [0, 1, 2, ..., 11]
print("Original:", a, "| shape:", a.shape)

b = a.reshape(3, 4)  # 3 rows, 4 columns
print("Reshaped (3,4):\n", b)

c = a.reshape(2, 2, 3)  # 3D: 2 blocks of 2x3
print("Reshaped (2,2,3):\n", c)

# Use -1 to let NumPy figure out one dimension
d = a.reshape(4, -1)  # 4 rows, NumPy calculates cols → 3
print("reshape(4, -1):\n", d)

# .flatten() goes back to 1D
print("Flattened:", d.flatten())

# --- BROADCASTING EXAMPLES ---
print("\n--- Broadcasting ---")

# Matrix + vector: vector is added to EACH row
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
bias = np.array([10, 20, 30])
print("Matrix:\n", matrix)
print("Bias:", bias)
print("Matrix + Bias:\n", matrix + bias)

# Scalar broadcast: applies to every element
print("Matrix * 2:\n", matrix * 2)

# Column broadcast: need shape (2,1) to broadcast across columns
col = np.array([[100], [200]])  # shape (2, 1)
print("Matrix + column:\n", matrix + col)


# ============================================================
# PROBLEMS — solve below
# ============================================================

# Problem 1: Create a 1D array of numbers 1 through 16.
#             Reshape it into a 4x4 matrix. Print both shapes.

a = np.arange(1,17)
print(a.shape)
b = a.reshape(4,4)
print(b)
print(b.shape)

# Problem 2: Given this matrix, use broadcasting to:
scores = np.array([[80, 70, 90],
                   [60, 85, 75],
                   [95, 80, 70]])
# Each column is a different test. The weights for each test are:
weights = np.array([0.3, 0.3, 0.4])
# Multiply each test score by its weight and print the result.

print(scores * weights)

# Problem 3: You have a flattened image of 48 pixel values (1D).
#             Reshape it into an RGB image of shape (3, 4, 4)
#             meaning 3 channels, 4 rows, 4 columns.
#             Then flatten it back to 1D. Print shapes at each step.
flat_image = np.arange(48)

rgb_image = flat_image.reshape(3,4,4)
print(rgb_image)

reverse_image = rgb_image.flatten()
print(reverse_image.shape)


