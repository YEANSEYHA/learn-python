import numpy as np

# ============================================================
# MODULE 2, TOPIC 1: NumPy Arrays - Creation, Shapes, dtypes
# ============================================================

# --- EXAMPLES (run this file to see output) ---

# Creating arrays from Python lists
a = np.array([1, 2, 3])              # 1D array (vector)
b = np.array([[1, 2, 3],
              [4, 5, 6]])            # 2D array (matrix)

print("1D array:", a)
print("2D array:\n", b)

# Shape tells you the dimensions
print("\na.shape:", a.shape)          # (3,) — 1D, 3 elements
print("b.shape:", b.shape)          # (2, 3) — 2 rows, 3 columns

# ndim = number of dimensions, size = total elements
print("b.ndim:", b.ndim)            # 2
print("b.size:", b.size)            # 6

# dtype = data type of elements
print("\na.dtype:", a.dtype)          # int64 (inferred from integers)
c = np.array([1.0, 2.0, 3.0])
print("c.dtype:", c.dtype)          # float64 (inferred from floats)

# Force a specific dtype — ML almost always uses float32
d = np.array([1, 2, 3], dtype=np.float32)
print("d.dtype:", d.dtype)          # float32

# --- Common creation functions ---
print("\n--- Creation Functions ---")
print("zeros:\n", np.zeros((2, 3)))          # 2x3 of zeros
print("ones:\n", np.ones((3, 2)))            # 3x2 of ones
print("full:\n", np.full((2, 2), 7))         # 2x2 filled with 7
print("arange:", np.arange(0, 10, 2))        # [0, 2, 4, 6, 8] like range()
print("linspace:", np.linspace(0, 1, 5))     # 5 evenly spaced from 0 to 1
print("eye:\n", np.eye(3))                   # 3x3 identity matrix

# --- Why this matters for ML ---
# A batch of 32 grayscale 28x28 images = shape (32, 28, 28)
# A batch of 32 RGB 224x224 images     = shape (32, 3, 224, 224)
# A weight matrix connecting 128 to 64 neurons = shape (128, 64)

batch = np.zeros((32, 28, 28))
print("\nImage batch shape:", batch.shape)
print("Total pixels:", batch.size)


# ============================================================
# PROBLEMS — solve below
# ============================================================

# Problem 1: Create a 3x4 array filled with the value 0.5, dtype float32.
#             Print the array, its shape, and its dtype.

datas = np.full((3,4),0.5 , dtype=np.float32)
print(datas)


# Problem 2: Create a 1D array of 10 evenly spaced values between -1 and 1.
#             Print it. (Hint: linspace)

problem2 = np.linspace(-1,1,10)
print(problem2)


# Problem 3: Create two arrays that represent:
#             - A weight matrix for a layer with 64 inputs and 32 outputs
#             - A bias vector for that layer's 32 outputs
#             Use zeros for both. Print their shapes.

weight = np.zeros((64,32))
bias = np.zeros(32)
print(weight.shape)


# Problem 4: Create a 3D array representing a mini-batch of 8 RGB images,
#             each 32x32 pixels. Print its shape, ndim, size, and dtype.

minibatch = np.zeros((8,32,32))