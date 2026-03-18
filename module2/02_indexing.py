import numpy as np

# ============================================================
# MODULE 2, TOPIC 2: Indexing, Slicing, Boolean Masks
# ============================================================

# --- EXAMPLES ---

# 1D Indexing & Slicing
a = np.array([10, 20, 30, 40, 50])
print("Array:", a)
print("a[0]:", a[0])           # first element
print("a[-1]:", a[-1])         # last element
print("a[1:4]:", a[1:4])       # index 1,2,3 (end excluded)
print("a[:3]:", a[:3])         # first 3 elements
print("a[::2]:", a[::2])       # every other element

# 2D Indexing & Slicing
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print("\n2D Array:\n", b)
print("b[0, 2]:", b[0, 2])       # row 0, col 2 → 3
print("b[1, :]:", b[1, :])       # entire row 1 → [4, 5, 6]
print("b[:, 1]:", b[:, 1])       # entire col 1 → [2, 5, 8]
print("b[:2, :2]:\n", b[:2, :2]) # top-left 2x2 block

# Boolean Masking — filter by condition
print("\n--- Boolean Masking ---")
scores = np.array([45, 82, 67, 91, 55, 73])
print("Scores:", scores)
print("scores > 70:", scores > 70)          # [False, True, False, True, False, True]
print("Passing:", scores[scores > 70])      # [82, 91, 73] — only values > 70

# Real ML use: filter out negative values (like ReLU does!)
data = np.array([-2, 3, -1, 5, 0, -4, 7])
print("\nData:", data)
print("Positive only:", data[data > 0])     # [3, 5, 7]


# ============================================================
# PROBLEMS — solve below
# ============================================================

# Problem 1: Given this array of model predictions:
predictions = np.array([0.12, 0.89, 0.45, 0.93, 0.31, 0.78, 0.05, 0.67])
# a) Print the first 3 predictions
print(predictions[:3])
# b) Print the last 2 predictions
print(predictions[-2:])
# c) Print every other prediction starting from index 0
print(predictions[::2])

# Problem 2: Given this 2D array (3 students, 4 test scores each):
grades = np.array([[85, 92, 78, 90],
                   [70, 65, 80, 72],
                   [95, 88, 91, 97]])
# a) Print student 2's scores (row index 2)
print(grades[1])
# b) Print all students' scores on test 1 (column index 1)
print(grades[:, 1])  # → [92, 65, 88]

# c) Print the top-left 2x2 block (first 2 students, first 2 tests)
print(grades[:2,:2])

# Problem 3: Boolean Masking
data = np.array([3, -1, 7, -4, 2, -8, 5, 0, -3, 6])
# a) Print only the negative values
print(data[data < 0])
# b) Print only values greater than 4
print(data[data > 4])
# c) Replace all negative values with 0 and print (hint: data[condition] = value)
data[data < 0] = 0
print(data)
