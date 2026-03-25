import numpy as np

# ============================================================
# MODULE 2, TOPIC 2: Indexing, Slicing, Boolean Masks
# ============================================================

# --- STEP 1: 1D Indexing ---
# An array is like a row of boxes, each with a number (index)
#
#   index:   0    1    2    3    4
# array: [ 10 , 20 , 30 , 40 , 50 ]
#          ↑                    ↑
#        a[0]=10             a[4]=50
#
#   Negative index counts from the end:
#   index:  -5   -4   -3   -2   -1
# array: [ 10 , 20 , 30 , 40 , 50 ]
#                              ↑
#                          a[-1]=50

# --- Example ---
a = np.array([10, 20, 30, 40, 50])
print("Array:", a)
print("a[0]:", a[0])       # → 10
print("a[2]:", a[2])       # → 30
print("a[-1]:", a[-1])     # → 50

# --- Exercise ---
# Given this array of model accuracies for 5 experiments:
acc = np.array([0.72, 0.85, 0.91, 0.68, 0.79])
# Print the FIRST accuracy
# Print the THIRD accuracy
# Print the LAST accuracy (use negative index)

# Your code below:
print(acc[0])

print(acc[2])

print(acc[-1])


# --- STEP 2: 1D Slicing ---
# Slicing grabs a RANGE of items: array[start:stop]
# start is INCLUDED, stop is EXCLUDED
#
#   index:   0    1    2    3    4
# array: [ 10 , 20 , 30 , 40 , 50 ]
#
# a[1:4] → items at index 1, 2, 3 → [20, 30, 40]
#           starts at 1, stops BEFORE 4
#
# Shortcuts:
#   a[:3]  → from beginning to index 2  → [10, 20, 30]
#   a[2:]  → from index 2 to the end    → [30, 40, 50]

# --- Example ---
a = np.array([10, 20, 30, 40, 50])
print("\n--- Slicing ---")
print("a[1:4]:", a[1:4])   # → [20, 30, 40]
print("a[:3]:", a[:3])     # → [10, 20, 30]
print("a[2:]:", a[2:])     # → [30, 40, 50]

# --- Exercise ---
# Given these 6 test scores:
scores = np.array([88, 72, 95, 61, 84, 77])
# Print the first 3 scores
# Print the last 3 scores
# Print scores at index 1, 2, 3 (middle section)

# Your code below:
print(scores[:3])
print(scores[3:])
print(scores[1:4])


# --- STEP 3: Step Slicing ---
# Full syntax: array[start:stop:step]
# step = how many to skip between picks
#
#   index:   0    1    2    3    4    5
# array: [ 10 , 20 , 30 , 40 , 50 , 60 ]
#
# a[0::2] → start at 0, pick every 2nd → [10, 30, 50]
#            0  skip  2  skip  4
#
# a[1::2] → start at 1, pick every 2nd → [20, 40, 60]
#            1  skip  3  skip  5
#
# a[::-1] → step -1 = REVERSE the whole array → [60, 50, 40, 30, 20, 10]

# --- Example ---
a = np.array([10, 20, 30, 40, 50, 60])
print("\n--- Step Slicing ---")
print("a[0::2]:", a[0::2])   # → [10, 30, 50]  (every 2nd, start at 0)
print("a[1::2]:", a[1::2])   # → [20, 40, 60]  (every 2nd, start at 1)
print("a[::-1]:", a[::-1])   # → [60, 50, 40, 30, 20, 10]  (reversed)

# --- Exercise ---
# Given these 8 training losses:
losses = np.array([2.5, 2.1, 1.8, 1.5, 1.2, 0.9, 0.6, 0.3])
#  index:            0    1    2    3    4    5    6    7
# Print every other loss starting from index 0 (0, 2, 4, 6)
# Print every other loss starting from index 1 (1, 3, 5, 7)
# Print the losses in reverse order

# Your code below:
print(losses[0::2])
print(losses[1::2])
print(losses[::-1])


# --- STEP 4: 2D Indexing ---
# A 2D array has ROWS and COLUMNS
# Access with: array[row, column]
#
#            col 0  col 1  col 2
# row 0  [[  1  ,   2  ,   3  ],
# row 1   [  4  ,   5  ,   6  ],
# row 2   [  7  ,   8  ,   9  ]]
#
# b[0, 2] → row 0, col 2 → 3
# b[1, 1] → row 1, col 1 → 5
#
# Grab a whole ROW:    b[1, :]  → [4, 5, 6]    (row 1, ALL columns)
# Grab a whole COLUMN: b[:, 2]  → [3, 6, 9]    (ALL rows, col 2)
#
# The comma separates [rows, columns]
# The colon : means "all"

# --- Example ---
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print("\n--- 2D Indexing ---")
print("b[0, 2]:", b[0, 2])    # → 3
print("b[1, :]:", b[1, :])    # → [4, 5, 6]  (row 1, all cols)
print("b[:, 2]:", b[:, 2])    # → [3, 6, 9]  (all rows, col 2)

# --- Exercise ---
# Given 3 students' scores on 4 tests:
grades = np.array([[85, 92, 78, 90],
                   [70, 65, 80, 72],
                   [95, 88, 91, 97]])
#          test:    0   1   2   3
# student 0:      85  92  78  90
# student 1:      70  65  80  72
# student 2:      95  88  91  97
#
# Print student 1's all scores (row 1, all columns)
# Print everyone's test 3 scores (all rows, column 3)
# Print the single value: student 2, test 1

# Your code below:
print(grades[1, :])
print(grades[:,3])
print(grades[2,1])


# --- STEP 5: 2D Slicing ---
# You can use slicing on EACH dimension separately!
# array[row_slice, col_slice]
#
#            col 0  col 1  col 2
# row 0  [[  1  ,   2  ,   3  ],
# row 1   [  4  ,   5  ,   6  ],
# row 2   [  7  ,   8  ,   9  ]]
#
# b[:2, :2]   → rows 0,1 + cols 0,1 → [[1,2],[4,5]]   (top-left block)
# b[:, ::-1]  → all rows + reverse cols → [[3,2,1],[6,5,4],[9,8,7]]
# b[::-1]     → reverse rows → [[7,8,9],[4,5,6],[1,2,3]]
#
# Think of it as: [what to do with rows, what to do with columns]

# --- Example ---
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print("\n--- 2D Slicing ---")
print("Top-left 2x2:\n", b[:2, :2])     # rows 0-1, cols 0-1
print("Reverse cols:\n", b[:, ::-1])     # all rows, flip columns
print("Reverse rows:\n", b[::-1])        # flip rows, keep columns

# --- Exercise ---
# Same grades array:
# grades = [[85, 92, 78, 90],
#            [70, 65, 80, 72],
#            [95, 88, 91, 97]]
#
# Print the top-left 2x3 block (first 2 students, first 3 tests)
# Print all grades with columns reversed (test 3 first, test 0 last)
# Print all grades with rows reversed (student 2 first, student 0 last)

# Your code below:

# print(grades[:2,:3])
# print(grades[:,::-1])
print(grades[::-1,:])

