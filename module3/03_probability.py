import numpy as np

# ============================================================
# TOPIC: Probability (Beginner Friendly)
# ============================================================
#
# 3 formulas. That's all.
#
#   1. P(A)         = count of A / total       "how common is A?"
#   2. P(A AND B)   = count of both / total    "how often do both happen?"
#   3. P(A | B)     = P(A AND B) / P(B)        "if B happened, how likely is A?"
#
# NumPy trick:
#   np.mean(array)  on 0s and 1s = probability!
#   Because: (0+0+1+0+1) / 5 = 2/5 = 0.40 = P(A)
#
# ============================================================


# PROBLEM 1: Basic Probability — P(A)
# ============================================================
# A classroom has 20 students. Each student either passed (1) or failed (0).
#
# Calculate:  What fraction of students passed?
#
# VISUAL:
#   students: [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1]
#             pass     fail
#             ████████████   ██████
#              14 passed     6 failed     → P(pass) = 14/20 = 0.70

results = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1])

# Print: "P(pass) = ___"
# Your code below:

pa = np.mean(results)
print(pa)


# PROBLEM 2: Joint Probability — P(A AND B)
# ============================================================
# A class has 20 students. We know two things about each:
#   passed:  1 = passed the exam, 0 = failed
#   studied: 1 = studied,         0 = didn't study
#
# Calculate: What fraction both studied AND passed?
#
# VISUAL:
#                        passed?
#                      YES     NO
#                    ┌──────┬──────┐
#   studied?  YES    │  10  │   2  │  12
#                    ├──────┼──────┤
#             NO     │   4  │   4  │   8
#                    └──────┴──────┘
#                      14      6      20
#
#   P(studied AND passed) = 10 / 20 = 0.50
#
# HINT: Use & to combine two conditions:
#   np.mean((array1 == 1) & (array2 == 1))

passed  = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1])
studied = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1])

# Print: "P(studied AND passed) = ___"
# Your code below:

p_study_pass = np.mean((passed == 1) & (studied ==1))
print(f"P(studied AND passed) {p_study_pass}")


# PROBLEM 3: Conditional Probability — P(A | B)
# ============================================================
# Using the SAME passed and studied arrays from Problem 2.
#
# Question: If a student STUDIED, what's the chance they passed?
#           → P(passed | studied)
#
# VISUAL:
#   All 20 students
#   ┌──────────────────────────────────────┐
#   │ o o o o o o o o o o o o o o o o o o o│
#   └──────────────────────────────────────┘
#           ↓ filter: only students who studied (12)
#   ┌─────────────────────────┐
#   │ o o o o o o o o o o o o │
#   └─────────────────────────┘
#           ↓ of those, how many passed? (10)
#   ┌────────────────────┐
#   │ o o o o o o o o o o│
#   └────────────────────┘
#
#   P(passed | studied) = 10 / 12 = 0.8333
#
# FORMULA:
#   P(A | B) = P(A AND B) / P(B)
#
# You already have P(A AND B) from Problem 2.
# You just need P(studied), then divide.

# Print: "P(passed | studied) = ___"
# Your code below:

p_studied = np.mean(studied == 1)
print(p_studied)

p_a_b = p_study_pass/p_studied

print(p_a_b)

