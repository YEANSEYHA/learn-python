# ============================================================
# LOOP PRACTICE — Module 1 | Python Foundations
# ============================================================
# Solve each problem using what you learned from the loop lesson.


# --- Problem 1: For Loop ---
# Given this list of temperatures in Celsius, print each one
# converted to Fahrenheit. Formula: F = C * 9/5 + 32
# Expected output:
# 0°C = 32.0°F
# 20°C = 68.0°F
# 37°C = 98.6°F
# 100°C = 212.0°F
celsius = [0, 20, 37, 100]
# YOUR CODE HERE:

for c in celsius:
    f = c * 9/5 +32
    print(f"{c}C = {f}f")


# --- Problem 2: Range + While ---
# Use a for loop with range to calculate the sum of numbers 1 to 10.
# Then print the result.
# Expected output: The sum of 1 to 10 is 55
# YOUR CODE HERE:
sum = 0
for i in range(1,11,1):
    sum = sum + i

print(sum)


# --- Problem 3: Enumerate + Zip ---
# You have a list of students and their scores.
# Print each student's rank, name, and score like:
# "Rank 1: Alice - 95"
# "Rank 2: Bob - 82"  etc.
# Hint: combine zip() and enumerate()
students = ["Alice", "Bob", "Charlie", "Diana"]
scores = [95, 82, 78, 91]
# YOUR CODE HERE:
for index, (student, score) in enumerate(zip(students, scores), 1):
    print(f"Rank {index}: {student} - {score}")


# --- Problem 4: List Comprehension + Condition ---
# Given a list of numbers, create a new list containing
# only the numbers greater than 50 using list comprehension.
# Then print the new list.
# Expected output: [72, 88, 95, 61]
numbers = [12, 72, 45, 88, 95, 33, 61, 8]
# YOUR CODE HERE:

n = [x for x in numbers if x >50]
print(n)
