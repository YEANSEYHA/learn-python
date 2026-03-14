# ============================================================
# LOOPS IN PYTHON — Module 1 | Python Foundations
# ============================================================


# ============================================================
# TOPIC 1: FOR LOOP
# ============================================================
# Definition:
# A for loop iterates over a sequence (list, string, range, etc.)
# and executes a block of code for each item in that sequence.
# In ML, you'll use this to loop through datasets, training
# samples, and model parameters.

# Example:
# fruits = ["apple", "banana", "cherry"]
# for fruit in fruits:
#     print(fruit)

# --- Problem 1 ---
# Print each item in this list on a separate line:
animals = ["cat", "dog", "fish", "bird"]
# YOUR CODE HERE:
for animal in animals:
    print(animal)


# ============================================================
# TOPIC 2: RANGE()
# ============================================================
# Definition:
# range() generates a sequence of numbers.
# range(stop)             → 0 to stop-1
# range(start, stop)      → start to stop-1
# range(start, stop, step) → start to stop-1, jumping by step
# Stop value is NOT included.

# Example:
# for i in range(0, 10, 2):
#     print(i)              # prints 0, 2, 4, 6, 8

# --- Problem 2 ---
# Print all multiples of 5 from 5 to 50 (5, 10, 15, ... 50)
# YOUR CODE HERE:
for i in range(5,51,5):
    print(i)


# ============================================================
# TOPIC 3: WHILE LOOP
# ============================================================
# Definition:
# A while loop keeps running as long as its condition is True.
# Use when you don't know how many iterations you need.

# Example:
# count = 0
# while count < 5:
#     print(f"count is {count}")
#     count += 1

# --- Problem 3 ---
# Start with number = 1. Keep doubling it (number *= 2) and
# print it each time. Stop BEFORE it goes above 100.
# Expected output: 1, 2, 4, 8, 16, 32, 64
# YOUR CODE HERE:
number = 1
while number  < 100:
    print(number)
    number = number * 2


# ============================================================
# TOPIC 4: ENUMERATE()
# ============================================================
# Definition:
# enumerate() gives you both the index and value while looping.
# Pass a second argument to change the starting index.

# Example:
# colors = ["red", "green", "blue"]
# for index, color in enumerate(colors, 1):
#     print(f"{index}: {color}")    # 1: red, 2: green, 3: blue

# --- Problem 4 ---
# Print each tool with its position starting from 1:
# "Item 1: Python"
# "Item 2: NumPy"  etc.
tools = ["Python", "NumPy", "Pandas", "PyTorch"]
# YOUR CODE HERE:
for index, tool in enumerate(tools,1):
    print(f"Item {index}: {tool}")


# ============================================================
# TOPIC 5: ZIP()
# ============================================================
# Definition:
# zip() combines two or more lists and loops through them
# together, pairing elements by position.

# Example:
# names = ["Alice", "Bob", "Charlie"]
# scores = [85, 92, 78]
# for name, score in zip(names, scores):
#     print(f"{name} scored {score}")

# --- Problem 5 ---
# Pair each fruit with its price and print:
# "apple costs $1.2"
# "banana costs $0.5"  etc.
fruits = ["apple", "banana", "cherry"]
prices = [1.2, 0.5, 2.0]
# YOUR CODE HERE:

for fruit, price in zip(fruits, prices):
    print(f"{fruit} costs ${price}")


# ============================================================
# TOPIC 6: BREAK AND CONTINUE
# ============================================================
# Definition:
# break    — exits the loop immediately
# continue — skips current iteration, moves to next

# Example:
# for i in range(10):
#     if i == 4:
#         break       # stops at 4, prints 0,1,2,3
#     print(i)

# --- Problem 6 ---
# Loop through numbers 1 to 20. Print each number, but
# stop as soon as you hit a number divisible by 7.
# Expected output: 1, 2, 3, 4, 5, 6, 7
# Hint: use % (modulo) to check divisibility
# YOUR CODE HERE:
for i in range(1,21,1):
    print(i)
    if i % 7 == 0:
        break


# ============================================================
# TOPIC 7: LIST COMPREHENSION
# ============================================================
# Definition:
# Creates a new list from a loop in one line.
# Format: [expression for item in sequence]
# With filter: [expression for item in sequence if condition]

# Example:
# squares = [x ** 2 for x in range(6)]    # [0, 1, 4, 9, 16, 25]
# evens = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]

# --- Problem 7 ---
# Create a list of cubes (x ** 3) for numbers 1 through 8
# using list comprehension. Then print the list.
# Expected output: [1, 8, 27, 64, 125, 216, 343, 512]
# YOUR CODE HERE:
cubes = [x ** 3 for x in range(1,9)]
print(cubes)
