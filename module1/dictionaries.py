# Dictionaries - Practice
# Examples:
# person = {"name": "Matin", "age": 25}
# person["name"]          → "Matin"
# person.get("gpa", 0)    → 0 (default)
# person["gpa"] = 3.8     → add new key
# for key, value in person.items(): ...

# ============================================
# Problem 1: Build a Profile
# ============================================
# Create a dictionary called `profile` with these keys:
#   "name", "age", "language", "level"
# Fill in your own values.
# Then print each key and value using a loop.

profile = {
    "name": "Ben walker",
    "age": "26",
    "language": "Khmer",
    "level": "999"
}

# for key , value in profile.items():
#     print(f"{key} : {value}")

# ============================================
# Problem 2: Word Counter
# ============================================
# Given a list of words, count how many times each word appears.
# Store the result in a dictionary.
#
# words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
#
# Expected output: {"apple": 3, "banana": 2, "cherry": 1}
#
# Hint: check if a word is already in the dict.
#   if word in counts:  ...
#   else: ...
words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
counts = {}

for word in words:
    if word in counts:
        counts[word] += 1
    else:
        counts[word] = 1

print(counts)

# ============================================
# Problem 3: Student Grades
# ============================================
# Given this nested dictionary:
students = {
    "Alice": {"math": 90, "science": 85},
    "Bob": {"math": 70, "science": 95},
    "Charlie": {"math": 80, "science": 75}
}
# Write a loop that prints each student's name and their AVERAGE score.
#
# Expected output:
#   Alice: 87.5
#   Bob: 82.5
#   Charlie: 77.5

for key, value in students.items():
    average_score = (value["math"] + value["science"])/2
    print(f"{key} : {average_score}")

# ============================================
# Problem 4: Dictionary Comprehension
# ============================================
# Just like list comprehensions, dicts have them too:
#   {key: value for item in iterable}
#
# Example: squares = {x: x**2 for x in range(5)}
#   → {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
#
# Given this list:
fruits = ["apple", "banana", "cherry", "date"]
# Create a dictionary where each fruit is the key
# and the value is the LENGTH of that fruit's name.
#
# Expected: {"apple": 5, "banana": 6, "cherry": 6, "date": 4}
# Hint: use len() and a dict comprehension
