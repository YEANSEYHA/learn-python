# Functions - Practice
# Examples:
# def greet(name):
#     return "Hello, " + name
#
# def add(a, b):
#     return a + b
#
# def power(base, exp=2):
#     return base ** exp
#
# def min_max(numbers):
#     return min(numbers), max(numbers)

# ============================================
# Problem 1: Square a Number
# ============================================
# Write a function called `square` that takes a number
# and returns it multiplied by itself.
#
# Test it:
#   print(square(4))   # should print 16
#   print(square(7))   # should print 49


def square(n):
    return n*n

print(square(1000))

# ============================================
# Problem 2: Greeting with Default
# ============================================
# Write a function called `greet` that takes a name
# and a greeting with a default value of "Hello".
# It should return "{greeting}, {name}!"
#
# Test it:
#   print(greet("Alice"))            # should print "Hello, Alice!"
#   print(greet("Bob", "Goodbye"))   # should print "Goodbye, Bob!"

def greet(name, message="Hello"):
    return f"{message}, {name}!"

print(greet("Matin","Good Night"))

# ============================================
# Problem 3: Return Multiple Values
# ============================================
# Write a function called `stats` that takes a list of numbers
# and returns THREE values: the sum, the minimum, and the maximum.
#
# Hint: use the built-in sum(), min(), max() functions.
#
# Test it:
#   total, lo, hi = stats([10, 20, 5, 40])
#   print(total)  # should print 75
#   print(lo)     # should print 5
#   print(hi)     # should print 40

def stats(list):
    total = sum(list)
    lo = min(list)
    hi = max(list)
    return total, lo, hi

total, lo, hi = stats([200, -99, 200,0, 142])

print(total)

# ============================================
# Problem 4: Lambda (One-line Functions)
# ============================================
# A lambda is a mini function written in one line:
#   double = lambda x: x * 2
#   print(double(5))  # 10
#
# Write a lambda called `cube` that returns x to the power of 3.
# Then test it:
#   print(cube(3))   # should print 27
#   print(cube(5))   # should print 125

cube = lambda x : x * x *x
print(cube(3))

