# Conditionals - Practice
# Examples:
# if x > 0:
#     print("positive")
# elif x == 0:
#     print("zero")
# else:
#     print("negative")
#
# Logical: and, or, not
# Falsy values: 0, "", [], None, False

# ============================================
# Problem 1: Grade Checker
# ============================================
# Write a program that takes a score variable and prints the grade:
#   90 and above → "A"
#   80-89        → "B"
#   70-79        → "C"
#   60-69        → "D"
#   below 60     → "F"
#
# Test with: score = 85 (should print "B")

def check_score(score):
    if (score == 90 or score >90):
        print("A")
    elif (80 <= score <= 89):
        print("B")
    elif(70 <= score <=79):
        print("C")
    elif(60 <= score <= 69):
        print("D")
    else:
        print("F")

check_score(96)

# ============================================
# Problem 2: Even or Odd
# ============================================
# Write a function called `even_or_odd` that takes a number
# and returns "even" or "odd".
#
# Hint: use the modulo operator %
#   10 % 2 = 0 (even)
#   7 % 2 = 1 (odd)
#
# Test it:
#   print(even_or_odd(4))   # should print "even"
#   print(even_or_odd(7))   # should print "odd"

def even_or_odd(n):
    if n % 2 == 0:
        print("even")
    else:
        print("odd")

even_or_odd(2)

# ============================================
# Problem 3: Login Checker
# ============================================
# Write a function called `login` that takes a username and password.
# It should return:
#   "Welcome!" if username is "admin" AND password is "1234"
#   "Wrong password" if username is "admin" but wrong password
#   "User not found" for any other username
#
# Test it:
#   print(login("admin", "1234"))   # Welcome!
#   print(login("admin", "abcd"))   # Wrong password
#   print(login("guest", "1234"))   # User not found

def login(username, password):
    if(username == "admin" and password == "1234"):
        return f"Welcome!"
    elif(username == "admin" and password != "1234"):
        return f"Wrong password"
    else:
        return "User not found"

# ============================================
# Problem 4: Safe Divide
# ============================================
# Write a function called `safe_divide` that takes two numbers (a, b).
#   - If b is 0 (falsy), return "Cannot divide by zero"
#   - Otherwise, return a / b
#
# Test it:
#   print(safe_divide(10, 2))   # should print 5.0
#   print(safe_divide(10, 0))   # should print "Cannot divide by zero"

def safe_divide(a,b):
    if (b == 0):
        return "Cannot divide by zero"
    else:
        return a/b

