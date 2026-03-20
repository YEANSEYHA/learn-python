import numpy as np

# ============================================================
# TOPIC: Bayes' Rule
# ============================================================
#
# WHAT:
#   Bayes' Rule lets you FLIP a conditional probability.
#   You know P(B | A) → you want P(A | B)
#
# WHY FOR ML:
#   - Spam filters: you know how often spam contains "free"
#     → but you WANT: if email has "free", is it spam?
#   - Medical AI: you know test accuracy
#     → but you WANT: if test is positive, is patient sick?
#   - Every classifier is doing Bayes behind the scenes
#
# THE FORMULA:
#
#                    P(B | A) * P(A)
#   P(A | B)  =  ─────────────────────
#                        P(B)
#
# VISUAL — Think of it as flipping the question:
#
#   You KNOW:                        You WANT:
#   P(positive | sick) = 0.90        P(sick | positive) = ???
#   "90% of sick people test +"      "if test is +, am I sick?"
#
#   These are NOT the same number!
#
#   Bayes connects them:
#
#              P(positive | sick) * P(sick)
#   P(sick | positive) = ─────────────────────────
#                               P(positive)
#
#              0.90 * 0.05
#            = ──────────── = 0.045 / 0.13 = 0.35
#                 0.13
#
#   Only 35% chance you're sick even with a positive test!
#   (because the disease is rare — most positives are false alarms)
#
# EXAMPLE:
#   p_sick = 0.05
#   p_positive_given_sick = 0.90
#   p_positive_given_healthy = 0.08
#
#   # P(positive) = can be positive two ways: sick+positive OR healthy+positive
#   p_positive = p_positive_given_sick * p_sick + p_positive_given_healthy * (1 - p_sick)
#   # = 0.90 * 0.05 + 0.08 * 0.95 = 0.045 + 0.076 = 0.121
#
#   p_sick_given_positive = (p_positive_given_sick * p_sick) / p_positive
#   # = (0.90 * 0.05) / 0.121 = 0.372
# ============================================================


# PROBLEM 1: Plug into Bayes (just the formula)
# ============================================================
# A school gives a math quiz to detect students who need extra help.
#
# You know:
#   P(struggling)                    = 0.20   ← 20% of students struggle
#   P(fail quiz | struggling)       = 0.85   ← 85% of struggling students fail
#   P(fail quiz | not struggling)   = 0.10   ← 10% of good students also fail
#
# Question: If a student failed the quiz, what's the probability
#           they are actually struggling?
#           → P(struggling | fail quiz) = ???
#
# Steps:
#   1. Calculate P(fail quiz)   ← can fail two ways: struggling+fail OR good+fail
#   2. Plug into Bayes formula

p_struggling = 0.20
p_fail_given_struggling = 0.85
p_fail_given_good = 0.10

# Print: "P(struggling | failed) = ___"
# Your code below:

p_fail = p_fail_given_struggling * p_struggling + p_fail_given_good * (1 - p_struggling)

print(p_fail)

p_struggling_given_fail = (p_fail_given_struggling * p_struggling) / p_fail

print(p_struggling_given_fail)


# PROBLEM 2: Bayes with NumPy arrays
# ============================================================
# Now instead of given numbers, you calculate from real data
# and THEN apply Bayes.
#
# A factory makes 1000 products.
#   defective: 1 = defective, 0 = good
#   flagged:   1 = machine flagged it, 0 = not flagged
#
# Question: If a product is flagged, what's the chance it's
#           actually defective?
#           → P(defective | flagged) = ???
#
# Steps:
#   1. Find P(defective)               ← np.mean(defective)
#   2. Find P(flagged)                 ← np.mean(flagged)
#   3. Find P(defective AND flagged)   ← np.mean((defective==1) & (flagged==1))
#   4. Bayes: P(defective | flagged)   ← step 3 / step 2

np.random.seed(42)
n = 1000
defective = np.random.choice([0, 1], size=n, p=[0.90, 0.10])
flagged = np.where(defective == 1,
                   np.random.choice([0, 1], size=n, p=[0.15, 0.85]),
                   np.random.choice([0, 1], size=n, p=[0.93, 0.07]))

print("defective",defective)
print("Flagged",flagged)

# Print: "P(defective | flagged) = ___"
# Your code below:
p_defective = np.mean(defective)
print(p_defective)
p_flagged = np.mean(flagged)
print(p_flagged)

# 3
p_d_f = np.mean((defective == 1) & (flagged ==1))
print(p_d_f)

p_or_f  = p_d_f / p_flagged

print(p_or_f)


