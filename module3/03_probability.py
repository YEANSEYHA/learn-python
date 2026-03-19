import numpy as np

# ============================================================
# TOPIC: Conditional & Joint Probability
# ============================================================
#
# WHAT:
#   Probability      = how likely something happens (0 to 1)
#   Joint Prob       = P(A AND B) — both events happen together
#   Conditional Prob = P(A | B)   — probability of A, GIVEN B already happened
#
# WHY FOR ML:
#   - Bayes classifiers: "what's P(spam | contains 'free money')?"
#   - Neural networks learn P(output | input)
#   - Every prediction is a conditional probability
#
# KEY FORMULAS:
#   P(A)           = count(A) / total
#   P(A AND B)     = count(A and B) / total
#   P(A | B)       = P(A AND B) / P(B)
#
# VISUAL:
#
#   Imagine 100 emails:
#
#                    has "free"?
#                    YES     NO
#                  ┌──────┬──────┐
#   SPAM     YES   │  20  │  10  │  30  (spam total)
#                  ├──────┼──────┤
#            NO    │   5  │  65  │  70  (not spam total)
#                  └──────┴──────┘
#                    25      75     100
#
#   Read from the table:
#   P(spam)                = 30 / 100 = 0.30
#   P(has "free")          = 25 / 100 = 0.25
#   P(spam AND "free")     = 20 / 100 = 0.20   ← top-left cell / total
#   P(spam | "free")       = 20 / 25  = 0.80   ← top-left cell / column total
#                                        ↑ if it has "free", 80% chance spam!
#
# EXAMPLE:
#   # Simulating with data
#   total = 1000
#   is_spam = np.random.choice([0, 1], size=total, p=[0.7, 0.3])
#   has_free = np.where(is_spam == 1,
#                       np.random.choice([0, 1], size=total, p=[0.3, 0.7]),
#                       np.random.choice([0, 1], size=total, p=[0.9, 0.1]))
#
#   p_spam = np.mean(is_spam)                              # ≈ 0.30
#   p_free = np.mean(has_free)                             # ≈ 0.28
#   p_spam_and_free = np.mean((is_spam == 1) & (has_free == 1))  # ≈ 0.21
#   p_spam_given_free = p_spam_and_free / p_free           # ≈ 0.75
# ============================================================

# PROBLEM 1: Medical Test Probability
# A hospital tested 10,000 patients for a disease.
# The data arrays below tell you:
#   has_disease: 1 = has disease, 0 = healthy
#   test_result: 1 = tested positive, 0 = tested negative
#
# Calculate and print:
#   1. P(disease)              — what fraction are actually sick?
#   2. P(positive)             — what fraction tested positive?
#   3. P(disease AND positive) — sick AND tested positive
#   4. P(disease | positive)   — if someone tests positive, what's the
#                                 chance they're actually sick?
#   5. P(positive | disease)   — if someone IS sick, what's the chance
#                                 the test catches it? (sensitivity)
#   6. P(positive | no disease) — false positive rate

np.random.seed(42)
n = 10000

has_disease = np.random.choice([0, 1], size=n, p=[0.95, 0.05])
test_result = np.where(has_disease == 1,
                       np.random.choice([0, 1], size=n, p=[0.1, 0.9]),
                       np.random.choice([0, 1], size=n, p=[0.92, 0.08]))

# Your code below:
