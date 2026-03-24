import numpy as np

# PROBLEM 1: Calculate Entropy
# A model outputs these predicted probabilities for 4 classes.
# Calculate the entropy of each distribution using:
#   H(P) = -Σ p(x) * log2(p(x))
# Print the entropy for each and print which distribution
# has the HIGHEST uncertainty.

dist_a = np.array([0.25, 0.25, 0.25, 0.25])  # uniform
dist_b = np.array([0.9, 0.05, 0.03, 0.02])   # confident
dist_c = np.array([0.4, 0.3, 0.2, 0.1])      # somewhat spread

# Your code below:



dists = [dist_a , dist_b, dist_c]
result = []

for dist in dists:
    entropy = -np.sum(dist * np.log2(dist))
    print(entropy)
    data = {
        "dist": dist,
        "entropy": entropy
    }
    result.append(data)

print(result)
best = max(result, key= lambda x:x['entropy'])

print(best)


# PROBLEM 2: Calculate KL Divergence
# A true distribution P and a predicted distribution Q are given.
# Calculate KL(P || Q) using:
#   KL(P || Q) = Σ p(x) * log(p(x) / q(x))    (use natural log: np.log)
# Print the KL divergence.
# Then swap them: calculate KL(Q || P) and print it too.
# Are they the same? Print "Symmetric" or "Not symmetric".

p = np.array([0.7, 0.2, 0.1])
q = np.array([0.3, 0.5, 0.2])

# Your code below:
kl_divergence_pq = np.sum(p * np.log(p/q))
kl_divergence_qp = np.sum(q * np.log(q/p))

print(kl_divergence_pq)
print(kl_divergence_qp)

if kl_divergence_pq == kl_divergence_qp:
    print("Symmetric")
else:
    print("Not symmetric")


# PROBLEM 3: Cross-Entropy Loss
# Cross-entropy is THE loss function for classification.
# Formula: H(P, Q) = -Σ p(x) * log(q(x))    (use natural log: np.log)
#
# Given the true labels (P) and two models' predictions (Q1, Q2):
#   - Calculate cross-entropy for each model
#   - Print both values
#   - Print which model is BETTER (lower cross-entropy = better)

true_dist = np.array([1.0, 0.0, 0.0])  # true class is class 0

model_1 = np.array([0.8, 0.1, 0.1])    # pretty confident, correct
model_2 = np.array([0.4, 0.3, 0.3])    # unsure

# Your code below:

ce_1 = -np.sum(true_dist * np.log(model_1))

models = [model_1, model_2]
results = []
for model in models:
    ce = -np.sum(true_dist * np.log(model))
    print("ce :",ce)
    result = {
        "ce": ce,
        "model": model
    }
    results.append(result)

print(results)

best = min(results , key = lambda x:x['ce'])

print(best)
    