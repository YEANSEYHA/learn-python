import numpy as np

# PROBLEM 1: Build a Confusion Matrix from Scratch
# Given actual labels and predicted labels, calculate and print:
#   1. The confusion matrix (2x2 grid)
#   2. Accuracy, Precision, Recall, F1 Score
#
# Use ONLY numpy — no sklearn.

actual    = np.array([1,1,0,1,0,0,1,1,0,1,0,0,1,0,1,1,0,1,0,0])
predicted = np.array([1,0,0,1,0,1,1,1,0,1,1,0,0,0,1,1,0,1,0,1])

# Your code below:

tp = np.sum((actual == 1) & (predicted == 1))
fn = np.sum((actual == 1) & (predicted == 0))
fp = np.sum((actual == 0) & (predicted == 1))
tn = np.sum((actual == 0) & (predicted == 0))

precision = tp/(tp+fp)
recall = tp/(tp+fn)
accuracy = (tp+ tn)/len(actual)
f1 = 2*(precision*recall)/(precision+recall)

maxtrix = np.array(
    [
        [tp,fn],
        [fp,tn]
    ]
)

print(precision)
print(recall)
print(f1)
print(maxtrix)


# PROBLEM 2: Threshold Tuning
# A model outputs confidence scores (0.0 to 1.0) for 15 patients.
# You decide the threshold: score >= threshold → predict positive (1)
#
# Try 3 thresholds: 0.3 (aggressive), 0.5 (default), 0.7 (cautious)
# For EACH threshold:
#   - Convert scores to predictions (1 if score >= threshold, else 0)
#   - Calculate Precision, Recall, and F1
#   - Print results
#
# Which threshold is best if this is a cancer test? (hint: don't miss sick patients)

actual_labels = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1])
model_scores  = np.array([0.9, 0.3, 0.8, 0.6, 0.2, 0.7, 0.4, 0.1, 0.95, 0.35, 0.55, 0.15, 0.8, 0.75, 0.3])

thresholds = [0.3, 0.5, 0.7]

# Your code below:
