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
