import numpy as np

# PROBLEM 1: Build an ROC Curve from Scratch
# Given actual labels and model confidence scores:
#   1. Loop through thresholds from 0.0 to 1.0 (step 0.1)
#   2. At each threshold, calculate TPR and FPR
#   3. Store them in two lists: tpr_list and fpr_list
#   4. Calculate AUC using np.trapz(tpr_list, fpr_list)
#   5. Print AUC and print each (FPR, TPR) pair
#
# Formulas:
#   TPR = TP / (TP + FN)
#   FPR = FP / (FP + TN)

actual = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1])
scores = np.array([0.95, 0.3, 0.85, 0.7, 0.2, 0.65, 0.4, 0.1, 0.9, 0.35, 0.55, 0.15, 0.75, 0.8, 0.25])

thresholds = np.arange(0.0, 1.1, 0.1)

# Your code below:
# Step1

tpr_list = []
fpr_list = []

for t in thresholds:

    # Get new predition with apply threshold
    prediction = (scores >= t).astype(int)
    print(prediction)
    tp = np.sum((actual ==1) & (prediction == 1))
    fp = np.sum((actual ==0) & (prediction == 1))
    fn = np.sum((actual == 1)& (prediction == 0))
    tn = np.sum((actual == 0) & (prediction == 0))

    tpr = tp / (tp+fn)
    fpr = fp / (fp+tn)
    tpr_list.append(tpr)
    fpr_list.append(fpr)


print(np.trapz(tpr_list,fpr_list))


# PROBLEM 2: Compare Two Models Using AUC
# Calculate AUC for both models. Print which model is better.

actual2 = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
model_a_scores = np.array([0.9, 0.2, 0.8, 0.7, 0.3, 0.1, 0.85, 0.15, 0.95, 0.4, 0.6, 0.05])
model_b_scores = np.array([0.7, 0.5, 0.6, 0.55, 0.45, 0.4, 0.65, 0.3, 0.8, 0.5, 0.5, 0.35])

thresholds2 = np.arange(0.0, 1.1, 0.1)

# Your code below:

tpr_list = []
fpr_list = []

for t in thresholds2:
    prediction = (model_a_scores >= t).astype(int)
    print(prediction)
    tp = np.sum((actual2 ==1) & (prediction == 1))
    fp = np.sum((actual2 ==0) & (prediction == 1))
    fn = np.sum((actual2 == 1)& (prediction == 0))
    tn = np.sum((actual2 == 0) & (prediction == 0))

    tpr = tp / (tp+fn)
    fpr = fp / (fp+tn)
    tpr_list.append(tpr)
    fpr_list.append(fpr)


print(np.trapz(tpr_list,fpr_list))



tpr_list = []
fpr_list = []

for t in thresholds2:
    prediction = (model_b_scores >= t).astype(int)
    print(prediction)
    tp = np.sum((actual2 ==1) & (prediction == 1))
    fp = np.sum((actual2 ==0) & (prediction == 1))
    fn = np.sum((actual2 == 1)& (prediction == 0))
    tn = np.sum((actual2 == 0) & (prediction == 0))

    tpr = tp / (tp+fn)
    fpr = fp / (fp+tn)
    tpr_list.append(tpr)
    fpr_list.append(fpr)


print(np.trapz(tpr_list,fpr_list))


# PROBLEM 3: Find the Best Threshold
# Find the threshold that gives the best balance of TPR and FPR.
# Best = highest (TPR - FPR). Print the best threshold, its TPR, and its FPR.

actual3 = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
scores3 = np.array([0.85, 0.4, 0.75, 0.6, 0.35, 0.7, 0.5, 0.1, 0.9, 0.25])

thresholds3 = np.arange(0.0, 1.1, 0.1)


results = []

# Your code below:
for t in thresholds3:
    prediction = (scores3 >= t).astype(int)
    print(prediction)

    tp = np.sum((actual3 ==1) & (prediction == 1))
    fp = np.sum((actual3 ==0) & (prediction == 1))
    fn = np.sum((actual3 == 1)& (prediction == 0))
    tn = np.sum((actual3 == 0) & (prediction == 0))

    tpr = tp / (tp+fn)
    fpr = fp / (fp+tn)
    
    result = {
        "t" : t,
        "tpr-fpr":tpr - fpr
    }

    results.append(result)

# print(f"Result TPR - FPR : {results}")

best = max(results, key=lambda x:x['tpr-fpr'])
print(best)
