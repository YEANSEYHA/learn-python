import numpy as np

# PROBLEM 1: Dataset Summary
# You have exam scores from 3 classes. For EACH class:
# - Print the mean, median, and standard deviation
# - Print which class has the highest average
# - Print which class is most consistent (lowest std dev)

class_a = np.array([85, 90, 78, 92, 88, 76, 95, 82, 89, 91])
class_b = np.array([70, 95, 60, 100, 55, 98, 62, 97, 58, 95])
class_c = np.array([80, 82, 79, 81, 83, 78, 84, 80, 81, 82])

# Your code below:
mean_a = np.mean(class_a)
median_a = np.median(class_a)
std_a = np.std(class_a)
print(f"Class A - Mean: {mean_a}, Median: {median_a}, Std: {std_a:.2f}")

mean_b = np.mean(class_b)
median_b = np.median(class_b)
std_b = np.std(class_b)
print(f"Class B - Mean: {mean_b}, Median: {median_b}, Std: {std_b:.2f}")

mean_c = np.mean(class_c)
median_c = np.median(class_c)
std_c = np.std(class_c)
print(f"Class C - Mean: {mean_c}, Median: {median_c}, Std: {std_c:.2f}")

classes = ['A', 'B', 'C']
means = [mean_a, mean_b, mean_c]
stds = [std_a, std_b, std_c]

best = classes[np.argmax(means)]
print(f"\nHighest average: Class {best}")

consistent = classes[np.argmin(stds)]
print(f"Most consistent: Class {consistent}")


# ============================================================
# TOPIC: Z-Score Normalization (Feature Scaling)
# ============================================================
#
# WHAT: Transforms data so mean = 0 and std = 1
# WHY:  ML models struggle when features have different scales.
#       A neural network seeing age (21-62) vs salary (28k-120k)
#       will think salary matters more just because the numbers are bigger.
#
# FORMULA: z = (x - mean) / std
#
# VISUAL:
#   Before normalization:
#     Ages:     |--21==========62--|           (range ~40)
#     Salaries: |--28000====================120000--|  (range ~92000)
#     Scores:   |--2.9===4.5--|               (range ~1.6)
#
#   After z-score normalization:
#     Ages:     |---(-1.5)====0====(1.5)---|
#     Salaries: |---(-1.5)====0====(1.5)---|   ALL on the same scale!
#     Scores:   |---(-1.5)====0====(1.5)---|
#
# EXAMPLE:
#   data = np.array([10, 20, 30, 40, 50])
#   mean = np.mean(data)          # 30.0
#   std  = np.std(data)           # 14.14
#   z    = (data - mean) / std    # [-1.41, -0.71, 0.0, 0.71, 1.41]
#   print(np.mean(z))             # ≈ 0.0
#   print(np.std(z))              # ≈ 1.0
# ============================================================

# PROBLEM 2: Z-Score Normalizer
# Normalize each feature using z-score: z = (x - mean) / std
# Then verify each normalized feature has mean ≈ 0 and std ≈ 1
# Print the original and normalized values for each feature.

ages = np.array([25, 32, 47, 51, 62, 21, 36, 28, 43, 55])          # range: 21-62
salaries = np.array([30000, 45000, 72000, 85000, 120000, 28000, 52000, 35000, 68000, 95000])  # range: 28k-120k
scores = np.array([3.1, 4.2, 3.8, 2.9, 4.5, 3.5, 4.0, 3.3, 3.7, 4.1])   # range: 2.9-4.5

# Your code below:
features = {'Ages': ages, 'Salaries': salaries, 'Scores': scores}

for name, data in features.items():
    mean = np.mean(data)
    std = np.std(data)
    z = (data - mean) / std

    print(f"\n{name}:")
    print(f"  Original:   {data}")
    print(f"  Normalized: {np.round(z, 2)}")
    print(f"  Verify -> Mean: {np.mean(z):.4f}, Std: {np.std(z):.4f}")


# ============================================================
# TOPIC: Outlier Detection with IQR
# ============================================================
#
# WHAT: Finds data points that are "abnormally" far from the rest
# WHY:  Outliers can wreck ML models — one salary of $10M in training
#       data can shift a model's predictions for everyone else.
#       Detecting and handling outliers is a standard preprocessing step.
#
# METHOD: Interquartile Range (IQR)
#   Q1 = 25th percentile (25% of data falls below)
#   Q3 = 75th percentile (75% of data falls below)
#   IQR = Q3 - Q1 (the "middle 50%" spread)
#   Lower bound = Q1 - 1.5 * IQR
#   Upper bound = Q3 + 1.5 * IQR
#   Anything outside these bounds → outlier
#
# VISUAL:
#   Outliers      Normal Range         Outliers
#   ◄──────|─────[====|════|════]──────|──────►
#        lower   Q1  median  Q3      upper
#        bound                        bound
#          │                            │
#       Q1 - 1.5*IQR            Q3 + 1.5*IQR
#
# EXAMPLE:
#   data = np.array([2, 4, 5, 6, 7, 8, 9, 100])
#   q1 = np.percentile(data, 25)     # 4.75
#   q3 = np.percentile(data, 75)     # 8.25
#   iqr = q3 - q1                    # 3.5
#   lower = q1 - 1.5 * iqr          # -0.5
#   upper = q3 + 1.5 * iqr          # 13.5
#   outliers = data[(data < lower) | (data > upper)]  # [100]
# ============================================================

# PROBLEM 3: Outlier Detector
# A sensor recorded temperature readings, but some are faulty.
# Use the IQR method to:
# - Calculate Q1, Q3, and IQR
# - Find lower and upper bounds
# - Identify and print the outliers
# - Print the "clean" dataset (without outliers)
# - Print mean BEFORE and AFTER removing outliers to see the impact

temps = np.array([22.1, 23.4, 21.8, 22.5, 23.1, 22.9, 85.0, 23.0,
                  22.7, 21.5, -15.3, 22.8, 23.2, 22.4, 22.6, 23.3,
                  22.0, 22.3, 99.9, 21.9, 22.1, 23.0, 22.5, 22.8])

# Your code below:
q1 = np.percentile(temps, 25)
q3 = np.percentile(temps, 75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

print("\n--- Outlier Detection ---")
print(f"Q1: {q1}, Q3: {q3}, IQR: {iqr:.2f}")
print(f"Bounds: [{lower:.2f}, {upper:.2f}]")

outliers = temps[(temps < lower) | (temps > upper)]
clean = temps[(temps >= lower) & (temps <= upper)]

print(f"\nOutliers found: {outliers}")
print(f"Clean data: {clean}")
print(f"\nMean WITH outliers:    {np.mean(temps):.2f}")
print(f"Mean WITHOUT outliers: {np.mean(clean):.2f}")
print(f"Difference:            {abs(np.mean(temps) - np.mean(clean)):.2f}")
