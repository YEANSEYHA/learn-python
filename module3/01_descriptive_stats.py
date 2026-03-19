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


# PROBLEM 2: Z-Score Normalizer
# You have features with very different scales.
# Normalize each feature using z-score: z = (x - mean) / std
# Then verify each normalized feature has mean ≈ 0 and std ≈ 1
#
# Print the original and normalized values for each feature.

ages = np.array([25, 32, 47, 51, 62, 21, 36, 28, 43, 55])          # range: 21-62
salaries = np.array([30000, 45000, 72000, 85000, 120000, 28000, 52000, 35000, 68000, 95000])  # range: 28k-120k
scores = np.array([3.1, 4.2, 3.8, 2.9, 4.5, 3.5, 4.0, 3.3, 3.7, 4.1])   # range: 2.9-4.5

# Your code below:
