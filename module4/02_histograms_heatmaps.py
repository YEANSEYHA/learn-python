import numpy as np
import matplotlib.pyplot as plt

# PROBLEM 1: Histogram of Model Predictions
# A model outputs probabilities between 0 and 1.
# Plot a histogram to see how the predictions are distributed.
#
# Steps:
#   1. Generate 500 random predictions: predictions = np.random.rand(500)
#   2. Plot a histogram with plt.hist(predictions, bins=20, edgecolor='black')
#   3. Add xlabel("Predicted Probability"), ylabel("Frequency")
#   4. Add title("Model Prediction Distribution")
#   5. plt.show()

# Your code below:
# predictions = np.random.rand(500)
# plt.hist(predictions, bins=20, edgecolor='black')
# plt.xlabel("Predicted Probability")
# plt.ylabel("Frequency")
# plt.title("Model Prediction Distribution")

# plt.show()


# PROBLEM 2: Heatmap of a Confusion Matrix
# You trained a model to classify digits: 0, 1, 2
# Here's how it performed — display it as a heatmap.
#
# Steps:
#   1. Create this confusion matrix as a NumPy array:
#      [[45,  3,  2],
#       [ 1, 48,  1],
#       [ 4,  2, 44]]
#   2. Display it with plt.imshow(cm, cmap='Blues')
#   3. Add plt.colorbar(label="Count")
#   4. Add xticks and yticks: plt.xticks([0, 1, 2], ["0", "1", "2"])
#      (same for yticks)
#   5. Add xlabel("Predicted"), ylabel("Actual"), title("Confusion Matrix")
#   6. plt.show()

# Your code below:

# data = np.array([[45,  3,  2],
#       [ 1, 48,  1],
#       [ 4,  2, 44]])

# plt.imshow(data, cmap="Blues")
# plt.colorbar(label="Count")
# plt.xticks([0, 1, 2], ["0", "1", "2"])
# plt.yticks([0, 1, 2], ["0", "1", "2"])


# plt.show()


# PROBLEM 3: Histogram Comparing Two Distributions
# Compare weights from two neural network layers.
#
# Steps:
#   1. layer1 = np.random.randn(500) * 0.5    (narrow spread)
#   2. layer2 = np.random.randn(500) * 2.0    (wide spread)
#   3. Plot BOTH on the same chart using:
#      plt.hist(layer1, bins=30, alpha=0.5, label="Layer 1", edgecolor='black')
#      plt.hist(layer2, bins=30, alpha=0.5, label="Layer 2", edgecolor='black')
#      (alpha=0.5 makes them semi-transparent so you can see overlap)
#   4. Add xlabel("Weight Value"), ylabel("Frequency")
#   5. Add title("Weight Distributions"), plt.legend(), plt.show()

# Your code below:
layer1 = np.random.randn(500) * 0.5
layer2 = np.random.randn(500) * 2.0

plt.hist(layer1, bins=30, alpha=0.5, label="Layer 1", edgecolor='black')
plt.hist(layer2, bins=30, alpha=0.5, label="Layer 2", edgecolor='black')

plt.xlabel("Weight Value")
plt.ylabel("Frequency")
plt.title("Weight Distributions")
plt.legend()

plt.show()