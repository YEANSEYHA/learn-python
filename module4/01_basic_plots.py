import numpy as np
import matplotlib.pyplot as plt

# PROBLEM 1: Plot a Training Loss Curve
# Simulate a model training for 20 epochs.
# The loss starts at 2.5 and decreases each epoch.
#
# Steps:
#   1. Create an array of epochs from 1 to 20
#   2. Calculate loss: loss = 2.5 * exp(-0.2 * epochs)  (use np.exp)
#   3. Plot it as a line plot with plt.plot()
#   4. Add xlabel("Epoch"), ylabel("Loss"), title("Training Loss")
#   5. Call plt.show()

# Your code below:

epochs = np.arange(1,21,1)
print(epochs)

loss = 2.5 * np.exp(-0.2 * epochs)

# print(loss)

# plt.plot(epochs ,loss)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss")

# plt.show()


# PROBLEM 2: Scatter Plot — Two Classes
# Generate random data for two classes and plot them.
#
# Steps:
#   1. Generate 30 random points for class_a centered at (2, 2)
#      class_a = np.random.randn(30, 2) + [2, 2]
#   2. Generate 30 random points for class_b centered at (6, 6)
#   3. Plot class_a with plt.scatter() using label="Class A"
#   4. Plot class_b with plt.scatter() using label="Class B"
#   5. Add xlabel, ylabel, title, plt.legend(), and plt.show()
#
# Hint for scatter: plt.scatter(data[:, 0], data[:, 1])
#   data[:, 0] = all rows, column 0 (x values)
#   data[:, 1] = all rows, column 1 (y values)

# Your code below:
# class_a = np.random.randn(30, 2) + [2, 2]
# class_b = np.random.randn(30, 2) + [6, 6]


# plt.scatter(class_a[:, 0], class_a[:, 1], label="Class A")
# plt.scatter(class_b[:, 0], class_b[:, 1], label="Class B")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.title("Two Classes")
# plt.legend()
# plt.show()


# PROBLEM 3: Training vs Validation Loss
# Plot TWO lines on the same chart.
#   - Training loss decreases smoothly
#   - Validation loss decreases then goes UP (overfitting!)
#
# Steps:
#   1. epochs = np.arange(1, 21)
#   2. train_loss = 2.5 * np.exp(-0.25 * epochs)
#   3. val_loss = 2.5 * np.exp(-0.15 * epochs) + 0.02 * epochs  (goes back up!)
#   4. Plot both lines: plt.plot(epochs, train_loss, label="Train")
#      and plt.plot(epochs, val_loss, label="Validation")
#   5. Add xlabel, ylabel, title("Training vs Validation Loss"), legend, show

# Your code below:

epochs = np.arange(1,21)
train_loss = 2.5 * np.exp(-0.25 * epochs)
val_loss = 2.5 * np.exp(-0.15 * epochs) + 0.02 * epochs

plt.plot(epochs , train_loss, label="Train")
plt.plot(epochs, val_loss, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

