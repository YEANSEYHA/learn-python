import numpy as np
import matplotlib.pyplot as plt

# PROBLEM 1: Side-by-Side Loss and Accuracy
# Create a figure with 1 row, 2 columns.
# Left plot: training loss curve. Right plot: accuracy curve.
#
# Steps:
#   1. fig, axes = plt.subplots(1, 2, figsize=(10, 4))
#   2. epochs = np.arange(1, 21)
#   3. Left plot (axes[0]):
#      - Plot loss = 2.5 * np.exp(-0.2 * epochs)
#      - axes[0].plot(epochs, loss)
#      - axes[0].set_title("Loss"), axes[0].set_xlabel("Epoch")
#   4. Right plot (axes[1]):
#      - Plot accuracy = 1 - np.exp(-0.3 * epochs)
#      - axes[1].plot(epochs, accuracy)
#      - axes[1].set_title("Accuracy"), axes[1].set_xlabel("Epoch")
#   5. plt.tight_layout() then plt.show()

# Your code below:

# fig, axes = plt.subplots(1, 2, figsize=(10,4))
# epochs = np.arange(1,21)

# loss = 2.5 * np.exp(-0.2 * epochs)
# axes[0].plot(epochs,loss)
# axes[0].set_title("loss"), axes[0].set_xlabel("Epoch")

# accuracy = 1 - np.exp(-0.3 * epochs)

# axes[1].plot(epochs, accuracy)
# axes[1].set_title("Accuracy"), axes[1].set_xlabel("Epoch")



# plt.show()


# PROBLEM 2: 2x2 Grid — Four Different Plot Types
# Create a 2x2 grid showing four different visualizations.
#
# Steps:
#   1. fig, axes = plt.subplots(2, 2, figsize=(10, 8))
#   2. Top-left axes[0, 0] — Line plot:
#      x = np.arange(1, 21)
#      axes[0, 0].plot(x, np.log(x))
#      axes[0, 0].set_title("Log Growth")
#   3. Top-right axes[0, 1] — Scatter plot:
#      data = np.random.randn(50, 2)
#      axes[0, 1].scatter(data[:, 0], data[:, 1])
#      axes[0, 1].set_title("Random Points")
#   4. Bottom-left axes[1, 0] — Histogram:
#      axes[1, 0].hist(np.random.randn(300), bins=20, edgecolor='black')
#      axes[1, 0].set_title("Weight Distribution")
#   5. Bottom-right axes[1, 1] — Heatmap:
#      grid = np.random.rand(5, 5)
#      axes[1, 1].imshow(grid, cmap='Blues')
#      axes[1, 1].set_title("Heatmap")
#   6. plt.tight_layout() then plt.show()

# Your code below:
# fig , axes = plt.subplots(2,2, figsize=(10,8))

# x = np.arange(1,21)
# axes[0, 0].plot(x, np.log(x))
# axes[0, 0].set_title("Log Growth")

# data= np.random.randn(50,2)
# axes[0,1].scatter(data[:, 0], data[:,1])
# axes[0,1].set_title("Random Points")


# axes[1,0].hist(np.random.randn(300),bins=20,edgecolor="black")
# axes[1,0].set_title("Weight Distribution")

# grid = np.random.rand(5,5)
# axes[1,1].imshow(grid, cmap='Blues')
# axes[1,1].set_title("Heatmap")
# plt.tight_layout()



# plt.show()


# PROBLEM 3: Three Activation Functions Side by Side
# Plot sigmoid, tanh, and ReLU — the 3 most common activation functions.
# You'll use these in Module 6 when building neural networks!
#
# Steps:
#   1. fig, axes = plt.subplots(1, 3, figsize=(12, 4))
#   2. x = np.linspace(-5, 5, 100)   (100 points from -5 to 5)
#   3. Left axes[0] — Sigmoid:
#      sigmoid = 1 / (1 + np.exp(-x))
#      axes[0].plot(x, sigmoid)
#      axes[0].set_title("Sigmoid")
#   4. Middle axes[1] — Tanh:
#      axes[1].plot(x, np.tanh(x))
#      axes[1].set_title("Tanh")
#   5. Right axes[2] — ReLU:
#      relu = np.maximum(0, x)
#      axes[2].plot(x, relu)
#      axes[2].set_title("ReLU")
#   6. plt.tight_layout() then plt.show()


# Your code below:

fig, axes = plt.subplots(1,3,figsize=(12,4))
x=np.linspace(-5,5,100)

sigmoid = 1/(1+np.exp(-x))

axes[0].plot(x, sigmoid)
axes[0].set_title("Sigmoid")

axes[1].plot(x, np.tanh(x))
axes[1].set_title("Tanh")

relu = np.maximum(0,x)
axes[2].plot(x, relu)
axes[2].set_title("Relu")

plt.tight_layout()
plt.show()
