"""
PROJECT 1A: Classifying Handwritten Digits (MNIST)
Book: Deep Learning Crash Course, Chapter 1 (p.64-78)
Time: ~2 hours

YOUR FIRST REAL AI PROJECT!
============================
You will build a neural network that looks at handwritten digit images
and predicts which digit (0-9) it sees.

Dataset: MNIST — 60,000 training images + 10,000 test images
Each image: 28×28 pixels, grayscale


VISUAL — What the data looks like:
====================================

    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ ██████   │    │    ██    │    │  █████   │
    │      ██  │    │   ███    │    │ ██   ██  │
    │    ███   │    │  ████    │    │      ██  │
    │  ███     │    │    ██    │    │    ███   │
    │ ████████ │    │ ████████ │    │      ██  │
    └──────────┘    └──────────┘    └──────────┘
      Label: 2        Label: 1        Label: 3


VISUAL — How the neural network works:
========================================

    Input (784 pixels)      Hidden layers         Output (10 classes)
    ┌───┐
    │ . │──┐               ┌──┐
    │ . │──┼──────────────▶│32│──┐                ┌──┐
    │ . │──┤   flatten     │  │  │    ┌──┐        │ 0│ → 0.01
    │   │  │   28×28       └──┘  ├───▶│32│──┐     │ 1│ → 0.02
    │784│  │   = 784              │  │  │     │ 2│ → 0.91  ← winner!
    │   │  │   numbers     ┌──┐  │    └──┘  ├───▶│ 3│ → 0.01
    │ . │──┤               │32│──┘          │     │...│
    │ . │──┼──────────────▶│  │─────────────┘     │ 9│ → 0.00
    │ . │──┘               └──┘                   └──┘
    └───┘
    (image)            (learn patterns)        (probability per digit)


THE BUILD — Step by step:
==========================
We'll build this project in 6 steps, improving accuracy each time:

    Step 1: Load & visualize MNIST data
    Step 2: Build a basic neural network          → ~67% accuracy
    Step 3: Add softmax output                    → ~77% accuracy
    Step 4: Switch to ReLU activation             → ~94% accuracy
    Step 5: Use mini-batches + better optimizer   → ~97% accuracy
    Step 6: Analyze failures (which digits are hardest?)

Each step, you write the code. I show you the concept first.
"""

# ============================================================
# STEP 1: Load and Visualize the MNIST Data
# ============================================================
#
# The MNIST dataset contains images of handwritten digits 0-9.
# Each image is 28×28 pixels (784 total values).
#
# We'll use torchvision to download it automatically.
#
# EXAMPLE:
# --------
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Download MNIST (auto-downloads to ./data folder)
train_dataset = datasets.MNIST(root='./data', train=True, download=True,
                                transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, download=True,
                               transform=transforms.ToTensor())

print(f"Training images: {len(train_dataset)}")   # 60,000
print(f"Test images:     {len(test_dataset)}")     # 10,000
print(f"Image shape:     {train_dataset[0][0].shape}")  # [1, 28, 28]

# Visualize 10 random digits
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for ax in axes:
    idx = np.random.randint(len(train_dataset))
    image, label = train_dataset[idx]
    ax.imshow(image.squeeze(), cmap='Greys')
    ax.set_title(f"{label}", fontsize=14)
    ax.axis('off')
plt.suptitle("Sample MNIST Digits", fontsize=16)
plt.tight_layout()
plt.savefig("crash_course/ch01/step1_mnist_samples.png", dpi=100)
plt.show()


# ============================================================
# EXERCISE 1: Explore the data
# ============================================================
# (a) Pick one image from the dataset and print its pixel values
#     using: image, label = train_dataset[0]
#     Then print image.shape, image.min(), image.max()
#
# (b) How many pixels does each image have? (hint: 28 * 28 = ?)
#     This number becomes the INPUT SIZE of our neural network.
#
# (c) Plot a 3×10 grid (like the book's Figure 1-19) showing
#     30 random digits with their labels as titles.

# --- your code here ---


# ============================================================
# STEP 2: Build a Basic Neural Network (~67% accuracy)
# ============================================================
#
# Architecture:
#   Input:  784 neurons (one per pixel)
#   Hidden: 32 neurons → 32 neurons (2 hidden layers)
#   Output: 10 neurons (one per digit class)
#   Activation: Sigmoid
#   Loss: MSE (mean squared error)
#   Optimizer: SGD (learning rate = 0.1)
#
# KEY CONCEPTS:
# - Flatten: reshape 28×28 image into 784-length vector
# - One-hot encoding: digit "3" becomes [0,0,0,1,0,0,0,0,0,0]
# - Forward pass: input → hidden layers → output prediction
# - Loss: how wrong the prediction is
# - Backpropagation: adjust weights to reduce loss
#
# EXAMPLE:
# --------
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Define the network
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(784, 32)
        self.layer2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)          # [batch, 1, 28, 28] → [batch, 784]
        x = self.sigmoid(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

model_v1 = DigitClassifier()
print(f"Parameters: {sum(p.numel() for p in model_v1.parameters()):,}")

# Training setup
criterion = nn.MSELoss()
optimizer = optim.SGD(model_v1.parameters(), lr=0.1)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Train for 1 epoch
model_v1.train()
for i, (image, label) in enumerate(train_loader):
    # One-hot encode the label
    target = torch.zeros(1, 10)
    target[0, label] = 1.0

    output = model_v1(image)            # forward pass
    loss = criterion(output, target)    # calculate loss
    optimizer.zero_grad()               # clear old gradients
    loss.backward()                     # backpropagation
    optimizer.step()                    # update weights

    if (i + 1) % 10000 == 0:
        print(f"  Processed {i+1}/60000 images, loss: {loss.item():.4f}")

# Test accuracy
model_v1.eval()
correct = 0
with torch.no_grad():
    for image, label in DataLoader(test_dataset, batch_size=1):
        output = model_v1(image)
        pred = output.argmax(dim=1)
        correct += (pred == label).sum().item()

accuracy_v1 = correct / len(test_dataset)
print(f"\nModel v1 accuracy: {accuracy_v1:.2%}")  # ~67%


# ============================================================
# EXERCISE 2: Understand the baseline
# ============================================================
# (a) Print the model architecture using: print(model_v1)
#     Count: how many layers? What are the sizes?
#
# (b) Pick a test image, run it through the model, and print
#     the 10 output values. Which digit got the highest score?
#     hint: output = model_v1(image)
#           print(output)
#           print(f"Predicted: {output.argmax(dim=1).item()}")

# --- your code here ---


# ============================================================
# STEP 3: Add Softmax Output (~77% accuracy)
# ============================================================
#
# PROBLEM: Sigmoid outputs don't sum to 1.
#          The model might say "70% it's a 3" AND "60% it's a 5"
#          That doesn't make sense — probabilities should sum to 1.
#
# SOLUTION: Softmax activation on the output layer.
#
#           softmax(y_i) = e^(y_i) / sum(e^(y_j) for all j)
#
#           This forces all 10 outputs to sum to exactly 1.0
#
# VISUAL:
#   Before (sigmoid):  [0.7, 0.6, 0.3, ...]  sum = ???
#   After (softmax):   [0.4, 0.3, 0.1, ...]  sum = 1.0  ✓


# ============================================================
# EXERCISE 3: Build model v2 with softmax
# ============================================================
# Copy the DigitClassifier class above, but change the output:
#   - Hidden layers: keep sigmoid
#   - Output layer: use nn.Softmax(dim=1) instead of sigmoid
#
# Then train it for 1 epoch (same as Step 2) and test accuracy.
# You should get ~77%.

# --- your code here ---


# ============================================================
# STEP 4: Switch to ReLU Activation (~94% accuracy)
# ============================================================
#
# PROBLEM: Sigmoid has "vanishing gradients" — for very large
#          or very small inputs, the gradient is nearly 0,
#          so the network stops learning.
#
# SOLUTION: ReLU (Rectified Linear Unit)
#
#           ReLU(x) = max(0, x)
#
#           Simple, fast, and gradients don't vanish for positive values.
#
# VISUAL:
#   Sigmoid:  ___/‾‾‾  (S-shaped, gradients vanish at extremes)
#   ReLU:     __/      (zero for negative, linear for positive)


# ============================================================
# EXERCISE 4: Build model v3 with ReLU
# ============================================================
# Copy your model v2, but change:
#   - Hidden layers: nn.ReLU() instead of sigmoid
#   - Output layer: keep nn.Softmax(dim=1)
#
# Train for 1 epoch, test accuracy. You should get ~94%.

# --- your code here ---


# ============================================================
# STEP 5: Mini-batches + Better Optimizer (~97% accuracy)
# ============================================================
#
# Two improvements:
#
# 1) MINI-BATCHES: Instead of training on 1 image at a time,
#    train on 32 images at once. This is faster AND more stable.
#
#    batch_size=1:   slow, noisy updates
#    batch_size=32:  fast, smoother updates  ← sweet spot
#
# 2) BETTER OPTIMIZER: Replace SGD with RMSprop
#    SGD:     same learning rate for all weights
#    RMSprop: adapts learning rate per weight (faster convergence)


# ============================================================
# EXERCISE 5: Build model v4 with batches + RMSprop
# ============================================================
# Changes from model v3:
#   - DataLoader: batch_size=32 (not 1)
#   - Optimizer: optim.RMSprop(model.parameters(), lr=0.001)
#   - Train for 10 epochs (not 1)
#   - Note: with batches, one-hot encoding changes:
#
#     target = torch.zeros(len(label), 10)        # batch of targets
#     for j in range(len(label)):
#         target[j, label[j]] = 1.0
#
# Train and test. You should get ~97%.

# --- your code here ---


# ============================================================
# STEP 6: Confusion Matrix + Failure Analysis
# ============================================================
#
# A confusion matrix shows WHERE the model makes mistakes.
#
# VISUAL:
#                    Predicted
#              0  1  2  3  4  5  6  7  8  9
#         0 [964  0  3  1  0  1  6  1  2  2]
#   True  1 [  0 1113 3  3  0  1  5  2  8  0]
#         2 [  6  2 999 6  3  1  3  3  9  0]  ← row = true digit
#         ...                                    column = what model guessed
#
# Diagonal = correct predictions (high = good)
# Off-diagonal = mistakes (which digits get confused?)


# ============================================================
# EXERCISE 6: Build and plot the confusion matrix
# ============================================================
# Using your best model (v4), create a 10×10 confusion matrix:
#
# (a) Create: confusion = np.zeros((10, 10), dtype=int)
#
# (b) Loop through test data:
#     for image, label in test_loader:
#         output = model(image)
#         pred = output.argmax(dim=1)
#         for t, p in zip(label, pred):
#             confusion[t, p] += 1
#
# (c) Plot as a heatmap:
#     plt.imshow(confusion, cmap='Blues')
#     plt.xlabel("Predicted")
#     plt.ylabel("True")
#     plt.colorbar()
#
# (d) Which two digits does the model confuse most often?

# --- your code here ---


# ============================================================
# EXERCISE 7 (BONUS): Failure analysis
# ============================================================
# Find images that the model got WRONG and display them.
# Can you see why the model was confused?
#
# Plot a grid: 3 wrong predictions per digit (3 rows × 10 cols)
# Title each image with: "True: 5, Pred: 3"
#
# hint: collect wrong predictions while looping through test data

# --- your code here ---


# ============================================================
# CHALLENGE PROJECT (from the book, p.79)
# ============================================================
# Build the same classifier for Fashion-MNIST instead of digits.
# Fashion-MNIST has the same format (28×28, 10 classes) but with
# clothing items: T-shirt, trouser, pullover, dress, coat,
# sandal, shirt, sneaker, bag, ankle boot.
#
# Use: datasets.FashionMNIST(root='./data', download=True, ...)
# How does your accuracy compare to digit MNIST?

# --- your code here ---
