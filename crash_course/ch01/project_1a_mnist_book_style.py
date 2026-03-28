"""
PROJECT 1A: Classifying Handwritten Digits (MNIST)
Book: Deep Learning Crash Course, Chapter 1 (p.64-78)
Style: Book's original code using Deeplay library

SETUP REQUIRED:
===============
pip install deeplay torchvision seaborn
"""

# ============================================================
# STEP 1: Load the MNIST Data (Book Listing 1-36, 1-37)
# ============================================================
import os
import matplotlib.pyplot as plt
import numpy as np

# Download MNIST dataset
from torchvision import datasets
datasets.MNIST(root=".", download=True)

# Load training images
train_path = os.path.join("MNIST", "raw", "train-images-idx3-ubyte")

# Alternative: use the book's file-based approach
# The book loads from image files in a folder structure
# We'll use torchvision's built-in loader for convenience
from torchvision import transforms

train_dataset = datasets.MNIST(root=".", train=True, download=True)
train_images = [np.array(img) for img, _ in train_dataset]
train_digits = [label for _, label in train_dataset]

print(f"Number of training images: {len(train_images)}")   # 60,000
print(f"Image shape: {train_images[0].shape}")              # (28, 28)


# ============================================================
# STEP 2: Visualize the Data (Book Listing 1-38)
# ============================================================
fig, axs = plt.subplots(nrows=3, ncols=10, figsize=(20, 6))
for ax in axs.ravel():
    idx_image = np.random.choice(60000)
    ax.imshow(train_images[idx_image], cmap="Greys")
    ax.set_title(f"Label: {train_digits[idx_image]}", fontsize=20)
    ax.axis("off")
plt.show()


# ============================================================
# STEP 3: Define the Neural Network (Book Listing 1-39)
# ============================================================
import deeplay as dl
from torch.nn import Sigmoid

mlp_template = dl.MultiLayerPerceptron(
    in_features=28 * 28, hidden_features=[32, 32], out_features=10,
)
mlp_template[..., "activation"].configure(Sigmoid)
mlp_model = mlp_template.create()

print(mlp_model)
print(f"{sum(p.numel() for p in mlp_model.parameters())} trainable parameters")


# ============================================================
# STEP 4: Compile the Neural Network (Book Listing 1-40)
# ============================================================
from torch.nn import MSELoss

classifier_template = dl.Classifier(
    model=mlp_template, num_classes=10, make_targets_one_hot=True,
    loss=MSELoss(), optimizer=dl.SGD(lr=.1),
)
classifier = classifier_template.create()
print(classifier)


# ============================================================
# STEP 5: Create Data Loader & Train (Book Listing 1-41)
# ============================================================
train_images_digits = list(zip(train_images, train_digits))
train_dataloader = dl.DataLoader(train_images_digits, shuffle=True)

trainer = dl.Trainer(max_epochs=1, accelerator="auto")
trainer.fit(classifier, train_dataloader)


# ============================================================
# STEP 6: Test Accuracy (Book Listing 1-42)
# ============================================================
test_dataset = datasets.MNIST(root=".", train=False, download=True)
test_images = [np.array(img) for img, _ in test_dataset]
test_digits = [label for _, label in test_dataset]

test_images_digits = list(zip(test_images, test_digits))
test_dataloader = dl.DataLoader(test_images_digits, shuffle=False)

trainer.test(classifier, test_dataloader)
# Expected accuracy: ~0.67


# ============================================================
# STEP 7: Confusion Matrix (Book Listing 1-43)
# ============================================================
from seaborn import cubehelix_palette, heatmap

def plot_confusion_matrix(classifier, dataloader):
    """Plot confusion matrix."""
    confusion_matrix = np.zeros((10, 10), dtype=int)
    for image, gt_digit in dataloader:
        predictions = classifier(image)
        max_prediction, pred_digit = predictions.max(dim=1)
        np.add.at(confusion_matrix, (gt_digit, pred_digit), 1)
    plt.figure(figsize=(10, 8))
    heatmap(confusion_matrix, annot=True, fmt=".0f", square=True,
            cmap=cubehelix_palette(light=0.95, as_cmap=True), vmax=150)
    plt.xlabel("Predicted digit", fontsize=15)
    plt.ylabel("Ground truth digit", fontsize=15)
    plt.show()

plot_confusion_matrix(classifier, test_dataloader)


# ============================================================
# STEP 8: Improvement — Softmax Output (Book Listing 1-44)
# ============================================================
from torch.nn import Softmax

classifier_template[..., "activation#-1"].configure(Softmax, dim=-1)
classifier_softmax = classifier_template.create()
trainer_softmax = dl.Trainer(max_epochs=1, accelerator="auto")
trainer_softmax.fit(classifier_softmax, train_dataloader)
trainer_softmax.test(classifier_softmax, test_dataloader)
# Expected accuracy: ~0.77

plot_confusion_matrix(classifier_softmax, test_dataloader)


# ============================================================
# STEP 9: Improvement — ReLU Activation (Book Listing 1-45)
# ============================================================
from torch.nn import ReLU

classifier_template[..., "activation#:-1"].configure(ReLU)
classifier_relu = classifier_template.create()
trainer_relu = dl.Trainer(max_epochs=1, accelerator="auto")
trainer_relu.fit(classifier_relu, train_dataloader)
trainer_relu.test(classifier_relu, test_dataloader)
# Expected accuracy: ~0.94

plot_confusion_matrix(classifier_relu, test_dataloader)


# ============================================================
# STEP 10: Improvement — Mini-batches + RMSprop (Book Listing 1-46)
# ============================================================
train_dataloader_batch = dl.DataLoader(train_images_digits, shuffle=True,
                                        batch_size=32)

classifier_template.configure(optimizer=dl.RMSprop(lr=0.001))
classifier_rmsprop = classifier_template.create()
trainer_rmsprop = dl.Trainer(max_epochs=10, accelerator="auto")
trainer_rmsprop.fit(classifier_rmsprop, train_dataloader_batch)
trainer_rmsprop.test(classifier_rmsprop, test_dataloader)
# Expected accuracy: ~0.97

plot_confusion_matrix(classifier_rmsprop, test_dataloader)


# ============================================================
# STEP 11: Failure Analysis (Book Listing 1-47)
# ============================================================
num_images_x_digit = 3
plt.figure(figsize=(10, num_images_x_digit))
num_fails_x_digit = np.zeros(10, int)
for image, gt_digit in test_dataloader:
    gt_digit = int(gt_digit)
    if num_fails_x_digit[gt_digit] < num_images_x_digit:
        predictions = classifier_rmsprop(image)
        max_prediction, pred_digit = predictions.max(dim=1)
        if pred_digit != gt_digit:
            num_fails_x_digit[gt_digit] += 1
            plt.subplot(num_images_x_digit, 10,
                        (num_fails_x_digit[gt_digit] - 1) * 10 + gt_digit + 1)
            plt.imshow(image.squeeze(), cmap="Greys")
            plt.annotate(str(int(pred_digit)), (.8, 1), (1, 1),
                         xycoords="axes fraction", textcoords="offset points",
                         va="top", ha="left", fontsize=20, color="red")
            plt.axis("off")
    if (num_fails_x_digit >= num_images_x_digit).all():
        break
plt.tight_layout()
plt.show()


# ============================================================
# EXERCISES (from the book, p.79)
# ============================================================

# EXERCISE 1-1: Implement the dense neural network FROM SCRATCH
# (no Deeplay) and train it to classify MNIST digits.

# --- your code here ---


# EXERCISE 1-2: Use Fashion-MNIST instead of digit MNIST.
# Fashion-MNIST has: T-shirt, trouser, pullover, dress, coat,
# sandal, shirt, sneaker, bag, ankle boot.
# How does accuracy compare?

# --- your code here ---
