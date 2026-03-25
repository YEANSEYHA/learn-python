import numpy as np
import matplotlib.pyplot as plt

# PROBLEM 1: Create and Display a Checkerboard Pattern
# Neural networks often use test patterns — let's make one.
#
# Steps:
#   1. Create an 8x8 array of zeros: board = np.zeros((8, 8))
#   2. Set alternating squares to 1 using slicing:
#      board[0::2, 1::2] = 1    (even rows, odd columns)
#      board[1::2, 0::2] = 1    (odd rows, even columns)
#   3. Display with plt.imshow(board, cmap='gray')
#   4. Add title("Checkerboard") and plt.show()
#
# Hint: 0::2 means "start at 0, step by 2" → indices 0, 2, 4, 6
#        1::2 means "start at 1, step by 2" → indices 1, 3, 5, 7

# Your code below:

# board = np.zeros((8,8))
# board[0::2 , 1::2] = 1
# board[1::2, 0::2] = 1

# plt.imshow(board ,cmap='gray')
# plt.title("Checkerboard")
# plt.show()


# PROBLEM 2: Image Transformations with Subplots
# Show an original image and 3 transformations side by side.
#
# Steps:
#   1. Create a random 8x8 image: image = np.random.rand(8, 8)
#   2. fig, axes = plt.subplots(1, 4, figsize=(12, 3))
#   3. axes[0] — Original:
#      axes[0].imshow(image, cmap='gray')
#      axes[0].set_title("Original")
#   4. axes[1] — Flipped vertically (reverse rows):
#      axes[1].imshow(image[::-1], cmap='gray')
#      axes[1].set_title("Flip Vertical")
#   5. axes[2] — Flipped horizontally (reverse columns):
#      axes[2].imshow(image[:, ::-1], cmap='gray')
#      axes[2].set_title("Flip Horizontal")
#   6. axes[3] — Rotated 90 degrees:
#      axes[3].imshow(np.rot90(image), cmap='gray')
#      axes[3].set_title("Rotated 90°")
#   7. plt.tight_layout() then plt.show()

# Your code below:
# image = np.array([
#     [1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 0, 0, 0, 0, 0, 1, 1],
#     [0, 0, 0, 0, 0, 1, 1, 0],
#     [0, 0, 0, 0, 1, 1, 0, 0],
#     [0, 0, 0, 1, 1, 0, 0, 0],
#     [0, 0, 1, 1, 0, 0, 0, 0],
#     [0, 0, 1, 1, 0, 0, 0, 0],
# ])
# fig, axes = plt.subplots(1,4, figsize=(12,3))

# # Original images
# axes[0].imshow(image, cmap='gray')
# axes[0].set_title("Original Image")

# # Flipped vertically
# flipped = image[::-1]
# axes[1].imshow(flipped,cmap='gray')
# axes[1].set_title("Flipped Vertically")


# # Flipped horizontal
# flipped_cols = image[:, ::-1]
# axes[2].imshow(flipped_cols,cmap='gray')
# axes[2].set_title("Flipped Horizontal")

# # Rotate 90 Degress

# rotate_img = np.rot90(image)

# axes[3].imshow(rotate_img,cmap='gray')
# axes[3].set_title("Rotate 90 Degree")


# plt.show()


# PROBLEM 3: Brighten an Image
# Images are arrays, so adding a number = brighter pixels!
# np.clip(array, 0, 1) keeps values from going above 1.
#
# Steps:
#   1. image2 = np.random.rand(8, 8)
#   2. bright = np.clip(image2 + 0.3, 0, 1)
#   3. Display bright with plt.imshow(bright, cmap='gray')
#   4. Add title("Brightened Image") and plt.show()

# Your code below:

image2 = np.random.rand(8,8)
bright = np.clip(image2+0.3, 0,1)

plt.imshow(bright,cmap='gray')
plt.title("Brightened Image")

plt.show()


