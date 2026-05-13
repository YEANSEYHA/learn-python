import torch
import torch.nn.functional as F

# ── EXAMPLE: What does a convolution do? ──────────────────────────────────────

# A 1x1 grayscale image of size 6x6 (batch=1, channels=1, H=6, W=6)
image = torch.tensor([[
    [1., 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1],
]]).unsqueeze(0)  # shape: (1, 1, 6, 6)

# A 3x3 filter (detects a checkerboard-like pattern)
filter_ = torch.tensor([[[[1., 0, 1],
                           [0, 1, 0],
                           [1, 0, 1]]]])  # shape: (1, 1, 3, 3)

# Apply convolution (no padding, stride=1)
output1 = F.conv2d(image, filter_, stride=1, padding=0)

# print("Input shape: ", image.shape)    # (1, 1, 6, 6)
# print("Filter shape:", filter_.shape)  # (1, 1, 3, 3)
# print("Output shape:", output.shape)   # (1, 1, 4, 4)
# print("\nFeature map:")
# print(output.squeeze())                # 4x4 result


# ── EXERCISE 1 ────────────────────────────────────────────────────────────────

# A 3x3 filter (detects a checkerboard-like pattern)
filter_a = torch.tensor([[[[-1., -1., -1.],
                           [0., 0., 0.],
                           [1., 1., 1.]]]])  # shape: (1, 1, 3, 3)

output = F.conv2d(image, filter_a,stride=1, padding=0)

print("Input shape: ", image.shape)    # (1, 1, 6, 6)
print("Filter shape:", filter_.shape)  # (1, 1, 3, 3)
print("Output shape:", output.shape)   # (1, 1, 4, 4)
print("\nFeature map:")
print(output.squeeze())                # 4x4 result




# Exercise 2
output2 = F.max_pool2d(output1, kernel_size=2)

print("Output 2: ",output2)

# Exercise 3 

output3 = torch.flatten(output2, start_dim=1)

print("Flattern Value :",output3)