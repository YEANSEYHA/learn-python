# ============================================
# PyTorch Tensors — The Foundation
# ============================================

import torch
import numpy as np

# ── Problem 1: Create and manipulate tensors ──
#
# 1. Create a tensor called `data` with values: [[1, 2, 3], [4, 5, 6]]  (use float)
# 2. Print its shape, dtype, and the element at row 1, col 2
# 3. Multiply every element by 10 and print the result
#
# Expected output:
#   Shape: torch.Size([2, 3])
#   Dtype: torch.float32
#   Element [1,2]: 6.0
#   Multiplied: tensor([[10., 20., 30.], [40., 50., 60.]])

# Your code below:
data = torch.tensor([[1, 2, 3], [4, 5, 6]],dtype=torch.float32)

print(data.shape)
print(data.dtype)

print(data[1,2])

print(data*10)


# ── Problem 2: NumPy ↔ PyTorch ──
#
# 1. Create a NumPy array: arr = np.array([10.0, 20.0, 30.0])
# 2. Convert it to a PyTorch tensor called `t` using torch.from_numpy()
# 3. Add 5 to `t` and store in `t2`
# 4. Convert `t2` back to NumPy using .numpy() and print it
#
# Expected output:
#   Tensor: tensor([10., 20., 30.])
#   Result as NumPy: [15. 25. 35.]

# Your code below:

arr = np.array([10.0, 20.0, 30.0])

t = torch.tensor(arr)

t2 = torch.add(t, 5)

print("T2 value",t2)


print(t2.numpy())


# ── Problem 3: Autograd (automatic gradients) ──
#
# 1. Create a tensor: w = torch.tensor([3.0], requires_grad=True)
# 2. Compute: y = w ** 2 + 2 * w    (so y = w² + 2w)
# 3. Call y.backward()               (PyTorch computes dy/dw automatically)
# 4. Print w.grad                    (the gradient)
#
# Math check: dy/dw = 2w + 2 = 2(3) + 2 = 8.0
#
# Expected output:
#   w.grad: tensor([8.])

# Your code below:


### requires_grad=True  →  "Track this variable"
w = torch.tensor([3.0], requires_grad=True)
y = w ** 2 + 2 * w
y.backward()
print(w.grad)