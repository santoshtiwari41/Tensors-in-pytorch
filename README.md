# Understanding Tensors in PyTorch

---

## Topics

1. [Introduction to Tensors](#introduction-to-tensors)
2. [Creating Tensors](#creating-tensors)
3. [Tensor Operations](#tensor-operations)
4. [Manipulating Tensors](#manipulating-tensors)
5. [Broadcasting](#broadcasting)
6. [Matrix Operations](#matrix-operations)
7. [Visualization of Tensors](#visualization-of-tensors)
8. [Tensors on GPU](#tensors-on-gpu)
9. [Neural Network Use Case](#neural-network-use-case)
10. [Requirements](#requirements)
11. [Usage](#usage)

---

## Introduction to Tensors
Tensors are the backbone of PyTorch, representing multidimensional arrays that generalize scalars, vectors, and matrices to higher dimensions. They are used for data representation and manipulation in machine learning, deep learning, and scientific computing.

---

## Creating Tensors

```python
import torch

# Scalar
scalar = torch.tensor(3.14)

# Vector
vector = torch.tensor([1.0, 2.0, 3.0])

# Matrix
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# 3D Tensor
tensor_3d = torch.rand(3, 2, 4)
```

---

## Tensor Operations

```python
# Element-wise addition and multiplication
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
result_add = a + b
result_mul = a * b

# Reduction operations
sum_result = a.sum()
mean_result = a.float().mean()
```

---

## Manipulating Tensors

```python
# Reshaping
reshaped_tensor = torch.arange(12).view(3, 4)

# Slicing
first_row = reshaped_tensor[0]
column_2 = reshaped_tensor[:, 1]

# Transposing
transposed_tensor = reshaped_tensor.T
```

---

## Broadcasting

```python
x = torch.tensor([[1, 2, 3]])
y = torch.tensor([[1], [2], [3]])
result = x + y
```

---

## Matrix Operations

```python
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])

# Matrix multiplication
C = torch.matmul(A, B)

# Element-wise multiplication
D = A * B
```

---

## Visualization of Tensors

```python
import matplotlib.pyplot as plt

# Visualize a 2D tensor (matrix)
matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
plt.imshow(matrix, cmap='viridis')
plt.colorbar()
plt.title("Matrix Visualization")
plt.show()
```

---

## Tensors on GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor_gpu = torch.tensor([1, 2, 3], device=device)
result_gpu = tensor_gpu * 2
```

---

## Neural Network Use Case

```python
import torch.nn as nn
import torch.nn.functional as F

# Dummy data
inputs = torch.rand(5, 3)

# Linear layer
linear = nn.Linear(3, 2)
outputs = linear(inputs)

# Activation (ReLU)
activated_outputs = F.relu(outputs)
```

---

## Requirements

- Python 3.8 or later
- PyTorch
- Matplotlib

Install the requirements:

```bash
pip install torch matplotlib
```

---

