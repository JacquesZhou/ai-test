import torch

x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

bias = torch.tensor([10, 20, 30])

print(x + bias)
