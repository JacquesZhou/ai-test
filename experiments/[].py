import torch
x = torch.arange(24).reshape(2, 3, 4)

print(x.shape)
print(x[0])
print(x[0, 1])


#2unsqueeze and squeeze

