#attention people
import torch
import math

B = 1
T = 3
head_dim = 4

Q = torch.randn(B, T, head_dim)
K = torch.randn(B, T, head_dim)
V = torch.randn(B, T, head_dim)

print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)

scores = Q @ K.transpose(-2, -1)

print("scores:",scores.shape)

scores = scores / math.sqrt(head_dim)

weights = torch.softmax(scores, dim=-1)

print("weights:")
print(weights)

output = weights @ V

print("output:", output.shape)





