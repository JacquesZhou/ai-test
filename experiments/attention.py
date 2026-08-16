import torch
import torch.nn as nn
import torch.nn.functional as F




x = torch.randn(1,4,8)
print("x shape:", x.shape)
#2 invent Q,K,V


d_model = 8

W_q = nn.Linear(d_model,d_model,bias=False)
W_k = nn.Linear(d_model,d_model,bias=False)
W_v = nn.Linear(d_model,d_model,bias=False)

Q = W_q(x)
K = W_k(x)
V = W_v(x)

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

#3   Atteneion score

scores = Q @ K.transpose(-2,-1)

print("scores shape:", scores.shape)

#4   Scaling

scores = scores / (d_model ** 0.5)

#5   Softmax

atteneion_weights = F.softmax(scores,dim=-1)

print("attention weights:")
print(atteneion_weights)

#6   weighted value
output = atteneion_weights @ V

print("output shape:", output.shape)
print(output)


