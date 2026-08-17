#to understand tensor
import torch
#1 reshape
x = torch.arange(12)

print(x)
print(x.shape)

y = x.reshape(3, 4)
print(y)
print(y.shape)

#2 multiplication @ :scores = Q @ K.T
#3 softmax 
scores = torch.tensor([
    [1., 2., 3.],
    [2., 1., 0.]
])

weights = torch.softmax(scores, dim=-1)
print(weights)
print(weights.sum(dim=-1))
#dim3 tensors
x = torch.randn(2, 4, 8)
print(x.shape)
#2=batchsize;4=sequencelength;8=d_model;;b,t,c

#4 split d_model into heads
B = 2
T = 4
d_model = 8
num_heads = 2

head_dim = d_model // num_heads

x = torch.randn(B, T, d_model)

print("before:", x.shape)

x = x.reshape(B, T, num_shape)


#[]
import torch
x = torch.arange(24).reshape(2, 3, 4)

print(x.shape)
print(x[0])
print(x[0, 1])