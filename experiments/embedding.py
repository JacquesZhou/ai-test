import torch

import torch.nn as nn

embedding = nn.Embedding(
    num_embeddings=10000,
    embedding_dim=512
)
print(embedding.weight.shape)

tokens = torch.tensor([
    [15, 83, 243]
])

x = embedding(tokens)

print("x=", x.shape)