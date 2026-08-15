import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# 人造数据 y = 3x + 2
x = torch.randn(10000, 1, device=device)
y = 3 * x + 2

model = nn.Linear(1, 1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(1000):
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch={epoch}, loss={loss.item():.6f}")

print("weight:", model.weight.item())
print("bias:", model.bias.item())
