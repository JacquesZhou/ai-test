import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# 1. 创建数据
x = torch.randn(10000, 2, device=device)

# 要学习的非线性函数
y = (
    x[:, 0] ** 2
    + 2 * x[:, 1]
).unsqueeze(1)


# 2. 定义神经网络
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


model = MLP().to(device)


# 3. 损失函数
loss_fn = nn.MSELoss()


# 4. 优化器
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
)


# 5. 训练
for epoch in range(1000):

    prediction = model(x)

    loss = loss_fn(prediction, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 100 == 0:
        print(
            f"epoch={epoch}, "
            f"loss={loss.item():.6f}"
        )


print("Training finished.")
