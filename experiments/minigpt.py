import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 超参数
# ============================================================

batch_size = 32       # 每次训练多少段文本
block_size = 64       # 每段文本最多包含多少个 token
d_model = 128         # 每个 token 的向量维度
num_heads = 4         # Multi-Head Attention 的 head 数量
num_layers = 4        # Transformer Block 数量
d_ff = 512            # MLP 中间层维度
dropout = 0.1

learning_rate = 3e-4
train_steps = 3000

device = "cuda" if torch.cuda.is_available() else "cpu"

print("device:", device)


# ============================================================
# 2. 读取训练文本
# ============================================================

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("text length:", len(text))


# ============================================================
# 3. 字符级 Tokenizer
# ============================================================

# 找到文本中所有不同字符
chars = sorted(list(set(text)))

vocab_size = len(chars)

print("vocab size:", vocab_size)


# 字符 -> token id
stoi = {ch: i for i, ch in enumerate(chars)}

# token id -> 字符
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    """
    字符串 -> token id
    例如：
    "hello"
        ↓
    [15, 12, 19, 19, 22]
    """
    return [stoi[c] for c in s]


def decode(tokens):
    """
    token id -> 字符串
    """
    return "".join(itos[i] for i in tokens)


# 把整份文本转换成 tensor
data = torch.tensor(
    encode(text),
    dtype=torch.long
)


# 90% 训练
split = int(len(data) * 0.9)

train_data = data[:split]

# 10% 验证
val_data = data[split:]


# ============================================================
# 4. 构造训练 batch
# ============================================================

def get_batch(split):

    if split == "train":
        source = train_data
    else:
        source = val_data

    # 随机选择 batch_size 个起点
    starts = torch.randint(
        0,
        len(source) - block_size - 1,
        (batch_size,)
    )

    # 输入
    x = torch.stack([
        source[i:i + block_size]
        for i in starts
    ])

    # target 比 input 向右移动一个 token
    y = torch.stack([
        source[i + 1:i + block_size + 1]
        for i in starts
    ])

    return x.to(device), y.to(device)


# ============================================================
# 5. 手写 Multi-Head Attention
# ============================================================

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads

        # 每个 head 的维度
        self.head_dim = d_model // num_heads

        # 把 x 映射成 Q K V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # 多个 head 拼接以后再进行一次线性变换
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # x:
        # [batch, seq_len, d_model]

        B, T, C = x.shape

        # ----------------------------------------------------
        # 生成 Q K V
        # ----------------------------------------------------

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 当前：
        #
        # q/k/v
        # [B, T, d_model]


        # ----------------------------------------------------
        # 拆成多个 head
        # ----------------------------------------------------

        q = q.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        )

        k = k.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        )

        v = v.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        )

        # 把 head 放到前面
        #
        # [B, T, heads, head_dim]
        #
        # ->
        #
        # [B, heads, T, head_dim]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)


        # ----------------------------------------------------
        # Attention Score
        # ----------------------------------------------------

        # Q @ K^T
        #
        # [B, heads, T, head_dim]
        #
        # @
        #
        # [B, heads, head_dim, T]
        #
        # =
        #
        # [B, heads, T, T]

        scores = q @ k.transpose(-2, -1)

        # scaled dot-product attention
        scores = scores / (self.head_dim ** 0.5)


        # ----------------------------------------------------
        # Causal Mask
        # ----------------------------------------------------

        # GPT 不能看到未来 token
        #
        # 例如：
        #
        # token1 可以看 token1
        # token2 可以看 token1 token2
        # token3 可以看 token1 token2 token3

        mask = torch.tril(
            torch.ones(
                T,
                T,
                device=x.device
            )
        )

        # 未来位置设为 -inf
        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )


        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        attention_weights = self.dropout(
            attention_weights
        )


        # ----------------------------------------------------
        # Attention × V
        # ----------------------------------------------------

        out = attention_weights @ v

        # out:
        #
        # [B, heads, T, head_dim]


        # ----------------------------------------------------
        # 多个 head 拼回来
        # ----------------------------------------------------

        out = out.transpose(1, 2)

        # [B, T, heads, head_dim]

        out = out.contiguous().view(
            B,
            T,
            C
        )

        # [B, T, d_model]


        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        out = self.out_proj(out)

        return out


# ============================================================
# 6. Transformer Block
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff
    ):
        super().__init__()

        # Attention 前 LayerNorm
        self.ln1 = nn.LayerNorm(d_model)

        self.attention = MultiHeadAttention(
            d_model,
            num_heads
        )

        # MLP 前 LayerNorm
        self.ln2 = nn.LayerNorm(d_model)

        # Feed Forward Network
        self.mlp = nn.Sequential(

            # 升维
            nn.Linear(
                d_model,
                d_ff
            ),

            # GPT 常用 GELU
            nn.GELU(),

            # 降回 d_model
            nn.Linear(
                d_ff,
                d_model
            ),

            nn.Dropout(dropout)
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Attention + Residual
        # ----------------------------------------------------

        x = x + self.attention(
            self.ln1(x)
        )

        # ----------------------------------------------------
        # MLP + Residual
        # ----------------------------------------------------

        x = x + self.mlp(
            self.ln2(x)
        )

        return x


# ============================================================
# 7. MiniGPT
# ============================================================

class MiniGPT(nn.Module):

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # Token Embedding
        # ----------------------------------------------------

        # token id
        #
        # 例如：
        #
        # 23
        #
        # ->
        #
        # [0.21, -0.13, ...]
        #
        # 长度 d_model

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )


        # ----------------------------------------------------
        # Position Embedding
        # ----------------------------------------------------

        # 告诉 Transformer：
        #
        # token 在第几个位置

        self.position_embedding = nn.Embedding(
            block_size,
            d_model
        )


        # ----------------------------------------------------
        # Transformer Blocks
        # ----------------------------------------------------

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff
                )
                for _ in range(num_layers)
            ]
        )


        # 最后一层 LayerNorm
        self.final_ln = nn.LayerNorm(d_model)


        # ----------------------------------------------------
        # Language Model Head
        # ----------------------------------------------------

        # d_model
        #
        # ->
        #
        # vocab_size
        #
        # 给每个 token 一个预测分数

        self.lm_head = nn.Linear(
            d_model,
            vocab_size
        )


    def forward(self, idx, targets=None):

        # idx:
        #
        # [batch, seq_len]

        B, T = idx.shape


        # ----------------------------------------------------
        # Token Embedding
        # ----------------------------------------------------

        token_emb = self.token_embedding(idx)

        # [B, T, d_model]


        # ----------------------------------------------------
        # Position Embedding
        # ----------------------------------------------------

        positions = torch.arange(
            T,
            device=idx.device
        )

        position_emb = self.position_embedding(
            positions
        )

        # [T, d_model]


        # ----------------------------------------------------
        # Token + Position
        # ----------------------------------------------------

        x = token_emb + position_emb

        # [B, T, d_model]


        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        x = self.blocks(x)


        # ----------------------------------------------------
        # Final LayerNorm
        # ----------------------------------------------------

        x = self.final_ln(x)


        # ----------------------------------------------------
        # Language Model Head
        # ----------------------------------------------------

        logits = self.lm_head(x)

        # logits:
        #
        # [B, T, vocab_size]


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = None

        if targets is not None:

            # CrossEntropy 需要：
            #
            # prediction:
            # [N, vocab_size]
            #
            # target:
            # [N]

            logits_flat = logits.view(
                B * T,
                vocab_size
            )

            targets_flat = targets.view(
                B * T
            )

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss


    # ========================================================
    # 文本生成
    # ========================================================

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens
    ):

        for _ in range(max_new_tokens):

            # GPT 最多只能处理 block_size 个 token
            idx_crop = idx[:, -block_size:]


            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            logits, _ = self(
                idx_crop
            )


            # ------------------------------------------------
            # 只取最后一个位置
            # ------------------------------------------------

            logits = logits[:, -1, :]

            # [B, vocab_size]


            # ------------------------------------------------
            # 转为概率
            # ------------------------------------------------

            probs = F.softmax(
                logits,
                dim=-1
            )


            # ------------------------------------------------
            # 按概率随机选择下一个 token
            # ------------------------------------------------

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            # [B, 1]


            # ------------------------------------------------
            # 拼到原来的 token 后面
            # ------------------------------------------------

            idx = torch.cat(
                [idx, next_token],
                dim=1
            )

        return idx


# ============================================================
# 8. 创建模型
# ============================================================

model = MiniGPT().to(device)

print(model)


# 查看参数量
num_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"parameters: {num_parameters / 1e6:.2f} M"
)


# ============================================================
# 9. Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# ============================================================
# 10. 训练
# ============================================================

model.train()

for step in range(train_steps):

    # 得到一批训练数据
    x, y = get_batch("train")


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    logits, loss = model(
        x,
        y
    )


    # --------------------------------------------------------
    # 清空上一轮梯度
    # --------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )


    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    loss.backward()


    # --------------------------------------------------------
    # 更新参数
    # --------------------------------------------------------

    optimizer.step()


    # 每 100 step 打印一次 loss
    if step % 100 == 0:

        print(
            f"step {step:4d} | "
            f"loss {loss.item():.4f}"
        )


# ============================================================
# 11. 文本生成
# ============================================================

model.eval()


# 取训练文本的第一个字符作为起点
start_token = torch.tensor(
    [[encode(text[0])[0]]],
    dtype=torch.long,
    device=device
)


generated = model.generate(
    start_token,
    max_new_tokens=500
)


result = decode(
    generated[0].tolist()
)

print("\n==============================")
print("Generated text")
print("==============================")

print(result)