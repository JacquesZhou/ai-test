import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, causal=False):
        super().__init__()
        
        assert d_model % num_heads == 0 
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.head_dim = d_model // num_heads
        
        self.causal = causal
        
        self.q_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )
        
        self.k_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )
        
        self.v_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )
        
        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )
        
    def forward(self, x):
        B, T, D = x.shape
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q = Q.view(
            B,
            T,
            self.num_heads,
            self.head_dim,
        )
        
        K = K.view(
            B,
            T,
            self.num_heads,
            self.head_dim,
        )
        
        V = V.view(
            B,
            T,
            self.num_heads,
            self.head_dim,
        )
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        
        print("scores shape:",scores.shape)
        print("one head scores:")
        print(scores[0, 0])
        
        scores = scores / math.sqrt(self.head_dim)
        
        if self.causal:
            mask = torch.triu(
                torch.ones(
                    T,
                    T,
                    device=x.device,dtype=torch.bool,
                ),
                diagonal=1,
            )
            
            scores = scores.masked_fill(
                mask,
                float("-inf"),
            )
            
            attention_weights = torch.softmax(
                scores,
                dim=-1,
            )
            print("masked scores:")
            print(scores[0, 0])
            
            print("attention weights:")
            print(attention_weights)
            
            out = attention_weights @ V
            
            print("out after attention @ V:")
            print(out.shape)
            
            out = out.transpose(1, 2)
            
            print("after transpose:",out.shape)
            
            out = out.contiguous()
            
            
            
            out = out.view(
                B,
                T,
                D,
            )
            print("after merge:",out.shape)
            
            
            out = self.out_proj(out)
            
            return out
        
        
if __name__ == "__main__":
    x = torch.randn(
        2,
        5,
        8,
    )
    attention = MultiHeadAttention(
        d_model=8,
        num_heads=2,
        causal=True,
    )
    
    y = attention(x)
    
    
    print("input shape:")
    print(x.shape)
    
    print()
    
    print("output shape:")
    print(y.shape)
    
   
    