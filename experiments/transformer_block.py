import torch
import torch.nn as nn

from multihead_attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    
    def __init__(self, d_model,num_heads, d_ff):
        super().__init__()
        
        #first layernorm
        self.ln1 = nn.LayerNorm(d_model)
        
        self.attention = MultiHeadAttention(d_model=d_model,
                                            num_heads=num_heads,
                                            causal=True,
                                            )
        
        self.ln2 = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            
        )
        
        def forward(self, x):
            norm_x = self.ln1(x)
        
            attention_out = self.attention(norm_x)
        
            x = x + attention_out
        
            norm_x = self.ln2(x)
            mlp_out = self.mlp(norm_x)
            x = x + mlp_out
        
            return x