import torch
from .multi_head_attention import MultiHeadSparseAttention
from .moe_layer import MoELayer

class MoETransformer(torch.nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_experts=8, dropout=0.1, block_size=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.dropout = dropout
        self.block_size = block_size
        
        # Multi-head sparse attention layer
        self.mha = MultiHeadSparseAttention(embed_dim, num_heads, block_size, dropout)
        
        # Add & norm layers with efficient memory access
        self.add_norm = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.LayerNorm(embed_dim)
        )
        
        # MoE layer with GEGLU activation and sparse attention
        self.moe_layer = MoELayer(embed_dim, embed_dim, num_experts)
    
    def forward(self, x, mask=None):
        # Multi-head sparse attention
        attn_output = self.mha(x, mask)
        
        # Add & norm with efficient memory access
        x = self.add_norm(attn_output + x)  # Residual connection
        
        # MoE layer with GEGLU activation and sparse attention
        moe_output = self.moe_layer(x)
        
        return moe_output
