import torch
import torch.nn as nn

class MultiHeadSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size=64, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.block_size = block_size
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        attn_output = torch.zeros_like(q @ k.transpose(-2, -1))
        
        for i in range(num_blocks):
            start = i * self.block_size
            end = min((i+1)*self.block_size, seq_len)
            
            q_block = q[:, :, start:end, :]
            k_block = k[:, :, start:end, :]
            attn_scores = (q_block @ k_block.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.head_dim)))
            
            if mask is not None:
                attn_mask = mask[:, :, start:end].unsqueeze(1)
                attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
            
            attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
            v_block = v[:, :, start:end, :]
            attn_output[:, :, start:end, :] = (attn_probs @ v_block).transpose(2, 3)
        output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        return output
