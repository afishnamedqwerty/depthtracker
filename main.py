import torch
from model.moe_transformer import MoETransformer

def main():
    config = {
        "embed_dim": 512,
        "num_heads": 8,
        "num_experts": 8,
        "dropout": 0.1,
        "block_size": 64,
        # ... other configurations
    }
    
    model = MoETransformer(
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_experts=config['num_experts'],
        dropout=config['dropout'],
        block_size=config['block_size']
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Example input
    batch_size = 32
    seq_len = 16
    x = torch.randn(batch_size, seq_len, config['embed_dim'])
    mask = torch.ones(batch_size, seq_len).bool()  # Dummy mask
    
    # Forward pass
    output = model(x, mask)
    print(f"Output shape: {output.shape}")  # Expected: [32, 16, 512]
    
if __name__ == "__main__":
    main()
