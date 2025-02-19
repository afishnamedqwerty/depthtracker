import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc1(torch.nn.functional.gelu(x))

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        return torch.nn.functional.softmax(self.fc(x), dim=-1)

class MoELayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=8):
        super().__init__()
        self.expert = Expert(input_dim, output_dim)
        self.gating = GatingNetwork(input_dim, num_experts)
    
    def forward(self, x):
        gate = self.gating(x)
        expert_output = torch.stack([self.expert(x * g) for g in gate.unbind(dim=-1)], dim=-1)
        output = (expert_output * gate.unsqueeze(-1)).sum(dim=-2)
        return output
