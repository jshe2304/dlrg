import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, in_channels, device=torch.device('cpu')):
        super().__init__()

        self.device = device

        h = 16
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, h), 
            nn.LayerNorm(h), 
            nn.SiLU(), 
            nn.Linear(h, 1)
        )
        
        self.to(device)
    
    def forward(self, x):
        return self.mlp(x).squeeze()
