import torch
import torch.nn as nn

class ResidualMLP(nn.Module):

    def __init__(self, in_channels, device=torch.device('cpu')):
        super().__init__()

        self.device = device

        h = 4
        
        self.in_block = nn.Sequential(
            nn.Linear(in_channels, h), 
            nn.LayerNorm(h), 
            nn.SiLU(), 
        )
        
        self.res_block = nn.Sequential(
            nn.Linear(h, h), 
            nn.LayerNorm(h), 
            nn.SiLU(), 
        )

        self.out_block = nn.Sequential(
            nn.Linear(h, h), 
            nn.LayerNorm(h), 
            nn.SiLU(), 
            nn.Linear(h, 1)
        )
        
        self.to(device)
    
    def forward(self, x):
        x1 = self.in_block(x)
        x2 = x1 + self.res_block(x1)
        x3 = self.out_block(x2)
        
        return x3
