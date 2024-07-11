import torch
import torch.nn as nn

class ResidualMLP(nn.Module):

    def __init__(self, in_channels, device=torch.device('cpu')):
        super().__init__()

        self.device = device
        
        self.in_block = nn.Sequential(
            nn.Linear(in_channels, 4), 
            nn.LayerNorm(4), 
            nn.SiLU(), 
        )
        
        self.res_block = nn.Sequential(
            nn.Linear(4, 4), 
            nn.LayerNorm(4), 
            nn.SiLU(), 
        )

        self.out_block = nn.Sequential(
            nn.Linear(4, 4), 
            nn.LayerNorm(4), 
            nn.SiLU(), 
            nn.Linear(4, 1)
        )
        
        self.to(device)
    
    def forward(self, x):
        x1 = self.in_block(x)
        x2 = x1 + self.res_block(x1)
        x3 = self.out_block(x2)
        
        return x3
