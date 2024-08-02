import torch
import torch.nn as nn

class MLP1(nn.Module):

    def __init__(self, dim=1, h=32, device=torch.device('cpu')):
        super().__init__()

        self.device = device

        self.mlp = nn.Sequential(
            nn.Linear(dim, h), 
            nn.LayerNorm(h), 
            nn.SiLU(), 
            nn.Linear(h, dim)
        )

        nn.init.uniform_(self.mlp[0].bias.data, a=-2, b=2)

        self.to(device)
    
    def forward(self, x):
        return self.mlp(x)

class MLP2(nn.Module):

    def __init__(self, dim=1, h=32, device=torch.device('cpu')):
        super().__init__()

        self.device = device

        self.mlp = nn.Sequential(
            nn.Linear(dim, h), 
            nn.LayerNorm(h), 
            nn.SiLU(), 
            nn.Linear(h, h), 
            nn.LayerNorm(h), 
            nn.SiLU(), 
            nn.Linear(h, dim)
        )
        
        self.to(device)
    
    def forward(self, x):
        return self.mlp(x)
