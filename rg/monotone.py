import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, dim=1, device=torch.device('cpu')):
        super().__init__()

        self.device = device

        h = 32
        self.mlp = nn.Sequential(
            nn.Linear(dim, h), 
            nn.LayerNorm(h), 
            nn.SiLU(), 
            nn.Linear(h, dim)
        )

        nn.init.uniform_(self.mlp[0].bias.data, b=2)

        self.to(device)
    
    def forward(self, x):
        return self.mlp(x).squeeze()

class Gaussian(nn.Module):
    def __init__(self, in_channels, device=torch.device('cpu')):
        super().__init__()

        self.device = device

        self.mu = nn.Parameter(torch.zeros(in_channels))
        self.info = nn.Parameter(torch.eye(in_channels))
        
        self.to(device)
    
    def forward(self, x):

        x_shift = x - self.mu

        exp = x_shift.t() @ self.info @ x_shift
        exp *= -1/2
        
        return torch.exp(exp).squeeze()