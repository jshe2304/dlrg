import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_depth=1, width=32, device=torch.device('cpu')):
        super().__init__()

        self.in_block = nn.Sequential(
            nn.Linear(in_dim, width), 
            #nn.LayerNorm(width), 
            nn.ReLU(), 
        )

        self.mlp_block = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(width, width), 
                #nn.LayerNorm(width), 
                nn.ReLU()
            ) for _ in range(hidden_depth)
        ])

        self.out_block = nn.Linear(width, out_dim)

        self.device = device
        self.to(device)
    
    def forward(self, x):

        x1 = self.in_block(x)
        x2 = self.mlp_block(x1)
        x3 = self.out_block(x2)
        
        return x3

class AntisymmetricMLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_depth=1, width=32, device=torch.device('cpu')):
        super().__init__()

        self.in_block = nn.Sequential(
            nn.Linear(in_dim, width), 
            nn.LayerNorm(width), 
            nn.ReLU(), 
        )

        self.mlp_block = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(width, width), 
                nn.LayerNorm(width), 
                nn.ReLU()
            ) for _ in range(hidden_depth)
        ])

        self.out_block = nn.Linear(width, out_dim)

        self.device = device
        self.to(device)
    
    def forward(self, x):

        x1 = self.in_block(torch.abs(x))
        x2 = self.mlp_block(x1)
        x3 = self.out_block(x2)
        
        return x3 * x.sign()
