import torch
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_depth=1, width=32, device=torch.device('cpu')):
        super().__init__()

        self.in_block = nn.Sequential(
            nn.Linear(in_dim, width), 
            nn.LayerNorm(width), 
            nn.SiLU(), 
        )

        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(width, width), 
                nn.LayerNorm(width), 
                nn.SiLU()
            ) for _ in range(hidden_depth)
        ])

        self.out_block = nn.Linear(width, out_dim)

        self.device = device
        self.to(device)
    
    def forward(self, x, sum=False):

        x = self.in_block(x)

        for res_block in self.res_blocks:
            x = x + res_block(x)
        
        x = self.out_block(x)
        
        return x.sum() if sum else x

class PowerSkipResNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_depth=1, width=32, p=2, device=torch.device('cpu')):
        super().__init__()

        self.p = p

        self.in_block = nn.Sequential(
            nn.Linear(in_dim, width), 
            nn.LayerNorm(width), 
            nn.SiLU(), 
        )

        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(width, width), 
                nn.LayerNorm(width), 
                nn.SiLU()
            ) for _ in range(hidden_depth)
        ])

        self.out_block = nn.Linear(width, out_dim)

        self.device = device
        self.to(device)
    
    def forward(self, x, sum=False):

        x = self.in_block(x)

        for i, res_block in enumerate(self.res_blocks):
            if i % 2 == 0:
                x = x ** self.p + res_block(x)
            else:
                x = x + res_block(x)
        
        x = self.out_block(x)
        
        return x.sum() if sum else x
