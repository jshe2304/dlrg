import torch
import torch.nn as nn

class ResidualMLP(nn.Module):

    def __init__(self, in_channels, device=torch.device('cpu')):
        super().__init__()
        
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
    
    @torch.enable_grad()
    def grad(self, x):
        '''
        Returns the derivative with respect to x
        '''
        
        assert (x.dim == 1 or x.size(0) == 1)
        
        x.requires_grad_(True)
        
        return torch.autograd.grad(
            self(x), 
            x, 
            create_graph=True
        )[0]