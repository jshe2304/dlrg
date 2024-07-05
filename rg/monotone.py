import torch
import torch.nn as nn

class Monotone(nn.Module):
    '''
    RG Monotone NeuralODE model
    '''
    
    def __init__(self, in_channels, device=torch.device('cpu')):
        super().__init__()
        
        self.in_channels = in_channels
        self.device = device
        
        self.f = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2), 
            nn.LayerNorm(in_channels * 2), 
            nn.Tanh(), 
            
            nn.Linear(in_channels * 2, in_channels * 4), 
            nn.LayerNorm(in_channels * 4), 
            nn.Tanh(), 
            
            nn.Linear(in_channels * 4, in_channels * 8), 
            nn.LayerNorm(in_channels * 8), 
            nn.Tanh(), 
            
            nn.Linear(in_channels * 8, in_channels * 4), 
            nn.LayerNorm(in_channels * 4), 
            nn.Tanh(), 
            
            nn.Linear(in_channels * 4, in_channels * 2), 
            nn.LayerNorm(in_channels * 2), 
            nn.Tanh(), 
            
            nn.Linear(in_channels * 2, in_channels), 
            nn.LayerNorm(in_channels), 
            nn.Tanh(), 
        )
        
        self.to(device)
        
    def forward(self, J):
        
        return self.f(J) - self.f(torch.zeros_like(J)).detach()
    
    