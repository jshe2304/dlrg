import torch
import torch.nn as nn

from .rbm import RBM

class Fine_Hex_RBM(RBM):
    '''
    Fine-grained Hexagonal lattice
    '''
    def __init__(self, J=None, device=torch.device('cpu')):
        super().__init__(device)
        
        self.coupler = torch.tensor(
            [[1., 0., 1.], 
             [1., 1., 0.], 
             [0 , 1., 1.]], 
            requires_grad=False, 
            device=device
        )

class A1_Hex_RBM(RBM):
    '''
    Fine-grained Hexagonal lattice
    '''
    def __init__(self, J=None, device=torch.device('cpu')):
        super().__init__(device)
        
        self.coupler = torch.tensor(
            [[1., 1., 1.]], 
            requires_grad=False, 
            device=device
        )
