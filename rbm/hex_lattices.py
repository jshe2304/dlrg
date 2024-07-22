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
        
    @property
    def J(self):
        return self._J
    
    @J.setter
    def J(self, value):
        self._J = value
        self.W = self._J * self.coupler


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
        
    @property
    def J(self):
        return self._J
    
    @J.setter
    def J(self, value):
        self._J = value
        self.W = self._J * self.coupler
