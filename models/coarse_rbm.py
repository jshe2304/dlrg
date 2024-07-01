import torch
import torch.nn as nn

from .spins_rbm import SpinsRBM

class A1_E_RBM(SpinsRBM):
    '''
    Coarse-grained lattice model on A1 representation
    '''
    def __init__(self, device=torch.device('cpu')):
        super().__init__(device)
        
        self.J_A = nn.Parameter(torch.randn(1))
        self.J_E = nn.Parameter(torch.randn(1))
        
        self._coupler_A = torch.Tensor([
            [1, 1, 1, 1], 
            [0, 0, 0, 0], 
            [0, 0, 0, 0]
        ]).detach().to(device)
        self.register_buffer('coupler_A', self._coupler_A)
        
        self._coupler_E = torch.Tensor([
            [0, 0, 0, 0], 
            [1, 0, -1, 0], 
            [0, 1, 0, -1]
        ]).detach().to(device)
        self.register_buffer('coupler_E', self._coupler_E)

        self.to(device)
        
    def W(self):
        '''
        Coupling matrix
        '''
        return self.J_A * self._coupler_A + self.J_E * self._coupler_E

class A1_RBM(SpinsRBM):
    '''
    Coarse-grained lattice model on A1+E representation
    '''
    def __init__(self, device=torch.device('cpu')):
        super().__init__(device)

        self.J = nn.Parameter(torch.randn(1))

        self._coupler = torch.Tensor([
            [1, 1, 1, 1]
        ]).detach().to(device)
        self.register_buffer('coupler', self._coupler)
        
        self.to(device)
        
    def W(self):
        '''
        Coupling matrix
        '''
        return self.J * self._coupler
