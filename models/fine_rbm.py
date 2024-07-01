import torch
import torch.nn as nn

from .spins_rbm import SpinsRBM

class FineRBM(SpinsRBM):
    '''
    Fine-grained lattice model
    '''
    def __init__(self, J, device=torch.device('cpu')):
        super().__init__(device)

        self._coupler = torch.Tensor([
            [1, 1, 0, 0], 
            [0, 1, 1, 0], 
            [0, 0, 1, 1], 
            [1, 0, 0, 1]
        ]).detach().to(device)
        self.register_buffer('coupler', self._coupler)

        self.J = nn.Parameter(torch.tensor(J).float())

        self.to(device)

    def W(self):
        '''
        Coupling matrix
        '''
        return self.J * self._coupler
