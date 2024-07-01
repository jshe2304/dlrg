import torch
import torch.nn as nn
import torch.nn.functional as F

class CoarseRBM(nn.Module):
    '''
    Fine-grained lattice model
    '''
    def __init__(self):
        super().__init__()
        
        self.J_A = nn.Parameter(torch.rand(1))
        self.J_E = nn.Parameter(torch.rand(1))
        
        self._coupler_A = torch.Tensor([
            [1, 1, 1, 1], 
            [0, 0, 0, 0], 
            [0, 0, 0, 0]
        ]).detach()
        
        self._coupler_E = torch.Tensor([
            [0, 0, 0, 0], 
            [1, 0, -1, 0], 
            [0, 1, 0, -1]
        ]).detach()
        
    def J(self):
        '''
        Return 2-parameter coupling matrix
        '''
        return self.J_A * self._coupler_A + self.J_E * self._coupler_E
        
    def hamiltonian(self, x, z):
        '''
        Returns the Hamiltonian energy of the RBM
        '''
        return -F.bilinear(x, z, self.J())
        
    def z_given_x(self, x):
        '''
        Returns the conditional distribution P(z | x)
        '''
        return torch.sigmoid(2 * F.linear(x, self.J()))
    
    def x_given_z(self, z):
        '''
        Returns the conditional distribution P(x | z)
        '''
        return torch.sigmoid(2 * F.linear(z, self.J().t()))
    
    def forward(self, x, k=1):
        '''
        Returns the distribution over visible spins after k Gibbs sampling steps. 
        '''
        for i in range(k):
            z = self.z_given_x(x.bernoulli() * 2 - 1)
            x = self.x_given_z(z.bernoulli() * 2 - 1)
        
        return x
    
    def sample(self, z_0=None, n=1, k=32):
        '''
        Samples from the joint distribution via repeated Gibbs sampling
        '''
        if z_0 is None:
            z_0 = torch.randint(0, 2, (n, 4)).float() * 2 - 1

        x_0 = self.x_given_z(z_0)
        
        return self.forward(x_0, k=k)
