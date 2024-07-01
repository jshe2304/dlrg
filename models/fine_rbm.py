import torch
import torch.nn as nn
import torch.nn.functional as F

class FineRBM(nn.Module):
    '''
    Fine-grained lattice model
    '''
    def __init__(self, J):
        super().__init__()
        
        self.J = J * torch.Tensor([
            [1, 1, 0, 0], 
            [0, 1, 1, 0], 
            [0, 0, 1, 1], 
            [1, 0, 0, 1]
        ]).detach()

    def energy(self, x, z):
        '''
        Returns the Hamiltonian energy of the RBM
        '''
        return -F.bilinear(x, z, self.J.unsqueeze(0))

    def free_energy(self, x):
        '''
        Returns the free energy
        '''
        exp = torch.exp(F.linear(x, self.J.t()))
        return -torch.sum(torch.log(1 + exp))

    def z_given_x(self, x):
        '''
        Returns the conditional distribution P(z | x). 
        x are assumed to be spins, not binary. 
        '''
        return torch.sigmoid(2 * F.linear(x, self.J.t()))
    
    def x_given_z(self, z):
        '''
        Returns the conditional distribution P(x | z). 
        z are assumed to be spins, not binary. 
        '''
        return torch.sigmoid(2 * F.linear(z, self.J))
    
    def forward(self, x, k=1):
        '''
        Returns the distribution over visible spins after k Gibbs sampling steps. 
        '''
        for i in range(k):
            z = self.z_given_x(x.bernoulli() * 2 - 1)
            x = self.x_given_z(z.bernoulli() * 2 - 1)
        
        return x
    
    def joint(self, z_0=None, n=1, k=32):
        '''
        Approximates the distribution of visible spins via repeated Gibbs sampling
        '''
        if z_0 is None:
            z_0 = torch.randint(0, 2, (n, 4)).float() * 2 - 1

        x_0 = self.x_given_z(z_0)
        
        return self.forward(x_0, k=k)
