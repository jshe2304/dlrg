import torch
import torch.nn as nn
import torch.nn.functional as F

class SpinsRBM(nn.Module):
    '''
    Basic spins RBM
    '''

    def __init__(self, device):
        super().__init__()
        
        self.device = device

    def W(self):
        '''
        Return the coupling matrix
        '''
        return

    def energy(self, v, h):
        '''
        Returns the Hamiltonian energy of the RBM
        '''
        return -F.bilinear(v, h, self.W().unsqueeze(0))

    def free_energy(self, v):
        '''
        Returns the free energy
        '''
        exp = torch.exp(F.linear(v, self.W().t()))
        return -torch.sum(torch.log(1 + exp))

    def h_given_v(self, v):
        '''
        Returns the conditional distribution P(h | v). 
        v are assumed to be spins, not binary. 
        '''
        return torch.sigmoid(2 * F.linear(v, self.W()))
    
    def v_given_h(self, h):
        '''
        Returns the conditional distribution P(v | h). 
        h are assumed to be spins, not binary. 
        '''
        return torch.sigmoid(2 * F.linear(h, self.W().t()))

    def forward(self, v, k=1):
        '''
        Returns the distribution over visible spins after k Gibbs sampling steps. 
        '''
        for i in range(k):
            h = self.h_given_v(v.bernoulli() * 2 - 1)
            v = self.v_given_h(h.bernoulli() * 2 - 1)
        
        return v

    def joint(self, h_0=None, n=1, k=32):
        '''
        Approximates the distribution of visible spins via repeated Gibbs sampling
        '''
        if h_0 is None:
            h_0 = torch.randint(0, 2, (n, 4)).float() * 2 - 1
        h_0 = h_0.to(self.device)

        v_0 = self.v_given_h(h_0)
        
        return self.forward(v_0, k=k)
    