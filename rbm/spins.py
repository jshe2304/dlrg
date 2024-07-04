import torch
import torch.nn as nn
import torch.nn.functional as F

class SpinsRBM(nn.Module):
    '''
    Basic up/down spins RBM with no biases
    '''

    def __init__(self, device):
        super().__init__()
        self.device = device

    def W(self):
        '''
        Returns the coupling matrix. 
        Child RBMs need to override this function. 
        '''
        return

    def energy(self, v, h):
        '''
        Returns the Hamiltonian energy of the RBM. 
        '''
        return -F.bilinear(v, h, self.W().unsqueeze(0))

    def free_energy(self, v):
        '''
        Returns the free energy. 
        '''
        arg = F.linear(v, self.W())

        return -torch.sum(
            torch.log(torch.exp(-arg) + torch.exp(arg)), 
        axis=-1)

    def p_h_given_v(self, v):
        '''
        Returns the conditional distribution p(h | v). 
        v are assumed to be spins, not binary. 
        '''
        return torch.sigmoid(2 * F.linear(v, self.W()))

    def p_v_given_h(self, h):
        '''
        Returns the conditional distribution p(v | h). 
        h are assumed to be spins, not binary. 
        '''
        return torch.sigmoid(2 * F.linear(h, self.W().t()))
        
    def p_v(self, p_v=None, n=1, k=1):
        '''
        Returns the approximate distribution over visible spins via repeated Gibbs sampling. 
        '''

        # If no p(v_0) provided, sample an initial, fully certain distribution
        if p_v is None:
            p_v = torch.randint(0, 2, (n, 4), device=self.device).float()

        # Gibbs sampling
        for i in range(k):
            p_h = self.p_h_given_v(p_v.bernoulli() * 2 - 1)
            p_v = self.p_v_given_h(p_h.bernoulli() * 2 - 1)
        
        return p_v

    def plot_samples(self, n=4, k=1):
        '''
        Plot some lattices sampled from p(v)
        '''

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, n)

        p_v = self.p_v(n=n, k=k)

        for i in range(n):
            axs[i].imshow(p_v[i].reshape(2, 2).detach().cpu(), vmin=0, vmax=1, cmap='binary')
            axs[i].set_xticks([])
            axs[i].set_yticks([])
