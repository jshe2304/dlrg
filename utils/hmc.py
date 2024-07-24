import torch
from .grad import grad

class HMC():
    '''
    Hamiltonian Monte Carlo sampler
    '''
    def __init__(self, dim=1, device=torch.device('cpu')):
        self.dim = dim
        self.device = device
        
        self.potential = lambda J : 0
        self.kinetic = lambda v: (v ** 2) / 2
    
    @torch.no_grad
    def solve(self, x, v, dt=0.001, runtime=32):
        '''
        Solve Hamiltonian dynamics over specified runtime
        '''
        v -= 0.5 * dt * grad(self.potential)(x)
        x += dt * v
        
        for t in range(runtime):
            v -= dt * grad(self.potential)(x)
            x += dt * v
    
        v -= 0.5 * dt * grad(self.potential)(x)
        
        return x, v

    def step(self, x_0):
        '''
        MCMC proposal step
        '''
        v_0 = torch.randn_like(x_0, device=self.device)
        H_0 = self.potential(x_0) + self.kinetic(v_0)

        x, v = self.solve(x_0, v_0)
        H = self.potential(x) + self.kinetic(v)

        p_accept = torch.exp(H_0 - H)
        if torch.all(p_accept > torch.rand_like(p_accept, device=self.device)):
            return x

        return x_0
