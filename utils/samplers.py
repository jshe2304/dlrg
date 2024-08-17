import torch
from .grad import grad, batch_grad

class HMC():
    '''
    Hamiltonian Monte Carlo sampler
    '''
    def __init__(self, dt=0.001, runtime=32, steps=1, device=torch.device('cpu')):
        self.device = device

        self.dt = dt
        self.runtime = runtime
        self.steps = steps
        
        self.potential = lambda J : 0
        self.kinetic = lambda v: (v ** 2) / 2
    
    @torch.no_grad
    def solve(self, x, v):
        '''
        Solve Hamiltonian dynamics
        '''

        force = grad(lambda _x : self.potential(_x).sum())
        #force = batch_grad(self.potential)

        v -= 0.5 * self.dt * force(x)
        x += self.dt * v
        
        for _ in range(self.runtime):
            v -= self.dt * force(x)
            x += self.dt * v
    
        v -= 0.5 * self.dt * force(x)
        
        return x, v

    @torch.no_grad
    def step(self, x_0):
        '''
        MCMC proposal step
        '''
        
        for _ in range(self.steps):
            batch_size, in_dims = x_0.shape
            
            v_0 = torch.randn_like(x_0, device=self.device)
            H_0 = self.potential(x_0).reshape(batch_size, 1) + self.kinetic(v_0)
    
            x, v = self.solve(x_0, v_0)
            H = self.potential(x).reshape(batch_size, 1) + self.kinetic(v)
    
            p_accept = torch.exp(H_0 - H)
    
            mask = p_accept > torch.rand_like(p_accept, device=self.device)
    
            x_0 = torch.where(mask, x, x_0)

        return x_0


