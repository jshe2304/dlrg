'''
Newton's root-finding algorithm
'''

import torch
from .grad import grad, jacobian

@torch.no_grad()
def find_root(f, x, eps=1e-7, max_iters=1000):
    
    for i in range(max_iters):
        if torch.norm(fx := f(x)) < eps: break

        J = jacobian(f)(x)
        if J.det() == 0: return torch.empty_like(x).fill_(float('nan'))
        
        x = x - J.inverse() @ fx

    return x