'''
Newton's root-finding algorithm
'''

import torch
from .grad import grad, jacobian

@torch.no_grad()
def find_root(f, x, flow_assist=None, eps=1e-7, max_iters=512):
    
    for i in range(max_iters):

        # Root Found
        if torch.norm(fx := f(x)) < eps:
            return x

        # Newton Step

        J = jacobian(f, randomness='different')(x)
        if J.det() != 0: 
            x = x - J.inverse() @ fx

        # Handle singular Jacobian
        
        elif flow_assist is not None:
            print('assist')
            x = x - fx * flow_assist
        
        else: break
    
    return torch.empty_like(x).fill_(float('nan'))

    
