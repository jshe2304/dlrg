'''
Newton's root-finding algorithm
'''

import torch

def find_root(x, f, eps=1e-7, max_iters=1000):
    
    for i in range(max_iters):

        fx = f(x)
        
        if torch.abs(fx) < eps: break
        
        dfdx = torch.autograd.grad(fx, x)[0]
        x = x - (fx / dfdx).detach()
        
    return float(x)