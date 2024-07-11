'''
Newton's root-finding algorithm
'''

import torch

def find_root(x, f, eps=1e-7):
    
    fx = f(x)
    while torch.abs(fx) > eps:
        
        dfdx = torch.autograd.grad(fx, x)[0]
        x = x - (fx / dfdx).detach()
        fx = f(x)
        
    return float(x)