import torch
from torch.func import grad as funcgrad

@torch.enable_grad()
def batch_grad(f):
    '''
    Returns the gradient function of f. 
    Supports batching. 
    '''
    return torch.vmap(funcgrad(f), randomness='same', in_dims=0)

@torch.enable_grad()
def grad(f):
    '''
    Returns the gradient function of f. 
    '''
    return funcgrad(f)