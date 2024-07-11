import torch

@torch.enable_grad()
def grad(f, x, create_graph=True, do_sum=False):
    '''
    Returns the derivative with respect to x
    '''
    
    x.requires_grad_(True)
    
    return torch.autograd.grad(
        f(x).sum() if do_sum else f(x), 
        x, 
        create_graph=create_graph
    )[0]
