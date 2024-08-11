import torch

'''
Contains a function that returns a tensor of Ising state configurations. 

Configuration matrices are used to parallel compute partition functions. 
'''

def get_configurations(n, device=torch.device('cpu')):

    if n == 1:
        return torch.tensor(
            [[-1.], 
             [ 1.]], 
            requires_grad=False, 
            dtype=torch.float32, 
            device=device
        )

    elif n == 4:
        return torch.tensor(
            [[-1., -1., -1., -1.], 
             [-1., -1., -1.,  1.], 
             [-1., -1.,  1., -1.], 
             [-1., -1.,  1.,  1.], 
             [-1.,  1., -1., -1.], 
             [-1.,  1.,  1., -1.], 
             [-1.,  1., -1.,  1.], 
             [-1.,  1.,  1.,  1.], 
             [ 1., -1., -1., -1.], 
             [ 1., -1., -1.,  1.], 
             [ 1., -1.,  1., -1.], 
             [ 1., -1.,  1.,  1.], 
             [ 1.,  1., -1., -1.], 
             [ 1.,  1.,  1., -1.], 
             [ 1.,  1., -1.,  1.], 
             [ 1.,  1.,  1.,  1.]], 
            requires_grad=False, 
            dtype=torch.float32, 
            device=device
        )

    return
    