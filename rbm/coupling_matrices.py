import torch
import torch.nn as nn

'''
Lieb Coupling Matrices
'''

lieb = {
    'a1': {
        'fine': torch.tensor(
            [[[ 1.,  1.,  0.,  0.], 
              [ 0.,  1.,  1.,  0.], 
              [ 0.,  0.,  1.,  1.], 
              [ 1.,  0.,  0.,  1.]]], 
            requires_grad=False
        ), 
        'coarse': torch.tensor(
            [[[ 1.,  1.,  1.,  1.]]], 
            requires_grad=False
        )
    }, 
    'a1e': {
        'fine': torch.tensor(
            [[[ 1.,  1.,  0.,  0.], 
              [ 0.,  1.,  1.,  0.], 
              [ 0.,  0.,  1.,  1.], 
              [ 1.,  0.,  0.,  1.]], 
             
             [[ 0.,  0.,  0.,  0.], 
              [ 0.,  0.,  0.,  0.], 
              [ 0.,  0.,  0.,  0.], 
              [ 0.,  0.,  0.,  0.]]], 
            requires_grad=False
        ), 
        'coarse': torch.tensor(
            [[[ 1.,  1.,  1.,  1.], 
              [ 0.,  0.,  0.,  0.], 
              [ 0.,  0.,  0.,  0.]], 
             
             [[ 0.,  0.,  0.,  0.], 
              [ 1.,  0., -1.,  0.], 
              [ 0.,  1.,  0., -1.]]], 
            requires_grad=False
        )
    }
}

'''
Hex Coupling Matrices
'''

hex = {
    'a1': {
        'fine': torch.tensor(
            [[[ 1.,  0.,  1.], 
              [ 1.,  1.,  0.], 
              [ 0 ,  1.,  1.]]], 
            requires_grad=False
        ), 
        'coarse': torch.tensor(
            [[[1., 1., 1.]]], 
            requires_grad=False
        )
    }
}
