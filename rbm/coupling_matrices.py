import torch
import torch.nn as nn

'''
Lieb Coupling Matrices
'''

fine_lieb_coupler = torch.tensor(
    [[[ 1.,  1.,  0.,  0.], 
      [ 0.,  1.,  1.,  0.], 
      [ 0.,  0.,  1.,  1.], 
      [ 1.,  0.,  0.,  1.]]], 
    requires_grad=False
)

a1_lieb_coupler = torch.tensor(
    [[[ 1.,  1.,  1.,  1.]]], 
    requires_grad=False
)

a1e_lieb_coupler = torch.tensor(
    [[[ 1.,  1.,  1.,  1.], 
      [ 0.,  0.,  0.,  0.], 
      [ 0.,  0.,  0.,  0.]], 
     
     [[ 0.,  0.,  0.,  0.], 
      [ 1.,  0., -1.,  0.], 
      [ 0.,  1.,  0., -1.]]], 
    requires_grad=False
)

'''
Hex Coupling Matrices
'''

fine_hex_coupler = torch.tensor(
    [[[ 1.,  0.,  1.], 
      [ 1.,  1.,  0.], 
      [ 0 ,  1.,  1.]]], 
    requires_grad=False
)

coarse_hex_coupler = torch.tensor(
    [[[1., 1., 1.]]], 
    requires_grad=False
)
