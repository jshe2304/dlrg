import torch

'''
Lieb Adjacency Matrices
'''

lieb_square = torch.tensor(
    [[ 1.,  1.,  0.,  0.], 
     [ 0.,  1.,  1.,  0.], 
     [ 0.,  0.,  1.,  1.], 
     [ 1.,  0.,  0.,  1.]], 
    requires_grad=False
)

lieb_cross = torch.tensor(
    [[ 1.,  1.,  1.,  1.]], 
    requires_grad=False
)