import torch

'''
Contains a function that returns adjacency matrices. 

Adjacency matrices are used to define the lattice connectivity for RBM models. 
'''

def get_adjacency_matrices(lattice):
    '''
    Return lattice adjacency matrix
    '''

    if lattice == 'lieb':

        square = torch.tensor(
            [[ 1.,  1.,  0.,  0.], 
             [ 0.,  1.,  1.,  0.], 
             [ 0.,  0.,  1.,  1.], 
             [ 1.,  0.,  0.,  1.]], 
            requires_grad=False
        )
        
        cross = torch.tensor(
            [[ 1.,  1.,  1.,  1.]], 
            requires_grad=False
        )

        return square, cross