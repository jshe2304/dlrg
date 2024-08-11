import torch

'''
Contains a function that returns coupling matrices. 

Coupling matrices are used to map 1D vectors of parameters to their symmetry-obeying coupling matrices. 
'''

def get_coupling_matrix(representation):

    '''
    Return parameter coupling matrices
    '''

    if representation == 'a1':
        
        a1 = torch.tensor(
            [[[1.]]], 
            requires_grad=False
        )

        return a1

    elif representation == 'a1e':

        a1e = torch.tensor(
            [[[1., 0., 0.], 
              [0., 0., 0.], 
              [0., 0., 0.]], 
        
             [[0., 1., 0.], 
              [0., 0., 0.], 
              [0., 0., 0.]], 
        
             [[0., 0., 0.], 
              [1., 0., 0.], 
              [0., 0., 0.]], 
        
             [[0., 0., 0.], 
              [0., 1., 0.], 
              [0., 0., 0.]], 
        
             [[0., 0., 0.], 
              [0., 0., 0.], 
              [0., 0., 1.]]], 
            requires_grad=False
        )

        return a1e