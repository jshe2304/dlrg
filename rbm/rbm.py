import torch
import torch.nn as nn
import torch.nn.functional as F

class RBM(nn.Module):
    '''
    A base spins RBM model for inheritance by specified RBMs. 
    No biases are used. 

    J                : (models  , parameters          )
    adjacency_matrix : (hidden  , visible             )
    W                : (models  , hidden     , visible)
    p_h, v           : (samples , hidden              )
    p_v, h           : (samples , visible             )
    '''

    def __init__(self, adjacency_matrix, coupling_matrix, device=torch.device('cpu')):
        super().__init__()

        self.register_buffer('adjacency_matrix', adjacency_matrix)
        self.register_buffer('coupling_matrix', coupling_matrix)

        self.n_parameters, self.n_dimensions, self.n_dimensions = coupling_matrix.shape
        
        self.device = device
        self.to(device)

    @property
    def J(self):
        return self._J
    
    @J.setter
    def J(self, J):
        '''
        Compute the (dim, dim) matrix from a (models, parameters) vector. 
        Compute the (models, hidden * dim, visible * dim) weight matrix. 
        '''
        
        self.n_models, _ = J.shape
        
        self._J = torch.tensordot(
            J, self.coupling_matrix, 
            dims=((1, ), (0, ))
        ).reshape(self.n_models, self.n_dimensions, self.n_dimensions)

        self.W = torch.kron(self.adjacency_matrix, self._J)

        self.n_models, self.n_hidden, self.n_visible = self.W.shape

    def free_energy(self, v):
        '''
        Returns the free energy. 
        '''
        
        # v @ self.W.t()
        arg = torch.bmm(v, self.W.permute(0, 2, 1))

        return -torch.sum(
            torch.log(torch.exp(-arg) + torch.exp(arg)), 
        axis=-1)

    def p_h_given_v(self, v):
        '''
        Returns the conditional distribution p(h | v). 
        v are assumed to be spins, not binary. 
        '''
        
        # v @ self.W.t()
        arg = torch.bmm(v, self.W.permute(0, 2, 1))
        
        return torch.sigmoid(2 * arg)

    def p_v_given_h(self, h):
        '''
        Returns the conditional distribution p(v | h). 
        h are assumed to be spins, not binary. 
        '''
        
        # h @ self.W
        arg = torch.bmm(h, self.W)
        
        return torch.sigmoid(2 * arg)
    
    def p_v(self, p_v=None, n=1, k=1):
        '''
        Returns the approximate distribution over visible spins via repeated Gibbs sampling. 
        '''
        
        # If no p(v_0) provided, sample from a uniform Bernoulli distribution
        if p_v is None:
            p_v = torch.randint(0, 2, (self.n_models, n, self.n_visible), device=self.device).float()

        # Gibbs sampling
        for i in range(k):
            
            p_h = self.p_h_given_v(p_v.bernoulli() * 2 - 1)
            p_v = self.p_v_given_h(p_h.bernoulli() * 2 - 1)
        
        return p_v
