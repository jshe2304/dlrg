import torch
import torch.nn as nn

CELoss = lambda v, v_hat : -torch.mean(v * torch.log(v_hat) + (1 - v) * torch.log(1 - v_hat))
MSELoss = nn.MSELoss()

def free_energy_mse(fine, coarse, batch_n=1, k=4):
    '''
    Mean-squared error of free energies
    '''
    
    # Gibbs-sampled marginal distributions
    p_x = fine.p_v(n=batch_n, k=k).detach()
    p_x_hat = coarse.p_v(p_x, k=k)

    # Sample from marginals
    x = p_x.bernoulli() * 2 - 1
    x_hat = p_x_hat.bernoulli() * 2 - 1

    # Compute free energy difference
    return MSELoss(
        coarse.free_energy(x), 
        coarse.free_energy(x_hat)
    )




def free_energy_difference(fine, coarse, batch_n=1, k=4):
    '''
    Mean-difference of free energies
    '''
    
    # Gibbs-sampled marginal distributions
    p_x = fine.p_v(n=batch_n, k=k).detach()
    p_x_hat = coarse.p_v(p_x, k=k)

    # Sample from marginals
    x = p_x.bernoulli() * 2 - 1
    x_hat = p_x_hat.bernoulli() * 2 - 1

    # Compute free energy difference
    return torch.mean(coarse.free_energy(x) - coarse.free_energy(x_hat))





def marginals_cross_entropy(fine, coarse, batch_n=1, k=4):
    '''
    Cross-entropy of marginal distributions
    '''
    
    # Gibbs-sampled marginals
    p_x = fine.p_v(n=batch_n, k=k)
    p_x_hat = coarse.p_v(p_x, k=k)

    # Cross-entropy of marginal distributions
    return CELoss(p_x, p_x_hat)