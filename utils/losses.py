import torch
import torch.nn as nn

import time

def free_energy_contrast(fine, coarse, batch_size=1, k_fine=4, k_coarse=1):
    '''
    Mean-contrast of free energies. 
    Lowers energy of fine-grained samples. 
    Raises energy of coarse-grained hallucinations. 
    '''
    
    # Gibbs-sampled marginal distributions
    start = time.time()
    
    p_x = fine.p_v(n=batch_size, k=k_fine).detach()
    p_x_hat = coarse.p_v(p_x, n=batch_size, k=k_coarse)

    # Sample from marginals
    x = p_x.bernoulli() * 2 - 1
    x_hat = p_x_hat.bernoulli() * 2 - 1

    # Compute free energy difference
    return torch.mean(coarse.free_energy(x) - coarse.free_energy(x_hat))
