import torch
import torch.nn as nn

from .configurations import get_configurations

def contrastive_divergence(fine, coarse, batch_size=1, k_fine=4, k_coarse=1):
    '''
    Mean-contrast of free energies. 
    Lowers energy of fine-grained samples. 
    Raises energy of coarse-grained hallucinations. 
    '''
    
    # Sample from target distribution
    p_x = fine.p_v(n=batch_size, k=k_fine).detach()
    x = p_x.bernoulli() * 2 - 1

    # Sample from model distribution
    p_x_hat = coarse.p_v(p_x, n=batch_size, k=k_coarse)
    x_hat = p_x_hat.bernoulli() * 2 - 1

    # Compute free energy difference
    return torch.mean(coarse.free_energy(x) - coarse.free_energy(x_hat))

def kl_divergence(fine, coarse):

    visible_configurations = get_configurations(fine.n_visible, device=fine.device)

    # Free Energies
    F_f = fine.free_energy(visible_configurations)
    F_c = coarse.free_energy(visible_configurations)

    # Partition Functions
    Z_f = fine.Z.reshape(fine.n_models, 1)
    Z_c = coarse.Z.reshape(coarse.n_models, 1)

    # KL divergence
    sum_arg = torch.exp(-F_f) * (-F_f - torch.log(Z_f) + F_c + torch.log(Z_c))
    kl_divergences = sum_arg.sum(dim=1, keepdim=True) / Z_f

    return torch.mean(kl_divergences)
