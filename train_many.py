import torch
import torch.nn as nn

import sys
from math import exp

from neuralnet.mlp import MLP

from rbm.rbm import RBM
from rbm.adjacency import get_adjacency_matrices
from rbm.coupling import get_coupling_matrix
from rbm.configurations import get_configurations
from rbm.loss import contrastive_divergence

from utils.newton import find_root
from utils.samplers import HMC

'''
Script for testing hyperparameters. 
'''

device = torch.device('cuda')

lattice = 'lieb'
representation = 'a1'

#################
# Constant Models
#################

# RBMs
lieb_square, lieb_cross = get_adjacency_matrices(lattice)
coupling_matrix = get_coupling_matrix(representation)
n_couplers = coupling_matrix.size(0)

fine = RBM(lieb_square, coupling_matrix, device=device)
coarse = RBM(lieb_cross, coupling_matrix, device=device)

# Sampler
sampler = HMC(device=device)

##########################
# Constant Hyperparameters
##########################

epochs = 4096
anneal_at = 1024
record_at = 2048

#########################
# Varying Hyperparameters
#########################

hyperparameters = {
    'nn_depth' : n_couplers, 
    'nn_width' : 16 * n_couplers, 
    
    'n_models' : 8 * (3 ** n_couplers), 
    
    'n_cd_samples' : 512, 
    'k_fine' : 32, 
    'k_coarse' : 1, 
}

hyperparameter_values = {
    'nn_depth' : [n_couplers], 
    'nn_width' : [16 * n_couplers], 
    
    'n_models' : torch.arange(1, 64), 
    
    'n_cd_samples' : torch.arange(1, 512, 4), 
    'k_fine' : torch.arange(1, 32), 
    'k_coarse' : torch.arange(1, 32), 
}

beta = lambda epoch : 16/(1 + exp( -16 * (epoch - anneal_at) / epochs ))

######
# Runs
######

param = sys.argv[1]

runs = []
for val in hyperparameter_values[param]:

    hyperparameters[param] = int(val)
    
    # Flow Model
    flow = MLP(
        in_dim=n_couplers, 
        out_dim=n_couplers, 
        hidden_depth=hyperparameters['nn_depth'], 
        width=hyperparameters['nn_width'], 
        device=device
    )
    flow.train()
    
    optimizer = torch.optim.Adam(flow.parameters())

    roots = []

    J = torch.randn(hyperparameters['n_models'], n_couplers, device=device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()

        sampler.potential = lambda J : beta(epoch) * (flow(J) ** 2).sum(dim=-1)

        # RG Flow
        J = sampler.step(J).detach()
        fine.J = J
        coarse.J = J + flow(J)
    
        # Loss
        contrastive_divergence(
            fine, coarse, 
            batch_size=hyperparameters['n_cd_samples'], 
            k_fine=hyperparameters['k_fine'], 
            k_coarse=hyperparameters['k_coarse'], 
        ).backward()
        optimizer.step()

        if epoch >= record_at:
            x = torch.tensor([0.8], device=device)
            #x = torch.tensor([0.8, 0., 0., 0.8, 0.8], device=device)
            roots.append(find_root(flow, x))

    roots = torch.stack(roots).reshape(epochs - record_at)
    runs.append(roots)

# Save data

path = f'data/hyperparameter_tuning_1/{param}/'

torch.save(torch.stack(runs), path + 'roots.pt')
torch.save(hyperparameter_values[param], path + 'values.pt')