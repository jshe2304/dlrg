import torch
import torch.nn as nn

import sys
from math import exp

from rbm.lieb_lattices import Fine_Lieb_RBM, A1_Lieb_RBM
from rbm.hex_lattices import Fine_Hex_RBM, A1_Hex_RBM
from rg.monotone import MLP
from utils.losses import free_energy_contrast
from utils.newton import find_root
from utils.hmc import HMC
from utils.grad import batch_grad, grad

device = torch.device('cuda')

# RBMs
lattice = sys.argv[1]
if lattice == 'lieb':
    fine = Fine_Lieb_RBM(device=device)
    coarse = A1_Lieb_RBM(device=device)
elif lattice == 'hex':
    fine = Fine_Hex_RBM(device=device)
    coarse = A1_Hex_RBM(device=device)

# Training Hyperparameters
n_models = 64
epochs = 1024
sample_batch_size = 32
cd_batch_size = 1024
k_fine = 64
k_coarse = 1
beta = lambda epoch : 64/(1 + exp( -16 * (epoch - (epochs / 2)) / epochs ))

# Sampler
sampler = HMC(device=device)

# Training
fname = './experiments/' + lattice + '.csv'
for model in range(n_models):
    
    # Flow
    flow = MLP(
        in_dims=1, 
        out_dims=1, 
        device=device
    )
    flow.train()

    optimizer = torch.optim.Adam(flow.parameters())

    J = torch.randn(sample_batch_size, 1, device=device)
    for epoch in range(epochs):
        optimizer.zero_grad()
    
        sampler.potential = lambda J : beta(epoch) * flow(J) ** 2
    
        # RG Flow
        J = sampler.step(J).detach()
    
        loss = 0
        for i in range(sample_batch_size):
            
            fine.J = J[i]
            coarse.J = J[i] + flow(J[i])
        
            # Loss
            loss += free_energy_contrast(
                fine, coarse, 
                batch_size=cd_batch_size, 
                k_fine=k_fine, 
                k_coarse=k_coarse
            )
        
        loss.backward()
        optimizer.step()

    crit = find_root(torch.tensor([0.5], device=device), flow)
    with open(fname, 'a') as f:
        f.write(str(crit) + '\n')
