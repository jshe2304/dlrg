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

device = torch.device('cpu')

lattice = sys.argv[1]

# RBMs
if lattice == 'lieb':
    fine = Fine_Lieb_RBM(device=device)
    coarse = A1_Lieb_RBM(device=device)
elif lattice == 'hex':
    fine = Fine_Hex_RBM(device=device)
    coarse = A1_Hex_RBM(device=device)

# Sampler
sampler = HMC(device=device)

# Training Hyperparameters
n_models = 64
epochs = 2048
batch_size = 1024
k_fine = 64
k_coarse = 1
J = torch.rand(1, device=device)

beta = lambda epoch : 4/(1 + exp(-0.01 * (epoch - (epochs / 2))))

# Training 
fname = './experiments/' + lattice + '.csv'
for model in range(n_models):
    
    # Monotone
    C = MLP(
        in_channels=1, 
        device=device
    )
    
    optimizer = torch.optim.Adam(C.parameters())
    
    for epoch in range(epochs):
        optimizer.zero_grad()
    
        sampler.potential = lambda J : -beta(epoch) * grad(C)(J).squeeze().norm()
    
        # RG Flow
        J = torch.clamp(sampler.step(J), min=0).detach()
        fine.J = J
        coarse.J = J - grad(C)(J)
    
        # Loss
        loss = free_energy_contrast(
            fine, coarse, 
            batch_size=batch_size, 
            k_fine=k_fine, 
            k_coarse=k_coarse
        )
        loss.backward()
        optimizer.step()

    crit = find_root(torch.tensor([0.5], device=device), lambda x: grad(C)(x).squeeze())
    with open(fname, 'a') as f:
        f.write(str(crit) + '\n')
