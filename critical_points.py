import torch
import torch.nn as nn

from math import exp

from rbm.lattices import Fine_RBM, A1_RBM
from rg.monotone import MLP
from utils.losses import free_energy_contrast
from utils.newton import find_root
from utils.hmc import HMC
from utils.grad import batch_grad, grad

device = torch.device('cpu')

# RBMs
fine = Fine_RBM(device=device)
coarse = A1_RBM(device=device)

# Sampler
sampler = HMC(device=device)

# Training Hyperparameters

n_models = 16
epoch = 0
epochs = 2048
batch_size = 512
k = 64
J = torch.rand(1, device=device)

beta = lambda epoch : 4/(1 + exp(-0.01 * (epoch - (epochs / 2))))

# Training 
fname = './experiments/critical_points.txt'
for model in range(n_models):
    
    # Monotone
    C = MLP(
        in_channels=1, 
        device=device
    )
    
    optimizer = torch.optim.Adam(C.parameters())
    
    for epoch in range(epoch, epoch + epochs):
        optimizer.zero_grad()
    
        sampler.potential = lambda J : beta(epoch) * -(grad(C)(J).squeeze() ** 2)
    
        # RG Flow
        J = torch.clamp(sampler.step(J), min=0).detach()
        fine.J = J
        coarse.J = J - grad(C)(J)
    
        # Loss
        loss = free_energy_contrast(
            fine, coarse, 
            batch_size=batch_size, k=k
        )
        loss.backward()
        optimizer.step()

    crit = find_root(torch.tensor([1.], device=device), lambda x: grad(C)(x).squeeze())
    with open(fname, 'a') as f:
        f.write(str(crit) + '\n')
    
