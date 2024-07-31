import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def plot1d(*functions, xlim=(0, 2), device=torch.device('cpu')):
    
    x_range = torch.linspace(*xlim, 100, device=device).unsqueeze(1)

    fig, axs = plt.subplots(len(functions), 1, sharex=True, figsize=(8, 2 * len(functions)))

    if len(functions) == 1:
        f_range = functions[0](x_range).cpu().detach()
        axs.plot(x_range.cpu().detach(), f_range)
        axs.axhline(0, color='black', linewidth=0.5)
        axs.axvline(0, color='black', linewidth=0.5)
        
        return fig, axs

    for i, f in enumerate(functions):
        
        f_range = f(x_range).cpu().detach()
        
        axs[i].plot(x_range.cpu().detach(), f_range)
        axs[i].axhline(0, color='black', linewidth=0.5)
        axs[i].axvline(0, color='black', linewidth=0.5)

    return fig, axs

@torch.no_grad()
def plot2d(f, u=None, xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), device=torch.device('cpu')):

    fig, (ax) = plt.subplots(1, 1, 
        figsize=(max(4, xlim[1]-xlim[0]), max(4, ylim[1]-ylim[0]))
    )

    x, y = torch.meshgrid(
        torch.linspace(*xlim, 32), 
        torch.linspace(*ylim, 32), 
        indexing='ij'
    )
    grid = torch.stack((x, y), dim=2).to(device)
        
    if u is not None:
        potential = torch.log(u(grid))
        ax.pcolormesh(x.cpu(), y.cpu(), potential.cpu())

    if f is not None:
        xvec, yvec = f(grid).split(1, dim=2)
        ax.quiver(x.cpu(), y.cpu(), xvec.squeeze().cpu(), yvec.squeeze().cpu())

    return ax