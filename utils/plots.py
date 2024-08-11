import torch
import matplotlib.pyplot as plt
from .grad import jacobian

@torch.no_grad()
def plot1d(*functions, xlim=(-2, 2), device=torch.device('cpu')):
    
    x_range = torch.linspace(*xlim, 100, device=device).unsqueeze(1)

    fig, axs = plt.subplots(len(functions), 1, sharex=True, figsize=(8, 4 * len(functions)))

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
def plot2d(f, u=None, xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), color_res=12, vector_res=4, device=torch.device('cpu')):

    xsize = int(max(4, xlim[1] - xlim[0]))
    ysize = int(max(4, ylim[1] - ylim[0]))
    
    fig, (ax) = plt.subplots(1, 1, figsize=(xsize, ysize))

    if u is not None:
        x, y = torch.meshgrid(
            torch.linspace(*xlim, xsize * color_res), 
            torch.linspace(*ylim, ysize * color_res), 
            indexing='xy'
        )
        grid = torch.stack((x, y), dim=2).to(device)
        
        potential = torch.log(u(grid))
        ax.pcolormesh(x.cpu(), y.cpu(), potential.cpu())

    if f is not None:
        x, y = torch.meshgrid(
            torch.linspace(*xlim, xsize * vector_res), 
            torch.linspace(*ylim, ysize * vector_res), 
            indexing='xy'
        )
        grid = torch.stack((x, y), dim=2).to(device)
        
        xvec, yvec = f(grid).split(1, dim=2)
        ax.quiver(x.cpu(), y.cpu(), xvec.squeeze().cpu(), yvec.squeeze().cpu())

    return ax

@torch.no_grad()
def vector_field(f, xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), color_res=12, vector_res=4, device=torch.device('cpu')):

    xsize = int(max(4, xlim[1] - xlim[0]))
    ysize = int(max(4, ylim[1] - ylim[0]))
    
    fig, (ax) = plt.subplots(1, 1, figsize=(xsize, ysize))

    # Divergence Heat Map
    x, y = torch.meshgrid(
        torch.linspace(*xlim, xsize * vector_res), 
        torch.linspace(*ylim, ysize * vector_res), 
        indexing='xy'
    )
    grid = torch.stack((x, y), dim=2).to(device)

    divergence = torch.empty(grid.size(0), grid.size(1), device=device)

    for i in range(grid.size(0)):
        for j in range(grid.size(1)):
            
            J = jacobian(f)(grid[i, j])
            divergence[i, j] = J.diagonal().sum(dim=-1)

    mesh = ax.pcolormesh(x.cpu(), y.cpu(), torch.exp(divergence).cpu(), vmin=0)
    fig.colorbar(mesh, ax=ax)

    # Vector Field
    
    xvec, yvec = f(grid).split(1, dim=2)
    ax.quiver(x.cpu(), y.cpu(), xvec.squeeze().cpu(), yvec.squeeze().cpu())

    return ax

def nd_flows(flow, potential, n_couplers, xlim = (-4, 4), ylim = (-4, 4), res = 64):
    
    fig, axs = plt.subplots(n_couplers-1, n_couplers-1, figsize=(n_couplers * 4, n_couplers * 4))
    
    for i in range(n_couplers-1):
        for j in range(i+1, n_couplers-1):
            axs[i, j].axis('off')
    
    for xindex in range(0, n_couplers-1):
        for yindex in range(xindex + 1, n_couplers):

            # Mesh Grid
            
            x, y = torch.meshgrid(
                torch.linspace(*xlim, res, requires_grad=False), 
                torch.linspace(*ylim, res, requires_grad=False), 
                indexing='xy'
            )
            grid = torch.zeros(res, res, n_couplers).to(flow.device).detach()
            grid[:, :, xindex] = y
            grid[:, :, yindex] = x
    
            # Vector Field
            
            flow_grid = flow(grid).cpu().detach()
    
            # Potential/Divergence

            ax = axs[yindex-1, xindex]
            
            potential_grid = potential(grid).cpu().detach()
            
            #jacobians = vmap(jacobian(flow))(grid.reshape(n * n, n_couplers)).reshape(n, n, n_couplers, n_couplers)
            #divergences = jacobians.diagonal(dim1=2, dim2=3).sum(dim=-1)
            
            mesh = ax.pcolormesh(x, y, potential_grid.cpu().detach())
            #fig.colorbar(mesh, ax=ax, label=r'$\log \: || f \: (J) ||$')
            
            ax.streamplot(
                x.numpy(), y.numpy(), 
                flow_grid[:, :, yindex], flow_grid[:, :, xindex], 
                density=2, 
                linewidth=0.5, 
                color='w'
            )
            
            ax.set_xlabel(rf'$J_{xindex}$')
            ax.set_ylabel(rf'$J_{yindex}$')

    return fig, axs

def boxplots(data, labels):
    
    fig, ax = plt.subplots(figsize=(10, 2 * len(data)))
    
    ax.axvline(0.7643, linestyle='dashed', alpha=0.5, label=r'$J_C$')
    ax.boxplot(
        data, 
        labels=labels, 
        vert=False, 
        flierprops={'marker': '+', 'markersize': 5}
    )
    ax.legend(loc='upper left', bbox_to_anchor=(1., 1., 1., -0.45))
    
    if len(labels) < 2: ax.set_yticks([])
    
    fig.tight_layout()

    return fig, ax
