import torch
import math
import torchviz

# problem set up:
# https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/Kovasznay.flow.html

Re = 20
nu = 1 / Re
l = 1 / (2 * nu) - math.sqrt(1 / (4 * nu ** 2) + 4 * math.pi ** 2)

def gradients(u, x, allow_unused=True):
    return torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True, allow_unused=allow_unused
        )[0]

def true_solution(X):
    u = 1 - torch.exp(l * X[:, 0:1]) * torch.cos(2 * torch.pi * X[:, 1:2])
    v = l / (2 * torch.pi) * torch.exp(l * X[:, 0:1]) * torch.sin(2 * torch.pi * X[:, 1:2])
    p = 1 / 2 * (1 - torch.exp(2 * l * X[:, 0:1]))
    return torch.cat([u, v, p], 1)

def sample_pde_x(num, dim, device):
    # 采样PDE的点
    # 采样区域是[0, 1] * [0, 1]
    # 采样num个点
    # 返回一个tensor
    X = []
    for i in range(dim):
        X.append(torch.rand(num, 1).requires_grad_(True).to(device))
    return tuple(X)

def sample_bc_x(num, dim, device):
    x = torch.rand(num, dim).to(device)
    return x

def pde(x, y, U):
    u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]
    u_x = gradients(u, x)
    u_y  = gradients(u, y)
    u_xx = gradients(u_x, x)
    u_yy = gradients(u_y, y)
    
    v_x = gradients(v, x)
    v_y = gradients(v, y)
    v_xx = gradients(v_x, x)
    v_yy = gradients(v_y, y)
    
    p_x = gradients(p, x)
    p_y = gradients(p, y)
    
    x_momentum = u * u_x + v * u_y + p_x - 1 / Re * (u_xx + u_yy)
    y_momentum = u * v_x + v * v_y + p_y - 1 / Re * (v_xx + v_yy)
    continuity = u_x + v_y
    
    return x_momentum, y_momentum, continuity


if __name__ == '__main__':
    # 测试
    x = torch.rand(1000, 1).requires_grad_(True)
    y = torch.rand(1000, 1).requires_grad_(True)
    X = torch.cat([x, y], 1)
    U = true_solution(X)
    x_momentum, y_momentum, continuity = pde(x, y, U)
    # mse
    print(x_momentum.pow(2).mean())
    print(y_momentum.pow(2).mean())
    print(continuity.pow(2).mean())