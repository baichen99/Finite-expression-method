import torch

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

def possion_eq(u, x, dim):
    # u: (bs, 1)
    # x: (bs, dim)

    # 计算 u 关于 x 的一阶导数
    du_dx = gradients(u, x)
    # 对每个维度进行二阶导数计算，并求和
    laplace = torch.zeros_like(u)
    for i in range(dim):
        d2u_dxdx = gradients(du_dx[:, i], x)[:, i].unsqueeze(1)
        laplace += d2u_dxdx
    return -laplace + dim * torch.ones_like(u)


# def RHS_pde(x):
#     bs = x.size(0)
#     dim = x.size(1)
#     return 

def possion_true_solution(x):
    # u = 1/2 * (x1^2 + x2^2)
    return 0.5*torch.sum(x**2, dim=1, keepdim=True)#1 / (2 * x[:, 0:1] + x[:, 1:2]-5)

def possion_sample_pde_x(num, dim=1):
    x = torch.rand(num, dim).requires_grad_(True)
    return x

def possion_sample_bc_x(num, dim=1):
    x = torch.rand(num, dim)
    return x

def cal_l2_relative_error(pred, true):
    return torch.sqrt(torch.sum((pred - true)**2)) / torch.sqrt(torch.sum(true**2))

if __name__ == '__main__':
    x = torch.rand(10, 2).requires_grad_(True)
    u = possion_true_solution(x)
    # cal possion equation
    pde = possion_eq(u, x, 2)
    print(pde)
    
    