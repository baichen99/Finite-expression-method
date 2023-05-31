import torch
from scipy.io import loadmat
import numpy as np

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True, allow_unused=True)[0]

# Navier Stokes Equation
def ns_eq(x, y, t, U, dim=3):
    c1 = 1
    c2 = 0.01
    # x, y, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    u, v, p = U[:, 0:1], U[:, 1:2], U[:, 2:3]
    u_x = gradients(u, x)
    u_y = gradients(u, y)
    u_t = gradients(u, t)
    
    v_x = gradients(v, x)
    v_y = gradients(v, y)
    v_t = gradients(v, t)
    
    p_x = gradients(p, x)
    p_y = gradients(p, y)
    
    u_xx = gradients(u_x, x)
    u_yy = gradients(u_y, y)
    v_xx = gradients(v_x, x)
    v_yy = gradients(v_y, y)
    
    x_momentum = u_t + c1 * (u * u_x + v * u_y) - c2 * (u_xx + u_yy) + p_x
    y_momentum = v_t + c1 * (u * v_x + v * v_y) - c2 * (v_xx + v_yy) + p_y
    continuity = u_x + v_y
    return (x_momentum, y_momentum, continuity)

def load_training_data(num, dim=2):
    data = loadmat("dataset/cylinder_nektar_wake.mat")
    U_star = data["U_star"]  # N x 2 x T
    P_star = data["p_star"]  # N x T
    t_star = data["t"]  # T x 1
    X_star = data["X_star"]  # N x 2
    N = X_star.shape[0]
    T = t_star.shape[0]
    # Rearrange Data
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T
    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T
    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1
    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    p = PP.flatten()[:, None]  # NT x 1
    # training domain: X × Y = [1, 8] × [−2, 2] and T = [0, 7]
    data1 = np.concatenate([x, y, t, u, v, p], 1)
    data2 = data1[:, :][data1[:, 2] <= 7]
    data3 = data2[:, :][data2[:, 0] >= 1]
    data4 = data3[:, :][data3[:, 0] <= 8]
    data5 = data4[:, :][data4[:, 1] >= -2]
    data_domain = data5[:, :][data5[:, 1] <= 2]
    # choose number of training points: num =7000
    idx = np.random.choice(data_domain.shape[0], num, replace=False)
    x_train = data_domain[idx, 0:1]
    y_train = data_domain[idx, 1:2]
    t_train = data_domain[idx, 2:3]
    u_train = data_domain[idx, 3:4]
    v_train = data_domain[idx, 4:5]
    p_train = data_domain[idx, 5:6]
    # return [x_train, y_train, t_train, u_train, v_train, p_train]
    # to tensor
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    t_train = torch.tensor(t_train, dtype=torch.float32)
    u_train = torch.tensor(u_train, dtype=torch.float32)
    v_train = torch.tensor(v_train, dtype=torch.float32)
    p_train = torch.tensor(p_train, dtype=torch.float32)
    # hstack
    out = torch.hstack([x_train, y_train, t_train, u_train, v_train, p_train])
    return out    

def ns_sample_pde_x(num, dim=3) -> torch.tensor:
    # domain: X × Y = [1, 8] × [−2, 2] and T = [0, 7]
    x = (torch.rand(num, 1) * 7 + 1).requires_grad_(True)
    y = (torch.rand(num, 1) * 4 - 2).requires_grad_(True)
    t = (torch.rand(num, 1) * 7).requires_grad_(True)

    return (x, y, t)
    