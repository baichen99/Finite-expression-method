import torch
from pde import cal_l2_relative_error

# custom pde
from kflow_pde import pde as cal_pde_loss
from kflow_pde import true_solution


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dim = 2
batch_size = 20
random_step = 0
epoch = 10  # sample epoch
boundary_num = 1000
pde_num = 5000
candidate_size = 10

left = 0
right = 1

percentile = 0.5    # equation 18 in paper
controller_lr = 2e-3
controller_grad_clip = 0
finetune_epochs = 10000
finetune_lr = 1e-2
base = 20000
bc_weight = 100

test_num = 20000


# mlp
class MLP(torch.nn.Module):
    def __init__(self, seq_net):
        # seq_net: [3, 20, 1]
        super(MLP, self).__init__()
        self.seq_net = seq_net
        self.num_layers = len(seq_net)
        self.layers = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.layers.append(torch.nn.Linear(seq_net[i], seq_net[i + 1]))
            self.layers.append(torch.nn.Tanh())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
# pinn train
def pinn_train(model, optimizer, dim, epoch):
    model.train()
    x = (torch.rand(1000, 1) * (right - left) + left).requires_grad_(True).to(device)
    y = (torch.rand(1000, 1) * (right - left) + left).requires_grad_(True).to(device)
    
    obs_x = (torch.rand(1000, 2) * (right - left) + left).to(device)
    
    bc_true = true_solution(obs_x)
    bc_pred = model(obs_x)
    bc_loss = torch.mean((bc_true - bc_pred)**2)
    
    pde_pred = model(torch.cat([x, y], 1))
    x_momentum, y_momentum, continuity =  cal_pde_loss(x, y, pde_pred)
    pde_loss = x_momentum.pow(2).mean() + y_momentum.pow(2).mean() + continuity.pow(2).mean()
    loss = 100 * bc_loss + pde_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        # cal l2 error
        x_test = (torch.rand(test_num, dim) * (right - left) + left)
        x_test = x_test.to(device)
        y_test = model(x_test)
        true_test = true_solution(x_test)
        l2_error = cal_l2_relative_error(y_test, true_test)
        print('epoch: {}, loss: {}, bc_loss: {}, pde_loss: {}, l2_error: {}'.format(epoch, loss, bc_loss, pde_loss, l2_error))
        
    return loss


#train
num_epoch = 10000
model = MLP(seq_net=[dim] + [50] * 4 + [3]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(num_epoch):
    pinn_train(model, optimizer, dim, epoch)
# LBFGS
optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-1, max_iter=500, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
def closure():
    optimizer.zero_grad()
    loss = pinn_train(model, optimizer, dim, epoch)
    return loss
optimizer.step(closure)

# l2 err == 0.23