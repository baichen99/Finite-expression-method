import torch
from possion.pde import cal_l2_relative_error

# custom pde
from possion.pde import possion_eq as cal_pde_loss
from possion.pde import possion_true_solution as true_solution
from possion.pde import possion_sample_bc_x as sample_bc_x
from possion.pde import possion_sample_pde_x as sample_pde_x


dim = 2
batch_size = 20
random_step = 0
epoch = 10  # sample epoch
boundary_num = 1000
pde_num = 5000
candidate_size = 10

percentile = 0.5    # equation 18 in paper
controller_lr = 2e-3
controller_grad_clip = 0
finetune_epochs = 10000
finetune_lr = 1e-2
base = 20000
bc_weight = 100

test_num = 20000
left = -1
right = 1

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
    pde_x = (torch.rand(1000, dim) * (right - left) + left).requires_grad_(True)
    obs_x = torch.rand(1000, dim) * (right - left) + left
    
    bc_true = true_solution(obs_x)
    bc_pred = model(obs_x)
    bc_loss = torch.mean((bc_true - bc_pred)**2)
    
    pde_pred = model(pde_x)
    pde_residual = cal_pde_loss(pde_pred, pde_x, dim)
    pde_loss = torch.mean(pde_residual**2)
    loss = 100 * bc_loss + pde_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        # cal l2 error
        x_test = torch.rand(test_num, dim) * (right - left) + left
        y_test = model(x_test)
        true_test = true_solution(x_test)
        l2_error = cal_l2_relative_error(y_test, true_test)
        print('epoch: {}, loss: {}, bc_loss: {}, pde_loss: {}, l2_error: {}'.format(epoch, loss, bc_loss, pde_loss, l2_error))
        
    return loss


#train
num_epoch = 10000
model = MLP(seq_net=[dim] + [50] * 4 + [1])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(num_epoch):
    pinn_train(model, optimizer, dim, epoch)