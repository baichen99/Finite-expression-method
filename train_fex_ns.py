from tqdm import tqdm
from learnable_tree import LearnableTree
from controller import Controller
from candidate import SaveBuffer, Candidate
import torch
from torch.utils.tensorboard import SummaryWriter
from tree import BinaryTree
from possion.pde import cal_l2_relative_error

# custom pde
from ns_pde import ns_eq as cal_pde_loss
from ns_pde import load_training_data as sample_bc_points
from ns_pde import ns_sample_pde_x as sample_pde_x


class args:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    dim = 3
    batch_size = 5
    random_step = 3
    epoch = 10  # sample epoch
    boundary_num = 1000
    pde_num = 5000
    candidate_size = 10
    
    percentile = 0.5    # equation 18 in paper
    controller_lr = 2e-3
    controller_grad_clip = 0
    finetune_epochs = 10000
    finetune_lr = 1e-3
    base = 20000
    bc_weight = 100
    
    test_num = 20000
    
args.tb_path = f'logs/dim_{args.dim}'
tb_writer = SummaryWriter(args.tb_path)

def cal_loss(boundary_num: int, pde_num: int, operator_idxs, learnable_tree: LearnableTree):
    pde_x, pde_y, pde_t = sample_pde_x(pde_num, args.dim)
    # to device
    pde_x, pde_y, pde_t = pde_x.to(args.device), pde_y.to(args.device), pde_t.to(args.device)
    # change this
    bc_points = sample_bc_points(boundary_num, args.dim)
    bc_points = bc_points.to(args.device)
    # u, v
    obs_x, bc_true = bc_points[:, :3], bc_points[:, 3:5]
    obs_x = obs_x.to(args.device)

    bc_pred = learnable_tree(obs_x, operator_idxs)
    bc_loss = torch.mean((bc_true - bc_pred)**2)
    
    pde_pred = learnable_tree(torch.cat([pde_x, pde_y, pde_t], dim=1), operator_idxs)
    x_momentum, y_momentum, continuity = cal_pde_loss(
        pde_x, pde_y, pde_t,
        pde_pred,
        dim=3
    )
    pde_residual = x_momentum + y_momentum + continuity
    pde_loss = torch.mean(pde_residual**2)
    
    loss = args.bc_weight * bc_loss + pde_loss
    return loss

# 计算当前树的loss/得分
def cal_tree_loss(batch_size: int, operator_idxs_list, learnable_tree: LearnableTree):
    """
        一共有batch_size个样本 每个样本的operator_idxs不同
        计算每个样本的loss 返回batch中每个样本的最小损失
    """
    losses = []
    formulas = []
    
    for batch_idx in tqdm(range(batch_size), desc='sample for batch'):
        operator_idxs = operator_idxs_list[batch_idx]
        # 初始化训练参数
        learnable_tree.normal_params()
        # 先用adam迭代20次, 再用LBFGS迭代20次
        tree_optim = torch.optim.Adam(learnable_tree.get_parameters(), lr=1e-3)
        for _ in range(20):
            loss = cal_loss(args.boundary_num, args.pde_num, operator_idxs, learnable_tree)
            tree_optim.zero_grad()
            loss.backward()
            tree_optim.step()
        
        tree_optim = torch.optim.LBFGS(learnable_tree.get_parameters(), lr=1, max_iter=20)
        
        # 创建tqdm, desc=T2
        def closure():
            tree_optim.zero_grad()
            loss = cal_loss(args.boundary_num, args.pde_num, operator_idxs, learnable_tree)
            loss.backward()
            return loss
        
        tree_optim.step(closure)
        # 计算当前树的loss
        loss = cal_loss(args.boundary_num, args.pde_num, operator_idxs, learnable_tree)
        losses.append(loss.item())
    # losses是一个长度为batch_size的list，每个元素是一个float
    return losses, formulas

# eval
def eval(test_num:int, learnable_tree: LearnableTree, operator_idxs: list):
    # 生成20000个点
    learnable_tree.eval()
    with torch.no_grad():
        points = sample_bc_points(test_num, args.dim)
        x, true = points[:, :3], points[:, 3:5]
        pred = learnable_tree(x, operator_idxs)[:, :2]
        u_err = cal_l2_relative_error(true[:, 0], pred[:, 0])
        v_err = cal_l2_relative_error(true[:, 1], pred[:, 1])
        return [u_err, v_err]


def TrainController(controller: Controller, learnable_tree: LearnableTree, optim: torch.optim.Adam=None):
    candidates = SaveBuffer(args.candidate_size)
    for epoch in tqdm(range(args.epoch), desc="sample epoch"):
        # 采样batch_size个树/表达式
        operators_idxs_list, log_probs = controller.sample(args.batch_size, step=epoch, random_step=args.random_step)
        losses, formulas = cal_tree_loss(args.batch_size, operators_idxs_list, learnable_tree)
        # (batch_size, 1)
        losses = torch.FloatTensor(losses).view(-1,1)
        
        base = args.base
        losses[losses > base] = base
        # losses[losses != losses] = 1e10
        
        scores = 1 / (1 + torch.sqrt(losses))
        # 选batch中loss最小的作为最优表达式
        batch_min_idx = torch.argmin(losses, dim=0)
        batch_min_loss = losses[batch_min_idx]
        batch_min_operator = operators_idxs_list[batch_min_idx]
        batch_best_score = scores[batch_min_idx]
        
        op_seq = learnable_tree.tree.get_operator_sequence(batch_min_operator.squeeze().tolist())
        op_seq_str = ",".join(op_seq)
        
        # best formula
        # best_formula = formulas[batch_min_idx]
        best_formula = ''
        
        # 加入到候选池
        candidates.add_new(
            Candidate(action=batch_min_operator, expression=best_formula, error=batch_min_loss)
        )
        # policy gradient
        # 让最好的表达式的下次采样概率增大，因为采样是controller的任务，更新controller的参数
        
        #(batch_size, 1) => (batch_size,)
        argsort = torch.argsort(scores.squeeze(), descending=True)
        score_sorted = scores[argsort]
        logs_prob_sorted = log_probs[argsort]
        # 排序后，将第k个score作为阈值
        kth = int(args.percentile * args.batch_size)
        kth_score = score_sorted[kth]
        adv = score_sorted[:kth] - kth_score
        # print(f'adv.shape: {adv.shape}, logs_prob_sorted.shape: {logs_prob_sorted.shape}')
        PG_loss = -torch.mean(adv * logs_prob_sorted[:kth])
        # print(f'pg_loss: {PG_loss.item()}, mean_score: {torch.mean(score_sorted).item()}')
        tb_writer.add_scalar('pg_loss', PG_loss.item(), epoch)
        tb_writer.add_scalar('mean_score', torch.mean(score_sorted).item(), epoch)
        tb_writer.add_scalar('best_score', batch_best_score.item(), epoch)
        
        optim.zero_grad()
        PG_loss.backward()
        optim.step()
        # log
        
    # print all candidates's score
    print('all candidates:')
    for candidate in candidates.candidates:
        print(f'operator_idxs: {candidate.action}')
    
    # finetune
    for candidate in tqdm(candidates.candidates, desc="Finetune"):
        tree_optim = torch.optim.Adam(learnable_tree.get_parameters(), lr=args.finetune_lr)
        finetune(learnable_tree, candidate.action, tree_optim)
        
def finetune(learnable_tree: LearnableTree, operator_idxs: list, tree_optim: torch.optim.Adam=None):
    # normal params
    learnable_tree.normal_params()
    for epoch in tqdm(range(args.finetune_epochs), desc="Finetune"):
        loss = cal_loss(args.boundary_num, args.pde_num, operator_idxs, learnable_tree)
        # print(f'finetune_loss: {loss.item()}')
        tree_optim.zero_grad()
        loss.backward()
        tree_optim.step()

        if epoch % 1000 == 0:
            # cal l2 relative error
            [u_err, v_err] = eval(args.test_num, learnable_tree, operator_idxs)
            print(f"l2 relative err: {u_err} , {v_err}")
    return loss.item()


if __name__ == '__main__':
    # build a computation tree
    #     X
    #    / \
    #   X   X
    #  / \  |
    # X  X  X  
    tree = BinaryTree(node_operator=None, is_unary=False)
    tree.leftChild = BinaryTree(node_operator=None, is_unary=False)
    tree.rightChild = BinaryTree(node_operator=None, is_unary=True)
    tree.leftChild.leftChild = BinaryTree(node_operator=None, is_unary=True)
    tree.leftChild.rightChild = BinaryTree(node_operator=None, is_unary=True)
    tree.rightChild.leftChild = BinaryTree(node_operator=None, is_unary=True)

    controller = Controller(tree, dim=args.dim)
    controller_optim = torch.optim.Adam(controller.parameters(), lr=args.controller_lr)
    learnable_tree = LearnableTree(tree, dim=args.dim).to(args.device)
    # operators, log_probs = controller.sample(args.batch_size)
    train_controller = TrainController(controller, learnable_tree, controller_optim)
    