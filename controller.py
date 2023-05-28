"""A module with NAS controller-related code."""
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from candidate import SaveBuffer, Candidate
import scipy
import torch.nn as nn
from tree import BinaryTree
import function as func
from tree import BinaryTree
from tqdm import tqdm
from learnable_tree import LearnableTree

unary = func.unary_functions
binary = func.binary_functions


class Controller(torch.nn.Module):
    def __init__(self, tree: BinaryTree, dim=1):
        self.tree = tree
        torch.nn.Module.__init__(self)

        self.softmax_temperature = 5.0
        self.tanh_c = 2.5
        self.mode = True

        self.input_size = 20
        self.hidden_size = 50
        self.output_size = self.tree.unary_num * len(unary) + tree.binary_num * len(binary)

        self._fc_controller = nn.Sequential(
            nn.Linear(self.input_size,self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size,self.output_size))

    def forward(self,x):
        logits = self._fc_controller(x)

        logits /= self.softmax_temperature

        # exploration # ??
        if self.mode == 'train':
            logits = (self.tanh_c*F.tanh(logits))

        return logits

    def sample(self, batch_size=1, step=0, random_step=0):
        inputs = torch.zeros(batch_size, self.input_size)
        log_probs = []
        operators = []
        operator_choices = []
        self.tree.inorder(lambda node: operator_choices.append(len(unary) if node.is_unary else len(binary)))
        # 输出是每个结点的operator概率分布
        # 输出的维度根据每个结点的operator数量决定
        total_logits = self.forward(inputs)
        cumsum = np.cumsum([0] + operator_choices)
        for i in range(self.tree.node_num):
            # 当前结点的logits
            logits = total_logits[:, cumsum[i]:cumsum[i+1]]
            # softmax得到当前结点所有操作分布
            # shape == (bath_size, operator_num)
            # dim =-1 表示最后一个维度，也就是将每一行的元素进行softmax
            probs = F.softmax(logits, dim=-1)
            # log_softmax，是用来在TrainController实现Policy Gradient
            log_prob = F.log_softmax(logits, dim=-1)
            if step >= random_step:
                # 使用多项式采样，按照概率分布采样一个，返回的是索引
                operator_idxs = probs.multinomial(num_samples=1).data
            else:
                # 表示仍处于探索阶段，还不需要进行采样
                # torch.randint 生成的随机整数将落在 0 到 structure_choice[idx] - 1 的范围内
                # 这样做可以确保每个计算节点都有一个随机的操作选择，以便进行全面的探索
                operator_idxs = torch.randint(0, len(operator_choices[i]), size=(batch_size, 1))
            # 将operator_idxs转换成一个variable
            variable = torch.autograd.Variable(operator_idxs, requires_grad=False)
            # variable是操作符索引，根据其从log_prob中取出对应的log_prob
            # 是从分布中采样一个操作符，返回shape为(batch_size, 1)
            selected_log_prob = log_prob.gather(1, variable)
            # [:, 0:1]获取每个节点的单个概率值和动作选择，以便后续使用
            log_probs.append(selected_log_prob[:, 0:1])
            operators.append(operator_idxs[:, 0:1])
            
        # len(log_probs) == node_num
        log_probs = torch.cat(log_probs, dim=1)
        operators = torch.cat(operators, dim=1)
        # operators.shape == log_probs.shape == (batch_size, node_num)
        return operators, log_probs
