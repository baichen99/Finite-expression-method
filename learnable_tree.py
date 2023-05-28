from tree import BinaryTree
import torch
import torch.nn as nn
import math
import function as func


unary = func.unary_functions
binary = func.binary_functions


# 把func.py中的函数转换成nn.Module
class UnaryOperation(nn.Module):
    def __init__(self, operator, is_leave):
        super(UnaryOperation, self).__init__()
        self.unary = operator
        if not is_leave:
            self.a = nn.Parameter(torch.Tensor(1))
            self.a.data.fill_(1)
            self.b = nn.Parameter(torch.Tensor(1))
            self.b.data.fill_(0)
        self.is_leave = is_leave

    def forward(self, x):
        if self.is_leave:
            return self.unary(x)
        else:
            return self.a * self.unary(x) + self.b

class BinaryOperation(nn.Module):
    def __init__(self, operator):
        super(BinaryOperation, self).__init__()
        self.binary = operator
    def forward(self, x, y):
        return self.binary(x, y)

class LearnableTree(nn.Module):
    def __init__(self, tree: BinaryTree, dim=1):
        super(LearnableTree, self).__init__()
        self.tree = tree
        self.dim = dim
        # learnable_operator_set给每个结点存放所有可能的操作
        self.learnable_operator_set = []
        self.linear = []
        for i in range(self.tree.leaves_num):
            linear_module = torch.nn.Linear(dim, 1, bias=True) #set only one variable
            linear_module.weight.data.normal_(0, 1/math.sqrt(dim))
            linear_module.bias.data.fill_(0)
            self.linear.append(linear_module)
            
        def add_op_callback_fn(node: BinaryTree):
            if node.is_unary:
                # 如果是一元操作，那么就把所有的一元操作都加进去
                self.learnable_operator_set.append([UnaryOperation(op, node.is_leaf) for op in unary])
            else:
                # 如果是二元操作，那么就把所有的二元操作都加进去
                self.learnable_operator_set.append([BinaryOperation(op) for op in binary])
        self.tree.inorder(add_op_callback_fn)
        
    def get_parameters(self):
        params = []
        for node_op_list in self.learnable_operator_set:
            for v in node_op_list:
                if isinstance(v, UnaryOperation):
                    if not v.is_leave:
                        # 一元操作且不是叶子结点，那么就把a和b加进去
                        params += [v.a, v.b]
                else:
                    pass
        
        for module in self.linear:
            for param in module.parameters():
                # print(f"param: {param}")
                params.append(param)
        return params
    
    def normal_params(self):
        for param in self.get_parameters():
            param.data.normal_(0.0, 0.1)

    def forward(self, x, operator_idxs):
        # shape of operator_idxs: (node_num, )
        operator_idxs = operator_idxs.reshape(-1)
        new_tree = self.tree.create_same_tree()
        new_tree.set_nn_operator(operator_idxs, self.learnable_operator_set.copy())
        for i, node in enumerate(new_tree.get_leaves()):
            node.linear_transform = self.linear[i]
        out = new_tree.compute_by_tree(x)
        return out
