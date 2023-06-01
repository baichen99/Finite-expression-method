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
        self.a = None
        self.b = None
        if not is_leave:
            self.a = nn.Parameter(torch.Tensor(1))
            self.a.data.fill_(1)
            self.b = nn.Parameter(torch.Tensor(1))
            self.b.data.fill_(0)
        self.is_leave = is_leave
    
    def get_op(self, idx):
        return func.unary_str[idx]
    
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
    
    def get_op(self, idx):
        return func.binary_str[idx]

class LearnableTree(nn.Module):
    def __init__(self, tree: BinaryTree, dim=1, output_dim=1):
        super(LearnableTree, self).__init__()
        self.tree = tree
        self.dim = dim
        # learnable_operator_set给每个结点存放所有可能的操作
        self.learnable_operator_set = nn.ModuleList()
        self.linear = nn.ModuleList()
        
        for i in range(self.tree.leaves_num):
            linear_module = torch.nn.Linear(dim, output_dim, bias=True) #set only one variable
            linear_module.weight.data.normal_(0, 1/math.sqrt(dim))
            linear_module.bias.data.fill_(0)
            self.linear.append(linear_module)
            
        def add_op_callback_fn(node: BinaryTree):
            if node.is_unary:
                # 如果是一元操作，那么就把所有的一元操作都加进去
                self.learnable_operator_set.append(nn.ModuleList(
                    [UnaryOperation(op, node.is_leaf) for op in unary])
                )
            else:
                # 如果是二元操作，那么就把所有的二元操作都加进去
                self.learnable_operator_set.append(nn.ModuleList(
                    [BinaryOperation(op) for op in binary])
                )
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
        operator_idxs_copy = operator_idxs.tolist().copy()  # create a copy of operator_idxs
        learnable_operator_set_copy = nn.ModuleList([op for op in self.learnable_operator_set])  # create a copy of learnable_operator_set
        new_tree.set_nn_operator(operator_idxs_copy, learnable_operator_set_copy)

        for i, node in enumerate(new_tree.get_leaves()):
            node.linear_transform = self.linear[i]
        out = new_tree.compute_by_tree(x)
        return out

    def get_node_info(node: BinaryTree):
            info = ''
            info += f'op_str: {node.op_str} '
            if node.is_leaf:
                w = node.linear_transform.weight.flatten().detach().numpy()
                b = node.linear_transform.bias.detach().numpy()
                info += f'w: {w}, b: {b}'
            elif node.is_unary:
                alpha = node.node_operator.a.detach().numpy()
                beta = node.node_operator.b.detach().numpy()
                info += f'alpha: {alpha}, beta: {beta}'
            print(info)
        
    def get_formula(self, operator_idxs):
        operator_idxs = operator_idxs.reshape(-1)
        new_tree = self.tree.create_same_tree()
        operator_idxs_copy = operator_idxs.tolist().copy()  # create a copy of operator_idxs
        learnable_operator_set_copy = nn.ModuleList([op for op in self.learnable_operator_set]) 
        new_tree.set_nn_operator(operator_idxs_copy, learnable_operator_set_copy)
        for i, node in enumerate(new_tree.get_leaves()):
            node.linear_transform = self.linear[i]
            
        new_tree.inorder(LearnableTree.get_node_info)

        def build_formula(node: BinaryTree):
            if node.is_leaf:
                linear_transform = node.linear_transform
                b = str(linear_transform.bias.item())
                
                if node.op_str == '0':
                    return '0'
                elif node.op_str == '1':
                    p = ' + '.join([f'{weight.item()}' for i, weight in enumerate(linear_transform.weight.flatten())])
                    return f'{p} + {b}'
                elif node.op_str in ['', '^2', '^3', '^4']: # ^1省略为''
                    # 后缀
                    # a * x + b
                    # (a_1 * x_1 + a_2 * x_2 + ... + a_n * x_n) + b
                    p = ' + '.join([f'{weight.item()} * (x_{i}){node.op_str}' for i, weight in enumerate(linear_transform.weight.flatten())])
                    return f'({p}) + {b}'
                else:
                    # 前缀：比如sin, cos
                    # w1*sin(x1) + w2*sin(x2) + .. + b
                    p = ' + '.join([f'{weight.item()} * {node.op_str}(x_{i})' for i, weight in enumerate(linear_transform.weight.flatten())])
                    return f'({p}) + {b}'
            else:
                if node.is_unary:
                    # 一元操作 sin(??) (??)^2 ...
                    if node.op_str in ['0', '1']:
                        return node.op_str
                    elif node.op_str in ['', '^2', '^3', '^4']: #  # ^1省略为''
                        return '(' + build_formula(node.leftChild) + ')' + node.op_str
                    else:
                        return node.op_str + '(' + build_formula(node.leftChild) + ')'
                else:
                    return '(' + build_formula(node.leftChild) + node.op_str + build_formula(node.rightChild) + ')'
        return build_formula(new_tree)