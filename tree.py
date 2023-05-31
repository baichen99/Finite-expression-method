import numpy as np
import function as func
from typing import Callable
import torch

unary = func.unary_functions
binary = func.binary_functions

class BinaryTree:
    def __init__(self, node_operator: Callable, is_unary: bool=True):
        self.node_operator = node_operator
        self.is_unary=is_unary
        self.leftChild = None
        self.rightChild = None
        
    def insert_leftChild(self, node: 'BinaryTree'):
        self.leftChild = node
        
    def insert_rightChild(self, node: 'BinaryTree'):
        self.rightChild = node
        
    def inorder(self, callback_fn, *args, **kwargs):
        if self.leftChild:
            self.leftChild.inorder(callback_fn, *args, **kwargs)
        callback_fn(self, *args, **kwargs)
        if self.rightChild:
            self.rightChild.inorder(callback_fn, *args, **kwargs)
            
    
    @property
    def nodes(self):
        nodes = []
        self.inorder(lambda node: nodes.append(node))
        return nodes
    
    def set_operator(self, operator_idxs):
        def callback_fn(node, operator_idxs):
            if node.is_unary:
                node.node_operator = unary[operator_idxs[0]]
            else:
                node.node_operator = binary[operator_idxs[0]]
            operator_idxs.pop(0)

        self.inorder(callback_fn, operator_idxs)
    
    def set_nn_operator(self, operator_idxs:list, learnable_operator_set:list):
        # set_operator设置每个operator的的类型为function
        # set_nn_operator设置每个operator的类型为nn.Module
        # operator_idxs是1d list，每个元素是一个int，表示operator的索引
        # learnable_operator_set是2d list，每个元素是一个nn.Module的list，表示operator的备选
        assert len(learnable_operator_set) == self.node_num
        assert len(operator_idxs) == self.node_num
        operator_idxs = operator_idxs
        def callback_fn(node, operator_idxs, learnable_operator_set):
            ops = learnable_operator_set[0]
            nn_op = ops[operator_idxs[0]]
            node.node_operator = nn_op
            operator_idxs.pop(0)
            learnable_operator_set.pop(0)
        self.inorder(callback_fn, operator_idxs, learnable_operator_set)
    
    def compute_by_tree(self, x):
        # 使用中序遍历计算表达式的值
        # 叶子
        if self.leftChild == None and self.rightChild == None:
            return self.linear_transform(self.node_operator(x))
        # 右为空，左不为空
        elif self.leftChild != None and self.rightChild == None:
            return self.node_operator(self.leftChild.compute_by_tree(x))
        # 左为空，右不为空
        elif self.leftChild == None and self.rightChild != None:
            return self.node_operator(self.rightChild.compute_by_tree(x))
        # 左右都不为空，二元操作符
        else:
            # 打印node_operator的类型
            return self.node_operator(self.leftChild.compute_by_tree(x), self.rightChild.compute_by_tree(x))

    
    @property
    def node_num(self):
        # 如果没有定义self._node_num，那么就计算一次
        if not hasattr(self, '_node_num'):
            self._node_num = 1
            if self.leftChild:
                self._node_num += self.leftChild.node_num
            if self.rightChild:
                self._node_num += self.rightChild.node_num
        return self._node_num
    
    @property
    def is_leaf(self):
        return self.leftChild == None and self.rightChild == None

    @property
    def leaves_num(self):
        # 如果没有定义self._leaves_num，那么就计算一次
        if not hasattr(self, '_leaves_num'):
            self._leaves_num = 1
            if self.leftChild:
                self._leaves_num += self.leftChild.leaves_num
            if self.rightChild:
                self._leaves_num += self.rightChild.leaves_num
        return self._leaves_num
    
    @property
    def unary_num(self):
        # 如果没有定义self._unary_num，那么就计算一次
        if not hasattr(self, '_unary_num'):
            self._unary_num = 1 if self.is_unary else 0
            if self.leftChild:
                self._unary_num += self.leftChild.unary_num
            if self.rightChild:
                self._unary_num += self.rightChild.unary_num
        return self._unary_num

    @property
    def binary_num(self):
        # 如果没有定义self._binary_num，那么就计算一次
        if not hasattr(self, '_binary_num'):
            self._binary_num = 0 if self.is_unary else 1
            if self.leftChild:
                self._binary_num += self.leftChild.binary_num
            if self.rightChild:
                self._binary_num += self.rightChild.binary_num
        return self._binary_num

    def create_same_tree(self):
        # 创建一棵与自身结构相同的树, 但是操作为空
        new_tree = BinaryTree(None, self.is_unary)
        if self.leftChild:
            new_tree.insert_leftChild(self.leftChild.create_same_tree())
        if self.rightChild:
            new_tree.insert_rightChild(self.rightChild.create_same_tree())
        return new_tree
    
    def get_leaves(self):
        # 获取叶子结点
        leaves = []
        self.inorder(lambda node: leaves.append(node) if node.is_leaf else None)
        return leaves

    
    
if __name__ == '__main__':
    # 测试
    # 生成一计算棵树，深度为3，(cosx*sinx)+sinx
    add_node = BinaryTree(binary[0], False)
    multiply_node = BinaryTree(binary[1], False)
    sin_node_1 = BinaryTree(unary[7])
    sin_node_2 = BinaryTree(unary[7])
    cos_node = BinaryTree(unary[8])
    # 生成一棵计算树，深度为3，(cosx*sinx)+sinx
    multiply_node.insert_leftChild(cos_node)
    multiply_node.insert_rightChild(sin_node_1)
    add_node.insert_leftChild(multiply_node)
    add_node.insert_rightChild(sin_node_2)
    
    # 中序遍历打印operator
    add_node.inorder(lambda node: print(node.node_operator))
    # # 测试计算树
    x = torch.tensor([0.5])
    # assert
    assert add_node.compute_by_tree(x) == (torch.cos(x)*torch.sin(x)+torch.sin(x))
    print('计算 test passed')
    
    assert add_node.node_num == 5
    print('node_num test passed')
    
    # 测试 set_operator
    add_node.set_operator([0, 1, 2, 3, 4])

