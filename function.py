import numpy as np
import torch
from torch import sin, cos, exp

unary_functions = [lambda x: 0*x**2,    # 0
                   lambda x: 1+0*x**2,  # 1
                   lambda x: x+0*x**2,  # x
                   lambda x: x**2,    # x^2
                   lambda x: x**3,  # x^3
                   lambda x: x**4,  # x^4
                   torch.exp,
                   torch.sin,
                   torch.cos,]

binary_functions = [lambda x,y: x+y,
                    lambda x,y: x*y,
                    lambda x,y: x-y]

unary_str = ['0', '1', '', '^2', '^3', '^4', 'exp', 'sin', 'cos']
binary_str = ['+', '*', '-']


if __name__ == '__main__':
    pass