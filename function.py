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

function_to_str = {
    lambda x: 0*x**2: '0',
    lambda x: 1+0*x**2: '1',
    lambda x: x+0*x**2: 'x',
    lambda x: x**2: 'x^2',
    lambda x: x**3: 'x^3',
    lambda x: x**4: 'x^4',
    torch.exp: 'exp',
    torch.sin: 'sin',
    torch.cos: 'cos',
    lambda x,y: x+y: '+',
    lambda x,y: x*y: '*',
    lambda x,y: x-y: '-',
}


# # 非叶子结点还有alpha beta两个参数
# unary_functions_str = ['({}*(0)+{})',
#                        '({}*(1)+{})',
#                        # '5',
#                        '({}*{}+{})',
#                        # '-{}',
#                        '({}*({})**2+{})',
#                        '({}*({})**3+{})',
#                        '({}*({})**4+{})',
#                        # '({})**5',
#                        '({}*exp({})+{})',
#                        '({}*sin({})+{})',
#                        '({}*cos({})+{})',]
#                        # 'ref({})',
#                        # 'exp(-({})**2/2)']

# unary_functions_str_leaf= ['(0)',
#                            '(1)',
#                            # '5',
#                            '({})',
#                            # '-{}',
#                            '(({})**2)',
#                            '(({})**3)',
#                            '(({})**4)',
#                            # '({})**5',
#                            '(exp({}))',
#                            '(sin({}))',
#                            '(cos({}))',]


# binary_functions_str = ['(({})+({}))',
#                         '(({})*({}))',
#                         '(({})-({}))']

if __name__ == '__main__':
    pass