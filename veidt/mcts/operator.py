import numpy as np


class Operator:
    def __init__(self, opt, rep, singular, order=False):
        self.opt = opt
        self.singular = singular
        self.order = order
        self.rep = rep

    @property
    def is_singular(self):
        return self.singular

    @property
    def is_ordered(self):
        return self.order

    def compute(self, input):
        if self.is_singular:
            return self.opt(input)
        else:
            return self.opt(input[0], input[1])

    def __call__(self, input):
        return self.compute(input)

    def __str__(self):
        return self.rep


def get_operators():
    squre = Operator(lambda x: x ** 2, '^2', True)
    cubic = Operator(lambda x: x ** 3, '^3', True)
    add = Operator(lambda x, y: x + y, '+', False, False)
    red = Operator(lambda x, y: x - y, '-', False, True)
    mul = Operator(lambda x, y: x * y, '*', False, False)
    div = Operator(lambda x, y: x / y, '/', False, True)
    abso = Operator(np.abs, '||', True, False)
    abslog10 = Operator(lambda x: np.log10(abs(x).astype('float64')), 'abslog10', True, False)
    exp = Operator(lambda x: np.exp(x.astype('float64')), 'exp', True, False)
    sqrt = Operator(lambda x: np.sqrt(x.astype('float64')), 'sqrt', True, False)
    cbrt = Operator(lambda x: np.cbrt(x.astype('float64')), 'cbrt', True, False)
    singular_op = [abso, abslog10, exp, squre, cubic, sqrt, cbrt]
    binary_op = [add, red, mul, div]

    return singular_op, binary_op
