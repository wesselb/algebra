from types import FunctionType as PythonFunction

from plum import Dispatcher, Self

from .. import _dispatch
from ..function import Function
from ..algebra import proven, new, add, mul
from ..util import identical

__all__ = ["TensorProductFunction"]


class TensorProductFunction(Function):
    """An element built from a product of functions for each input.

    Args:
        *fs (function): Per input, a elements.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, *fs):
        self.fs = fs

    def render(self, formatter):
        if len(self.fs) == 1:
            return self.fs[0].__name__
        else:
            return "{}".format(" x ".join(f.__name__ for f in self.fs))

    @_dispatch(Self)
    def __eq__(self, other):
        return identical(self.fs, other.fs)


# A tensor product elements needs parentheses if and only if it has more than
# one elements.


@_dispatch(TensorProductFunction, Function, precedence=proven())
def need_parens(el, parent):
    return len(el.fs) > 1


# Handle conversion of Python functions.


@mul.extend(Function, PythonFunction, precedence=proven())
def mul(a, b):
    return mul(a, new(a, TensorProductFunction)(b))


@mul.extend(PythonFunction, Function, precedence=proven())
def mul(a, b):
    return mul(new(b, TensorProductFunction)(a), b)


@add.extend(Function, PythonFunction, precedence=proven())
def add(a, b):
    return add(a, new(a, TensorProductFunction)(b))


@add.extend(PythonFunction, Function, precedence=proven())
def add(a, b):
    return add(new(b, TensorProductFunction)(a), b)
