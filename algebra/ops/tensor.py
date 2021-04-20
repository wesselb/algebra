from types import FunctionType as PythonFunction

from .. import _dispatch
from ..algebra import proven, new, add, mul
from ..function import Function
from ..util import identical

__all__ = ["TensorProductFunction"]


class TensorProductFunction(Function):
    """An element built from a product of functions for each input.

    Args:
        *fs (function): Per input, a elements.
    """

    def __init__(self, *fs):
        self.fs = fs

    def render(self, formatter):
        if len(self.fs) == 1:
            return self.fs[0].__name__
        else:
            return "{}".format(" x ".join(f.__name__ for f in self.fs))

    @_dispatch
    def __eq__(self, other: "TensorProductFunction"):
        return identical(self.fs, other.fs)


# A tensor product elements needs parentheses if and only if it has more than
# one elements.


@_dispatch(precedence=proven())
def need_parens(el: TensorProductFunction, parent: Function):
    return len(el.fs) > 1


# Handle conversion of Python functions.


@mul.dispatch(precedence=proven())
def mul(a: Function, b: PythonFunction):
    return mul(a, new(a, TensorProductFunction)(b))


@mul.dispatch(precedence=proven())
def mul(a: PythonFunction, b: Function):
    return mul(new(b, TensorProductFunction)(a), b)


@add.dispatch(precedence=proven())
def add(a: Function, b: PythonFunction):
    return add(a, new(a, TensorProductFunction)(b))


@add.dispatch(precedence=proven())
def add(a: PythonFunction, b: Function):
    return add(new(b, TensorProductFunction)(a), b)
