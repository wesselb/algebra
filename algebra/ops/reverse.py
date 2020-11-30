from plum import Dispatcher, Self

from .diff import DerivativeFunction
from .select import SelectedFunction
from .shift import ShiftedFunction
from .stretch import StretchedFunction
from .tensor import TensorProductFunction
from .transform import InputTransformedFunction
from .. import _dispatch
from ..function import (
    Function,
    OneFunction,
    ZeroFunction,
    WrappedFunction,
    ScaledFunction,
    SumFunction,
    ProductFunction,
    stretch,
    shift,
    select,
    transform,
    diff,
)
from ..algebra import proven, new, add, mul

__all__ = ["ReversedFunction"]


class ReversedFunction(WrappedFunction):
    """Function with arguments reversed.

    Args:
        e (:class:`.elements.Function`): Function to reverse arguments of.
    """

    _dispatch = Dispatcher(in_class=Self)

    def render_wrap(self, e, formatter):
        return f"Reversed({e})"

    @_dispatch(Self)
    def __eq__(self, other):
        return self[0] == other[0]


# A reversed elements will never need parentheses.


@_dispatch(Function, ReversedFunction, precedence=proven())
def need_parens(el, parent):
    return False


@_dispatch(ReversedFunction, Function, precedence=proven())
def need_parens(el, parent):
    return False


# Implement basic methods for reverse function.


@_dispatch(Function)
def reverse(a):
    return new(a, ReversedFunction)(a)


@_dispatch({ZeroFunction, OneFunction})
def reverse(a):
    return a


# Propagate reversal.


@_dispatch(SumFunction)
def reverse(a):
    return add(reverse(a[0]), reverse(a[1]))


@_dispatch(ProductFunction)
def reverse(a):
    return mul(reverse(a[0]), reverse(a[1]))


@_dispatch(ScaledFunction)
def reverse(a):
    return mul(a.scale, reverse(a[0]))


# Let reversal synergise with wrapped kernels.


@_dispatch(ReversedFunction)
def reverse(a):
    return a[0]


@_dispatch(StretchedFunction)
def reverse(a):
    return stretch(reverse(a[0]), *reversed(a.stretches))


@_dispatch(ShiftedFunction)
def reverse(a):
    return shift(reverse(a[0]), *reversed(a.shifts))


@_dispatch(SelectedFunction)
def reverse(a):
    return select(reverse(a[0]), *reversed(a.dims))


@_dispatch(InputTransformedFunction)
def reverse(a):
    return transform(reverse(a[0]), *reversed(a.fs))


@_dispatch(DerivativeFunction)
def reverse(a):
    return diff(reverse(a[0]), *reversed(a.derivs))


@_dispatch(TensorProductFunction)
def reverse(a):
    return new(a, TensorProductFunction)(*reversed(a.fs))
