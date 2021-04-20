from typing import Union

from .diff import DerivativeFunction
from .select import SelectedFunction
from .shift import ShiftedFunction
from .stretch import StretchedFunction
from .tensor import TensorProductFunction
from .transform import InputTransformedFunction
from .. import _dispatch
from ..algebra import proven, new, add, mul
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

__all__ = ["ReversedFunction"]


class ReversedFunction(WrappedFunction):
    """Function with arguments reversed.

    Args:
        e (:class:`.elements.Function`): Function to reverse arguments of.
    """

    def render_wrap(self, e, formatter):
        return f"Reversed({e})"

    @_dispatch
    def __eq__(self, other: "ReversedFunction"):
        return self[0] == other[0]


# A reversed elements will never need parentheses.


@_dispatch(precedence=proven())
def need_parens(el: Function, parent: ReversedFunction):
    return False


@_dispatch(precedence=proven())
def need_parens(el: ReversedFunction, parent: Function):
    return False


# Implement basic methods for reverse function.


@_dispatch
def reverse(a: Function):
    return new(a, ReversedFunction)(a)


@_dispatch
def reverse(a: Union[ZeroFunction, OneFunction]):
    return a


# Propagate reversal.


@_dispatch
def reverse(a: SumFunction):
    return add(reverse(a[0]), reverse(a[1]))


@_dispatch
def reverse(a: ProductFunction):
    return mul(reverse(a[0]), reverse(a[1]))


@_dispatch
def reverse(a: ScaledFunction):
    return mul(a.scale, reverse(a[0]))


# Let reversal synergise with wrapped kernels.


@_dispatch
def reverse(a: ReversedFunction):
    return a[0]


@_dispatch
def reverse(a: StretchedFunction):
    return stretch(reverse(a[0]), *reversed(a.stretches))


@_dispatch
def reverse(a: ShiftedFunction):
    return shift(reverse(a[0]), *reversed(a.shifts))


@_dispatch
def reverse(a: SelectedFunction):
    return select(reverse(a[0]), *reversed(a.dims))


@_dispatch
def reverse(a: InputTransformedFunction):
    return transform(reverse(a[0]), *reversed(a.fs))


@_dispatch
def reverse(a: DerivativeFunction):
    return diff(reverse(a[0]), *reversed(a.derivs))


@_dispatch
def reverse(a: TensorProductFunction):
    return new(a, TensorProductFunction)(*reversed(a.fs))
