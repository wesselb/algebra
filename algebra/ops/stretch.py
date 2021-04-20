import operator
from typing import Union

from .. import _dispatch
from ..algebra import proven, new
from ..function import Function, OneFunction, ZeroFunction, WrappedFunction
from ..util import to_tensor, squeeze, identical, broadcast

__all__ = ["StretchedFunction"]


class StretchedFunction(WrappedFunction):
    """Stretched elements.

    Args:
        e (:class:`.elements.Function`): Function to stretch.
        *stretches (tensor): Extent of stretches.
    """

    def __init__(self, e, *stretches):
        WrappedFunction.__init__(self, e)
        self.stretches = tuple(to_tensor(x) for x in stretches)

    def render_wrap(self, e, formatter):
        stretches = tuple(formatter(s) for s in self.stretches)
        return "{} > {}".format(e, squeeze(stretches))

    @_dispatch
    def __eq__(self, other: "StretchedFunction"):
        return self[0] == other[0] and identical(self.stretches, other.stretches)


@_dispatch
def stretch(a: Function, *stretches):
    return new(a, StretchedFunction)(a, *stretches)


@_dispatch(precedence=proven())
def stretch(a: Union[ZeroFunction, OneFunction], *stretches):
    return a


@_dispatch(precedence=proven())
def stretch(a: StretchedFunction, *stretches):
    return stretch(a[0], *broadcast(operator.mul, a.stretches, stretches))
