import operator

from plum import Dispatcher, Self

from .. import _dispatch
from ..function import Function, OneFunction, ZeroFunction, WrappedFunction
from ..algebra import proven, new
from ..util import to_tensor, squeeze, identical, broadcast

__all__ = ["StretchedFunction"]


class StretchedFunction(WrappedFunction):
    """Stretched elements.

    Args:
        e (:class:`.elements.Function`): Function to stretch.
        *stretches (tensor): Extent of stretches.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e, *stretches):
        WrappedFunction.__init__(self, e)
        self.stretches = tuple(to_tensor(x) for x in stretches)

    def render_wrap(self, e, formatter):
        stretches = tuple(formatter(s) for s in self.stretches)
        return "{} > {}".format(e, squeeze(stretches))

    @_dispatch(Self)
    def __eq__(self, other):
        return self[0] == other[0] and identical(self.stretches, other.stretches)


@_dispatch(Function, [object])
def stretch(a, *stretches):
    return new(a, StretchedFunction)(a, *stretches)


@_dispatch({ZeroFunction, OneFunction}, [object], precedence=proven())
def stretch(a, *stretches):
    return a


@_dispatch(StretchedFunction, [object], precedence=proven())
def stretch(a, *stretches):
    return stretch(a[0], *broadcast(operator.mul, a.stretches, stretches))
