import operator

from plum import Dispatcher, Self

from .. import _dispatch
from ..algebra import new, proven
from ..function import Function, OneFunction, ZeroFunction, WrappedFunction
from ..util import to_tensor, squeeze, broadcast, identical

__all__ = ["ShiftedFunction"]


class ShiftedFunction(WrappedFunction):
    """Shifted elements.

    Args:
        e (:class:`.elements.Function`): Function to shift.
        *shifts (tensor): Shift amounts.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e, *shifts):
        WrappedFunction.__init__(self, e)
        self.shifts = tuple(to_tensor(x) for x in shifts)

    def render_wrap(self, e, formatter):
        shifts = tuple(formatter(s) for s in self.shifts)
        return "{} shift {}".format(e, squeeze(shifts))

    @_dispatch(Self)
    def __eq__(self, other):
        return self[0] == other[0] and identical(self.shifts, other.shifts)


@_dispatch(Function, [object])
def shift(a, *shifts):
    return new(a, ShiftedFunction)(a, *shifts)


@_dispatch({ZeroFunction, OneFunction}, [object], precedence=proven())
def shift(a, *shifts):
    return a


@_dispatch(ShiftedFunction, [object], precedence=proven())
def shift(a, *shifts):
    return shift(a[0], *broadcast(operator.add, a.shifts, shifts))
