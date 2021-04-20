from typing import Union

import operator

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

    def __init__(self, e, *shifts):
        WrappedFunction.__init__(self, e)
        self.shifts = tuple(to_tensor(x) for x in shifts)

    def render_wrap(self, e, formatter):
        shifts = tuple(formatter(s) for s in self.shifts)
        return "{} shift {}".format(e, squeeze(shifts))

    @_dispatch
    def __eq__(self, other: "ShiftedFunction"):
        return self[0] == other[0] and identical(self.shifts, other.shifts)


@_dispatch
def shift(a: Function, *shifts):
    return new(a, ShiftedFunction)(a, *shifts)


@_dispatch(precedence=proven())
def shift(a: Union[ZeroFunction, OneFunction], *shifts):
    return a


@_dispatch(precedence=proven())
def shift(a: ShiftedFunction, *shifts):
    return shift(a[0], *broadcast(operator.add, a.shifts, shifts))
