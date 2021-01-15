from plum import Dispatcher, Self

from .. import _dispatch
from ..function import (
    Function,
    OneFunction,
    ZeroFunction,
    WrappedFunction,
)
from ..algebra import proven, new
from ..util import identical

__all__ = ["DerivativeFunction"]


class DerivativeFunction(WrappedFunction):
    """Compute the derivative of a elements.

    Args:
        e (:class:`.elements.Function`): Function to compute the
            derivative of.
        *derivs (tensor): Per input, the index of the dimension which to
            take the derivative of. Set to `None` to not take a derivative.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e, *derivs):
        WrappedFunction.__init__(self, e)
        self.derivs = derivs

    def render_wrap(self, e, formatter):
        if len(self.derivs) == 1:
            derivs = "({})".format(self.derivs[0])
        else:
            derivs = self.derivs
        return "d{} {}".format(derivs, e)

    @_dispatch(Self)
    def __eq__(self, other):
        return self[0] == other[0] and identical(self.derivs, other.derivs)


@_dispatch(Function, [object])
def diff(a, *derivs):
    return new(a, DerivativeFunction)(a, *derivs)


@_dispatch(ZeroFunction, [object], precedence=proven())
def diff(a, *derivs):
    return a


@_dispatch(OneFunction, [object], precedence=proven())
def diff(a, *derivs):
    return new(a, ZeroFunction)()
