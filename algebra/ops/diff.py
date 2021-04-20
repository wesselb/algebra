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

    def __init__(self, e, *derivs):
        WrappedFunction.__init__(self, e)
        self.derivs = derivs

    def render_wrap(self, e, formatter):
        if len(self.derivs) == 1:
            derivs = "({})".format(self.derivs[0])
        else:
            derivs = self.derivs
        return "d{} {}".format(derivs, e)

    @_dispatch
    def __eq__(self, other: "DerivativeFunction"):
        return self[0] == other[0] and identical(self.derivs, other.derivs)


@_dispatch
def diff(a: Function, *derivs):
    return new(a, DerivativeFunction)(a, *derivs)


@_dispatch(precedence=proven())
def diff(a: ZeroFunction, *derivs):
    return a


@_dispatch(precedence=proven())
def diff(a: OneFunction, *derivs):
    return new(a, ZeroFunction)()
