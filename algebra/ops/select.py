import lab as B
from plum import Dispatcher, Self

from .. import _dispatch
from ..function import Function, OneFunction, ZeroFunction, WrappedFunction
from ..algebra import proven, new
from ..util import squeeze, identical

__all__ = ["SelectedFunction"]


class SelectedFunction(WrappedFunction):
    """Select particular dimensions of the input features.

    Args:
        e (:class:`.elements.Function`): Function to wrap.
        *dims (tensor): Dimensions to select.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e, *dims):
        WrappedFunction.__init__(self, e)
        self.dims = tuple(None if x is None else _to_list(x) for x in dims)

    def render_wrap(self, e, formatter):
        return "{} : {}".format(e, squeeze(tuple(self.dims)))

    @_dispatch(Self)
    def __eq__(self, other):
        return self[0] == other[0] and identical(self.dims, other.dims)


def _to_list(x):
    if B.rank(x) == 0:
        return [x]
    elif B.rank(x) == 1:
        return list(x)
    else:
        raise ValueError(f'Could not convert "{x}" to a list.')


@_dispatch(Function, [object])
def select(a, *dims):
    return new(a, SelectedFunction)(a, *dims)


@_dispatch({ZeroFunction, OneFunction}, [object], precedence=proven())
def select(a, *dims):
    return a
