from plum import Dispatcher, Self

from .. import _dispatch
from ..function import Function, OneFunction, ZeroFunction, WrappedFunction
from ..algebra import proven, new
from ..util import identical

__all__ = ["InputTransformedFunction"]


class InputTransformedFunction(WrappedFunction):
    """Transform inputs of a elements.

    Args:
        e (:class:`.elements.Function`): Function to transform inputs of.
        *fs (tensor): Per input, the transformation. Set to `None` to not
            do a transformation.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e, *fs):
        WrappedFunction.__init__(self, e)
        self.fs = fs

    def render_wrap(self, e, formatter):
        # Safely get a elements's name.
        def name(f):
            return "None" if f is None else f.__name__

        if len(self.fs) == 1:
            fs = name(self.fs[0])
        else:
            fs = "({})".format(", ".join(name(f) for f in self.fs))
        return "{} transform {}".format(e, fs)

    @_dispatch(Self)
    def __eq__(self, other):
        return self[0] == other[0] and identical(self.fs, other.fs)


@_dispatch(Function, [object])
def transform(a, *fs):
    return new(a, InputTransformedFunction)(a, *fs)


@_dispatch({ZeroFunction, OneFunction}, [object], precedence=proven())
def transform(a, *fs):
    return a
