import lab as B
from plum import Dispatcher, Self

from .. import _dispatch
from ..algebra import proven, new, Element, Zero, One, Wrapped, Join
from ..util import identical

__all__ = ["Scaled", "Product"]


class Scaled(Wrapped):
    """Scaled element.

    Args:
        e (:class:`.algebra.Element`): Element to scale.
        scale (tensor): Scale.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e, scale):
        Wrapped.__init__(self, e)
        self.scale = scale

    @property
    def num_factors(self):
        return self[0].num_factors + 1

    def render_wrap(self, e, formatter):
        return f"{formatter(self.scale)} * {e}"

    def factor(self, i):
        if i >= self.num_factors:
            raise IndexError("Index out of range.")
        else:
            return self.scale if i == 0 else self[0].factor(i - 1)

    @_dispatch(Self)
    def __eq__(self, other):
        return self[0] == other[0] and identical(self.scale, other.scale)


class Product(Join):
    """Product of elements."""

    _dispatch = Dispatcher(in_class=Self)

    @property
    def num_factors(self):
        return self[0].num_factors + self[1].num_factors

    def factor(self, i):
        if i >= self.num_factors:
            raise IndexError("Index out of range.")
        if i < self[0].num_factors:
            return self[0].factor(i)
        else:
            return self[1].factor(i - self[0].num_factors)

    def render_join(self, e1, e2, formatter):
        return f"{e1} * {e2}"

    @_dispatch(Self)
    def __eq__(self, other):
        way1 = self[0] == other[0] and self[1] == other[1]
        way2 = self[0] == other[1] and self[1] == other[0]
        return way1 or way2


# Generic multiplication.


@_dispatch(Element, object)
def mul(a, b):
    if b is 0:
        return new(a, Zero)()
    elif b is 1:
        return a
    else:
        return new(a, Scaled)(a, b)


@_dispatch(object, Element)
def mul(a, b):
    return mul(b, a)


@_dispatch(Element, Element)
def mul(a, b):
    return new(a, Product)(a, b)


# Cancel redundant zeros and ones.


@_dispatch(Zero, object, precedence=proven())
def mul(a, b):
    return a


@_dispatch(object, Zero, precedence=proven())
def mul(a, b):
    return b


@_dispatch(Zero, Zero, precedence=proven())
def mul(a, b):
    return a


@_dispatch(One, Element, precedence=proven())
def mul(a, b):
    return b


@_dispatch(Element, One, precedence=proven())
def mul(a, b):
    return a


@_dispatch(One, One, precedence=proven())
def mul(a, b):
    return a


# Group factors and terms if possible.


@_dispatch(object, Scaled)
def mul(a, b):
    return mul(b.scale * a, b[0])


@_dispatch(Scaled, object)
def mul(a, b):
    return mul(a.scale * b, a[0])


@_dispatch(Scaled, Element)
def mul(a, b):
    return mul(a.scale, mul(a[0], b))


@_dispatch(Element, Scaled)
def mul(a, b):
    return mul(b.scale, mul(a, b[0]))


@_dispatch(Scaled, Scaled)
def mul(a, b):
    return new(a, Scaled)(mul(a[0], b[0]), a.scale * b.scale)
