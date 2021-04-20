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

    @_dispatch
    def __eq__(self, other: "Scaled"):
        return self[0] == other[0] and identical(self.scale, other.scale)


class Product(Join):
    """Product of elements."""

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

    @_dispatch
    def __eq__(self, other: "Product"):
        way1 = self[0] == other[0] and self[1] == other[1]
        way2 = self[0] == other[1] and self[1] == other[0]
        return way1 or way2


# Generic multiplication.


@_dispatch
def mul(a: Element, b):
    if b is 0:
        return new(a, Zero)()
    elif b is 1:
        return a
    else:
        return new(a, Scaled)(a, b)


@_dispatch
def mul(a, b: Element):
    return mul(b, a)


@_dispatch
def mul(a: Element, b: Element):
    return new(a, Product)(a, b)


# Cancel redundant zeros and ones.


@_dispatch(precedence=proven())
def mul(a: Zero, b):
    return a


@_dispatch(precedence=proven())
def mul(a, b: Zero):
    return b


@_dispatch(precedence=proven())
def mul(a: Zero, b: Zero):
    return a


@_dispatch(precedence=proven())
def mul(a: One, b: Element):
    return b


@_dispatch(precedence=proven())
def mul(a: Element, b: One):
    return a


@_dispatch(precedence=proven())
def mul(a: One, b: One):
    return a


# Group factors and terms if possible.


@_dispatch
def mul(a, b: Scaled):
    return mul(b.scale * a, b[0])


@_dispatch
def mul(a: Scaled, b):
    return mul(a.scale * b, a[0])


@_dispatch
def mul(a: Scaled, b: Element):
    return mul(a.scale, mul(a[0], b))


@_dispatch
def mul(a: Element, b: Scaled):
    return mul(b.scale, mul(a, b[0]))


@_dispatch
def mul(a: Scaled, b: Scaled):
    return new(a, Scaled)(mul(a[0], b[0]), a.scale * b.scale)
