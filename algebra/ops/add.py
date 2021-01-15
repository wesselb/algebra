from plum import Dispatcher, Self

from .. import _dispatch
from .mul import mul, Scaled
from ..algebra import proven, new, Element, Zero, One, Join

__all__ = ["Sum"]


class Sum(Join):
    """Sum of elements."""

    _dispatch = Dispatcher(in_class=Self)

    @property
    def num_terms(self):
        return self[0].num_terms + self[1].num_terms

    def term(self, i):
        if i >= self.num_terms:
            raise IndexError("Index out of range.")
        if i < self[0].num_terms:
            return self[0].term(i)
        else:
            return self[1].term(i - self[0].num_terms)

    def render_join(self, e1, e2, formatter):
        return f"{e1} + {e2}"

    @_dispatch(Self)
    def __eq__(self, other):
        way1 = self[0] == other[0] and self[1] == other[1]
        way2 = self[0] == other[1] and self[1] == other[0]
        return way1 or way2


# Generic addition.


@_dispatch(Element, object)
def add(a, b):
    if b is 0:
        return a
    else:
        return add(a, mul(b, new(a, One)()))


@_dispatch(object, Element)
def add(a, b):
    if a is 0:
        return b
    else:
        return add(mul(a, new(b, One)()), b)


@_dispatch(Element, Element)
def add(a, b):
    if a == b:
        return mul(2, a)
    else:
        return new(a, Sum)(a, b)


# Cancel redundant zeros and ones.


@_dispatch(Zero, object, precedence=proven())
def add(a, b):
    if b is 0:
        return a
    else:
        return mul(new(a, One)(), b)


@_dispatch(object, Zero, precedence=proven())
def add(a, b):
    if a is 0:
        return b
    else:
        return mul(a, new(b, One)())


@_dispatch(Zero, Zero, precedence=proven())
def add(a, b):
    return a


@_dispatch(Element, Zero, precedence=proven())
def add(a, b):
    return a


@_dispatch(Zero, Element, precedence=proven())
def add(a, b):
    return b


# Group factors and terms if possible.


@_dispatch(Scaled, Element)
def add(a, b):
    if a[0] == b:
        return mul(a.scale + 1, b)
    else:
        return new(a, Sum)(a, b)


@_dispatch(Element, Scaled)
def add(a, b):
    if a == b[0]:
        return mul(b.scale + 1, a)
    else:
        return new(a, Sum)(a, b)


@_dispatch(Scaled, Scaled)
def add(a, b):
    if a[0] == b[0]:
        return mul(a.scale + b.scale, a[0])
    else:
        return new(a, Sum)(a, b)
