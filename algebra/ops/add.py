from .mul import mul, Scaled
from .. import _dispatch
from ..algebra import proven, new, Element, Zero, One, Join

__all__ = ["Sum"]


class Sum(Join):
    """Sum of elements."""

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

    @_dispatch
    def __eq__(self, other: "Sum"):
        way1 = self[0] == other[0] and self[1] == other[1]
        way2 = self[0] == other[1] and self[1] == other[0]
        return way1 or way2


# Generic addition.


@_dispatch
def add(a: Element, b):
    if b is 0:
        return a
    else:
        return add(a, mul(b, new(a, One)()))


@_dispatch
def add(a, b: Element):
    if a is 0:
        return b
    else:
        return add(mul(a, new(b, One)()), b)


@_dispatch
def add(a: Element, b: Element):
    if a == b:
        return mul(2, a)
    else:
        return new(a, Sum)(a, b)


# Cancel redundant zeros and ones.


@_dispatch(precedence=proven())
def add(a: Zero, b):
    if b is 0:
        return a
    else:
        return mul(new(a, One)(), b)


@_dispatch(precedence=proven())
def add(a, b: Zero):
    if a is 0:
        return b
    else:
        return mul(a, new(b, One)())


@_dispatch(precedence=proven())
def add(a: Zero, b: Zero):
    return a


@_dispatch(precedence=proven())
def add(a: Element, b: Zero):
    return a


@_dispatch(precedence=proven())
def add(a: Zero, b: Element):
    return b


# Group factors and terms if possible.


@_dispatch
def add(a: Scaled, b: Element):
    if a[0] == b:
        return mul(a.scale + 1, b)
    else:
        return new(a, Sum)(a, b)


@_dispatch
def add(a: Element, b: Scaled):
    if a == b[0]:
        return mul(b.scale + 1, a)
    else:
        return new(a, Sum)(a, b)


@_dispatch
def add(a: Scaled, b: Scaled):
    if a[0] == b[0]:
        return mul(a.scale + b.scale, a[0])
    else:
        return new(a, Sum)(a, b)
