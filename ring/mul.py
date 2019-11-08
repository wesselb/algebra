from . import _dispatch
from .ring import (
    definite,
    priority,
    new,

    Element,
    Zero,
    One,
    Scaled,
    Product
)

__all__ = []


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
def mul(a, b): return new(a, Product)(a, b)


# Cancel redundant zeros and ones.

@_dispatch(Zero, object, precedence=definite)
def mul(a, b): return a


@_dispatch(object, Zero, precedence=definite)
def mul(a, b): return b


@_dispatch(Zero, Zero, precedence=definite)
def mul(a, b): return a


@_dispatch(One, Element, precedence=priority)
def mul(a, b): return b


@_dispatch(Element, One, precedence=priority)
def mul(a, b): return a


@_dispatch(One, One, precedence=priority)
def mul(a, b): return a


# Group factors and terms if possible.

@_dispatch(object, Scaled)
def mul(a, b): return mul(b.scale * a, b[0])


@_dispatch(Scaled, object)
def mul(a, b): return mul(a.scale * b, a[0])


@_dispatch(Scaled, Element)
def mul(a, b): return mul(a.scale, mul(a[0], b))


@_dispatch(Element, Scaled)
def mul(a, b): return mul(b.scale, mul(a, b[0]))


@_dispatch(Scaled, Scaled)
def mul(a, b):
    if a[0] == b[0]:
        return new(a, Scaled)(a[0], a.scale * b.scale)
    else:
        scaled = new(a, Scaled)(a[0], a.scale * b.scale)
        return new(a, Product)(scaled, b[0])
