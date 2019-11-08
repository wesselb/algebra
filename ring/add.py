from . import _dispatch
from .mul import mul
from .ring import (
    definite,
    new,

    Element,
    Zero,
    One,
    Scaled,
    Sum
)

__all__ = []


# Generic addition.

@_dispatch(Element, object)
def add(a, b):
    if b is 0:
        return a
    else:
        return new(a, Sum)(a, mul(b, new(a, One)()))


@_dispatch(object, Element)
def add(a, b):
    if a is 0:
        return b
    else:
        return new(b, Sum)(mul(a, new(b, One)()), b)


@_dispatch(Element, Element)
def add(a, b):
    if a == b:
        return mul(2, a)
    else:
        return new(a, Sum)(a, b)


# Cancel redundant zeros and ones.

@_dispatch(Zero, object, precedence=definite)
def add(a, b):
    if b is 0:
        return a
    else:
        return mul(new(a, One)(), b)


@_dispatch(object, Zero, precedence=definite)
def add(a, b):
    if a is 0:
        return b
    else:
        return mul(a, new(b, One)())


@_dispatch(Zero, Zero, precedence=definite)
def add(a, b): return a


@_dispatch(Element, Zero, precedence=definite)
def add(a, b): return a


@_dispatch(Zero, Element, precedence=definite)
def add(a, b): return b


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
