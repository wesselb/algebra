from . import _dispatch
from .ring import (
    pretty_print,
    Formatter,

    Element,
    Wrapped,
    Join,
    Scaled,
    Sum,
    Product
)

__all__ = []


@_dispatch(Wrapped, Formatter)
def pretty_print(el, formatter):
    return el.render(pretty_print(el[0], el, formatter), formatter)


@_dispatch(Join, Formatter)
def pretty_print(el, formatter):
    return el.render(pretty_print(el[0], el, formatter),
                     pretty_print(el[1], el, formatter), formatter)


@_dispatch(Element, Element, Formatter)
def pretty_print(el, parent, formatter):
    if need_parens(el, parent):
        return '(' + pretty_print(el, formatter) + ')'
    else:
        return pretty_print(el, formatter)


@_dispatch(Element, Sum)
def need_parens(el, parent):
    """Check whether `el` needs parentheses when printed in `parent`.

    Args:
        el (:class:`.field.Element`): Element to print.
        parent (:class:`.field.Element`): Parent of element to print.

    Returns:
        bool: Boolean whether `el` needs parentheses.
    """
    return False


@_dispatch(Element, Product)
def need_parens(el, parent):
    return False


@_dispatch({Sum, Wrapped}, Product)
def need_parens(el, parent):
    return True


@_dispatch(Scaled, Product)
def need_parens(el, parent):
    return False


@_dispatch(Element, Wrapped)
def need_parens(el, parent):
    return False


@_dispatch({Wrapped, Join}, Wrapped)
def need_parens(el, parent):
    return True


@_dispatch({Product, Scaled}, Scaled)
def need_parens(el, parent):
    return False
