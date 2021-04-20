from typing import Union

from . import _dispatch
from .algebra import pretty_print, Element, Wrapped, Join
from .ops.add import Sum
from .ops.mul import Scaled, Product

__all__ = []


@_dispatch
def pretty_print(el: Wrapped, formatter):
    return el.render_wrap(pretty_print(el[0], el, formatter), formatter)


@_dispatch
def pretty_print(el: Join, formatter):
    return el.render_join(
        pretty_print(el[0], el, formatter),
        pretty_print(el[1], el, formatter),
        formatter,
    )


@_dispatch
def pretty_print(el: Element, parent: Element, formatter):
    if need_parens(el, parent):
        return "(" + pretty_print(el, formatter) + ")"
    else:
        return pretty_print(el, formatter)


@_dispatch
def need_parens(el: Element, parent: Sum):
    """Check whether `el` needs parentheses when printed in `parent`.

    Args:
        el (:class:`.algebra.Element`): Element to print.
        parent (:class:`.algebra.Element`): Parent of element to print.

    Returns:
        bool: Boolean indicating whether `el` needs parentheses.
    """
    return False


@_dispatch
def need_parens(el: Element, parent: Product):
    return False


@_dispatch
def need_parens(el: Union[Sum, Wrapped], parent: Product):
    return True


@_dispatch
def need_parens(el: Scaled, parent: Product):
    return False


@_dispatch
def need_parens(el: Element, parent: Wrapped):
    return False


@_dispatch
def need_parens(el: Union[Wrapped, Join], parent: Wrapped):
    return True


@_dispatch
def need_parens(el: Union[Product, Scaled], parent: Scaled):
    return False
