from . import _dispatch
from .ops.add import Sum
from .ops.mul import Scaled, Product
from .algebra import pretty_print, Element, Wrapped, Join

__all__ = []


@_dispatch(Wrapped, object)
def pretty_print(el, formatter):
    return el.render_wrap(pretty_print(el[0], el, formatter), formatter)


@_dispatch(Join, object)
def pretty_print(el, formatter):
    return el.render_join(
        pretty_print(el[0], el, formatter),
        pretty_print(el[1], el, formatter),
        formatter,
    )


@_dispatch(Element, Element, object)
def pretty_print(el, parent, formatter):
    if need_parens(el, parent):
        return "(" + pretty_print(el, formatter) + ")"
    else:
        return pretty_print(el, formatter)


@_dispatch(Element, Sum)
def need_parens(el, parent):
    """Check whether `el` needs parentheses when printed in `parent`.

    Args:
        el (:class:`.algebra.Element`): Element to print.
        parent (:class:`.algebra.Element`): Parent of element to print.

    Returns:
        bool: Boolean indicating whether `el` needs parentheses.
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
