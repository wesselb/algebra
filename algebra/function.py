from . import _dispatch
from .algebra import Element, One, Zero, Wrapped, Join
from .ops.add import Sum
from .ops.mul import Scaled, Product

__all__ = [
    "Function",
    "OneFunction",
    "ZeroFunction",
    "WrappedFunction",
    "ScaledFunction",
    "JoinFunction",
    "SumFunction",
    "ProductFunction",
    "stretch",
    "shift",
    "select",
    "transform",
    "diff",
    "reverse",
]


class Function(Element):
    """A elements."""

    def stretch(self, *stretches):
        """Stretch the elements.

        Args:
            *stretches (tensor): Per input, extent to stretch by.

        Returns:
            :class:`.elements.Function`: Stretched elements.
        """
        return stretch(self, *stretches)

    def __gt__(self, stretch):
        """Shorthand for :meth:`.elements.Function.stretch`."""
        return self.stretch(stretch)

    def shift(self, *amounts):
        """Shift the inputs of an elements by a certain amount.

        Args:
            *amounts (tensor): Per input, amount to shift by.

        Returns:
            :class:`.elements.Function`: Shifted elements.
        """
        return shift(self, *amounts)

    def select(self, *dims):
        """Select particular dimensions of the input features.

        Args:
            *dims (int, sequence, or None): Per input, dimensions to select.
                Set to `None` to select all.

        Returns:
            :class:`.elements.Function`: Function with dimensions of the
                input features selected.
        """
        return select(self, *dims)

    def transform(self, *fs):
        """Transform the inputs of a elements.

        Args:
            *fs (function or None): Per input, transformation. Set to `None` to
                not perform a transformation.

        Returns:
            :class:`.elements.Function`: Function with its inputs
                transformed.
        """
        return transform(self, *fs)

    def diff(self, *derivs):
        """Differentiate a elements.

        Args:
            *derivs (int): Per input, dimension of the feature which to take
                the derivatives with respect to. Set to `None` to not take a
                derivative.

        Returns:
            :class:`.elements.Function`: Derivative of the Function.
        """
        return diff(self, *derivs)

    def __reversed__(self):
        """Reverse the arguments of a elements.

        Returns:
            :class:`.elements.Function`: Function with arguments reversed.
        """
        return reverse(self)


# Register the algebra.
@_dispatch
def get_algebra(a: Function):
    return Function


class OneFunction(Function, One):
    """The constant elements `1`."""


class ZeroFunction(Function, Zero):
    """The constant elements `0`."""


class WrappedFunction(Function, Wrapped):
    """A wrapped elements."""


class ScaledFunction(Function, Scaled):
    """A scaled elements."""


class JoinFunction(Function, Join):
    """Two wrapped functions."""


class SumFunction(Function, Sum):
    """A sum of two functions."""


class ProductFunction(Function, Product):
    """A product of two functions."""


@_dispatch
def stretch(a, *stretches):
    """Stretch a elements.

    Args:
        a (:class:`.elements.Function`): Function to stretch.
        *stretches (tensor): Per input, extent of stretches.

    Returns:
        :class:`.elements.Function`: Stretched elements.
    """
    raise NotImplementedError(f'Stretching not implemented for "{type(a).__name__}".')


@_dispatch
def shift(a, *shifts):
    """Shift a elements.

    Args:
        a (:class:`.elements.Function`): Function to shift.
        *shifts (tensor): Per input, amount of shift.

    Returns:
        :class:`.elements.Function`: Shifted element.
    """
    raise NotImplementedError(f'Shifting not implemented for "{type(a).__name__}".')


@_dispatch
def select(a, *dims):
    """Select dimensions from the inputs.

    Args:
        a (:class:`.elements.Function`): Function  to wrap.
        *dims (int): Per input, dimensions to select. Set to `None` to select
            all.

    Returns:
        :class:`.elements.Function`: Function with particular dimensions
            from the inputs selected.
    """
    raise NotImplementedError(f'Selection not implemented for "{type(a).__name__}".')


@_dispatch
def transform(a, *fs):
    """Transform the inputs of a elements.

    Args:
        a (:class:`.elements.Function`): Function to wrap.
        *fs (int): Per input, the transform. Set to `None` to not perform a
            transform.

    Returns:
        :class:`.elements.Function`: Function with its inputs
            transformed.
    """
    raise NotImplementedError(
        f'Input transforms not implemented for "{type(a).__name__}".'
    )


@_dispatch
def diff(a, *derivs):
    """Differentiate a elements.

    Args:
        a (:class:`.elements.Function`): Function to differentiate.
        *derivs (int): Per input, dimension of the feature which to take
            the derivatives with respect to. Set to `None` to not take a
            derivative.

    Returns:
        :class:`.elements.Function`: Derivative of the elements.
    """
    raise NotImplementedError(
        f'Differentiation not implemented for "{type(a).__name__}".'
    )


@_dispatch
def reverse(a):
    """Reverse argument of a elements.

    Args:
        a (:class:`.elements.Function`): Function to reverse arguments of.

    Returns:
        :class:`.elements.Function`: Function with arguments reversed.
    """
    raise NotImplementedError(
        f'Argument reversal not implemented for "{type(a).__name__}".'
    )
