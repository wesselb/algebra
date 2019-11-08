import operator
from types import FunctionType as PythonFunction

from lab import B
from plum import Dispatcher, Self

from . import _dispatch
from .ring import (
    priority,
    definite,

    Element,
    One,
    Zero,
    Wrapped,
    Scaled,
    Join,
    Product,
    Sum,

    add,
    mul,
    get_ring,
    new
)
from .util import squeeze, broadcast, to_tensor, tuple_equal

__all__ = ['Function',
           'OneFunction',
           'ZeroFunction',
           'WrappedFunction',
           'ScaledFunction',
           'JoinFunction',
           'SumFunction',
           'ProductFunction',

           'StretchedFunction',
           'ShiftedFunction',
           'SelectedFunction',
           'InputTransformedFunction',
           'DerivativeFunction',
           'ReversedFunction',
           'TensorProductFunction',

           'get_ring',
           'new',
           'add',
           'mul',

           'stretch',
           'shift',
           'select',
           'transform',
           'differentiate',
           'reverse']


class Function(Element):
    """A function."""
    _dispatch = Dispatcher(in_class=Self)

    def stretch(self, *stretches):
        """Stretch the function.

        Args:
            *stretches (tensor): Per input, extent to stretch by.

        Returns:
            :class:`.function.Function`: Stretched function.
        """
        return stretch(self, *stretches)

    def __gt__(self, stretch):
        """Shorthand for :meth:`.function.Function.stretch`."""
        return self.stretch(stretch)

    def shift(self, *amounts):
        """Shift the inputs of an function by a certain amount.

        Args:
            *amounts (tensor): Per input, amount to shift by.

        Returns:
            :class:`.function.Function`: Shifted function.
        """
        return shift(self, *amounts)

    def select(self, *dims):
        """Select particular dimensions of the input features.

        Args:
            *dims (int, sequence, or None): Per input, dimensions to select.
                Set to `None` to select all.

        Returns:
            :class:`.function.Function`: Function with dimensions of the
                input features selected.
        """
        return select(self, *dims)

    def transform(self, *fs):
        """Transform the inputs of a function.

        Args:
            *fs (function or None): Per input, transformation. Set to `None` to
                not perform a transformation.

        Returns:
            :class:`.function.Function`: Function with its inputs
                transformed.
        """
        return transform(self, *fs)

    def diff(self, *derivs):
        """Differentiate a function.

        Args:
            *derivs (int): Per input, dimension of the feature which to take
                the derivatives with respect to. Set to `None` to not take a
                derivative.

        Returns:
            :class:`.function.Function`: Derivative of the Function.
        """
        return differentiate(self, *derivs)

    def __reversed__(self):
        """Reverse the arguments of a function.

        Returns:
            :class:`.function.Function`: Function with arguments reversed.
        """
        return reverse(self)


# Register the ring.
@_dispatch(Function)
def get_ring(a):
    return Function


class OneFunction(Function, One):
    """The constant function `1`."""


class ZeroFunction(Function, Zero):
    """The constant function `0`."""


class WrappedFunction(Function, Wrapped):
    """A wrapped function."""


class ScaledFunction(Function, Scaled):
    """A scaled function."""


class JoinFunction(Function, Join):
    """Two wrapped functions."""


class SumFunction(Function, Sum):
    """A sum of two functions."""


class ProductFunction(Function, Product):
    """A product of two functions."""


class StretchedFunction(WrappedFunction):
    """Stretched function.

    Args:
        e (:class:`.function.Function`): Function to stretch.
        *stretches (tensor): Extent of stretches.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e, *stretches):
        WrappedFunction.__init__(self, e)
        self.stretches = tuple(to_tensor(x) for x in stretches)

    def render_wrap(self, e, formatter):
        stretches = tuple(formatter(s) for s in self.stretches)
        return '{} > {}'.format(e, squeeze(stretches))

    @_dispatch(Self)
    def __eq__(self, other):
        return self[0] == other[0] and \
               tuple_equal(self.stretches, other.stretches)


class ShiftedFunction(WrappedFunction):
    """Shifted function.

    Args:
        e (:class:`.function.Function`): Function to shift.
        *shifts (tensor): Shift amounts.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e, *shifts):
        WrappedFunction.__init__(self, e)
        self.shifts = tuple(to_tensor(x) for x in shifts)

    def render_wrap(self, e, formatter):
        shifts = tuple(formatter(s) for s in self.shifts)
        return '{} shift {}'.format(e, squeeze(shifts))

    @_dispatch(Self)
    def __eq__(self, other):
        return self[0] == other[0] and \
               tuple_equal(self.shifts, other.shifts)


def _to_list(x):
    if B.rank(x) == 0:
        return [x]
    elif B.rank(x) == 1:
        return x
    else:
        raise ValueError(f'Could not convert "{x}" to a list.')


class SelectedFunction(WrappedFunction):
    """Select particular dimensions of the input features.

    Args:
        e (:class:`.function.Function`): Function to wrap.
        *dims (tensor): Dimensions to select.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e, *dims):
        WrappedFunction.__init__(self, e)
        self.dims = tuple(None if x is None else _to_list(x) for x in dims)

    def render_wrap(self, e, formatter):
        return '{} : {}'.format(e, squeeze(tuple(self.dims)))

    @_dispatch(Self)
    def __eq__(self, other):
        return self[0] == other[0] and \
               tuple_equal(self.dims, other.dims)


class InputTransformedFunction(WrappedFunction):
    """Transform inputs of a function.

    Args:
        e (:class:`.function.Function`): Function to transform inputs of.
        *fs (tensor): Per input, the transformation. Set to `None` to not
            do a transformation.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e, *fs):
        WrappedFunction.__init__(self, e)
        self.fs = fs

    def render_wrap(self, e, formatter):
        # Safely get a function's name.
        def name(f):
            return 'None' if f is None else f.__name__

        if len(self.fs) == 1:
            fs = name(self.fs[0])
        else:
            fs = '({})'.format(', '.join(name(f) for f in self.fs))
        return '{} transform {}'.format(e, fs)

    @_dispatch(Self)
    def __eq__(self, other):
        return self[0] == other[0] and \
               tuple_equal(self.fs, other.fs)


class DerivativeFunction(WrappedFunction):
    """Compute the derivative of a function.

    Args:
        e (:class:`.function.Function`): Function to compute the
            derivative of.
        *derivs (tensor): Per input, the index of the dimension which to
            take the derivative of. Set to `None` to not take a derivative.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e, *derivs):
        WrappedFunction.__init__(self, e)
        self.derivs = derivs

    def render_wrap(self, e, formatter):
        if len(self.derivs) == 1:
            derivs = '({})'.format(self.derivs[0])
        else:
            derivs = self.derivs
        return 'd{} {}'.format(derivs, e)

    @_dispatch(Self)
    def __eq__(self, other):
        return self[0] == other[0] and \
               tuple_equal(self.derivs, other.derivs)


class ReversedFunction(WrappedFunction):
    """Function with arguments reversed.

    Args:
        e (:class:`.function.Function`): Function to reverse arguments of.
    """
    _dispatch = Dispatcher(in_class=Self)

    def render_wrap(self, e, formatter):
        return f'Reversed({e})'

    @_dispatch(Self)
    def __eq__(self, other):
        return self[0] == other[0]


# A reversed function will never need parentheses.

@_dispatch(Function, ReversedFunction, precedence=definite)
def need_parens(el, parent):
    return False


# Careful to break ambiguity with previous method.
@_dispatch(ReversedFunction, Function, precedence=definite + 1)
def need_parens(el, parent):
    return False


class TensorProductFunction(Function):
    """An element built from a product of functions for each input.

    Args:
        *fs (function): Per input, a function.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, *fs):
        self.fs = fs

    def render(self, formatter):
        if len(self.fs) == 1:
            return self.fs[0].__name__
        else:
            return '{}'.format(' x '.join(f.__name__ for f in self.fs))

    @_dispatch(Self)
    def __eq__(self, other):
        return tuple_equal(self.fs, other.fs)


# A tensor product function needs parentheses if and only if it has more than
# one function.

@_dispatch(TensorProductFunction, Element, precedence=definite)
def need_parens(el, parent):
    return len(el.fs) > 1


@_dispatch(object, [object])
def stretch(a, *stretches):
    """Stretch a function.

    Args:
        a (:class:`.function.Function`): Function to stretch.
        *stretches (tensor): Per input, extent of stretches.

    Returns:
        :class:`.function.Function`: Stretched function.
    """
    raise NotImplementedError('Stretching not implemented for "{}".'
                              ''.format(type(a).__name__))


@_dispatch(Function, [object])
def stretch(a, *stretches):
    return new(a, StretchedFunction)(a, *stretches)


@_dispatch({ZeroFunction, OneFunction}, [object])
def stretch(a, *stretches):
    return a


@_dispatch(StretchedFunction, [object])
def stretch(a, *stretches):
    return stretch(a[0], *broadcast(operator.mul, a.stretches, stretches))


@_dispatch(object, [object])
def shift(a, *shifts):
    """Shift a function.

    Args:
        a (:class:`.function.Function`): Function to shift.
        *shifts (tensor): Per input, amount of shift.

    Returns:
        :class:`.function.Function`: Shifted element.
    """
    raise NotImplementedError('Shifting not implemented for "{}".'
                              ''.format(type(a).__name__))


@_dispatch(Function, [object])
def shift(a, *shifts):
    return new(a, ShiftedFunction)(a, *shifts)


@_dispatch({ZeroFunction, OneFunction}, [object])
def shift(a, *shifts):
    return a


@_dispatch(ShiftedFunction, [object])
def shift(a, *shifts):
    return shift(a[0], *broadcast(operator.add, a.shifts, shifts))


@_dispatch(object, [object])
def select(a, *dims):
    """Select dimensions from the inputs.

    Args:
        a (:class:`.function.Function`): Function  to wrap.
        *dims (int): Per input, dimensions to select. Set to `None` to select
            all.

    Returns:
        :class:`.function.Function`: Function with particular dimensions
            from the inputs selected.
    """
    raise NotImplementedError('Selection not implemented for "{}".'
                              ''.format(type(a).__name__))


@_dispatch(Function, [object])
def select(a, *dims):
    return new(a, SelectedFunction)(a, *dims)


@_dispatch({ZeroFunction, OneFunction}, [object])
def select(a, *dims):
    return a


@_dispatch(object, [object])
def transform(a, *fs):
    """Transform the inputs of a function.

    Args:
        a (:class:`.function.Function`): Function to wrap.
        *fs (int): Per input, the transform. Set to `None` to not perform a
            transform.

    Returns:
        :class:`.function.Function`: Function with its inputs
            transformed.
    """
    raise NotImplementedError('Input transforms not implemented for "{}".'
                              ''.format(type(a).__name__))


@_dispatch(Function, [object])
def transform(a, *fs):
    return new(a, InputTransformedFunction)(a, *fs)


@_dispatch({ZeroFunction, OneFunction}, [object])
def transform(a, *fs):
    return a


@_dispatch(object, [object])
def differentiate(a, *derivs):
    """Differentiate a function.

    Args:
        a (:class:`.function.Function`): Function to differentiate.
        *derivs (int): Per input, dimension of the feature which to take
            the derivatives with respect to. Set to `None` to not take a
            derivative.

    Returns:
        :class:`.function.Function`: Derivative of the function.
    """
    raise NotImplementedError('Differentiation not implemented for "{}".'
                              ''.format(type(a).__name__))


@_dispatch(Function, [object])
def differentiate(a, *derivs):
    return new(a, DerivativeFunction)(a, *derivs)


@_dispatch(ZeroFunction, [object])
def differentiate(a, *derivs):
    return a


@_dispatch(OneFunction, [object])
def differentiate(a, *derivs):
    return new(a, Zero)()


@_dispatch(object)
def reverse(a):
    """Reverse argument of a function.

    Args:
        a (:class:`.function.Function`): Function to reverse arguments of.

    Returns:
        :class:`.function.Function`: Function with arguments reversed.
    """
    raise NotImplementedError('Argument reversal not implemented for "{}".'
                              ''.format(type(a).__name__))


@_dispatch(Function)
def reverse(a):
    return new(a, ReversedFunction)(a)


@_dispatch({ZeroFunction, OneFunction})
def reverse(a):
    return a


# Propagate reversal.

@_dispatch(SumFunction)
def reverse(a):
    return add(reverse(a[0]), reverse(a[1]))


@_dispatch(ProductFunction)
def reverse(a):
    return mul(reverse(a[0]), reverse(a[1]))


@_dispatch(ScaledFunction)
def reverse(a):
    return mul(a.scale, reverse(a[0]))


# Let reversal synergise with wrapped kernels.

@_dispatch(ReversedFunction)
def reverse(a):
    return a[0]


@_dispatch(StretchedFunction)
def reverse(a):
    return stretch(reverse(a[0]), *reversed(a.stretches))


@_dispatch(ShiftedFunction)
def reverse(a):
    return shift(reverse(a[0]), *reversed(a.shifts))


@_dispatch(SelectedFunction)
def reverse(a):
    return select(reverse(a[0]), *reversed(a.dims))


@_dispatch(InputTransformedFunction)
def reverse(a):
    return transform(reverse(a[0]), *reversed(a.fs))


@_dispatch(DerivativeFunction)
def reverse(a):
    return differentiate(reverse(a[0]), *reversed(a.derivs))


@_dispatch(TensorProductFunction)
def reverse(a):
    return TensorProductFunction(*reversed(a.fs))


# Handle conversion of Python functions.

@mul.extend(Element, PythonFunction, precedence=priority)
def mul(a, b):
    return mul(a, new(a, TensorProductFunction)(b))


@mul.extend(PythonFunction, Element, precedence=priority)
def mul(a, b):
    return mul(new(b, TensorProductFunction)(a), b)


@add.extend(Element, PythonFunction, precedence=priority)
def add(a, b):
    return add(a, new(a, TensorProductFunction)(b))


@add.extend(PythonFunction, Element, precedence=priority)
def add(a, b):
    return add(new(b, TensorProductFunction)(a), b)
