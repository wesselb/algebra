from abc import ABCMeta, abstractmethod

from plum import Dispatcher, Referentiable, Self
from lab import B

from . import _dispatch
from .util import get_subclasses

__all__ = ['Formatter',

           # Precedence levels:
           'priority',
           'definite',

           # Ring elements:
           'Element',
           'One',
           'Zero',
           'Wrapped',
           'Join',
           'Scaled',
           'Product',
           'Sum',

           # Ring functions:
           'pretty_print',
           'add',
           'mul',
           'get_ring',
           'new']

Formatter = object  #: A formatter can be any object.
priority = 10  #: Priority precedence level.
definite = 20  #: Highest precedence level.


class Element(metaclass=Referentiable(ABCMeta)):
    """An element in a ring.

    Elements can be added and multiplied.
    """

    _dispatch = Dispatcher(in_class=Self)

    def __eq__(self, other):
        return False

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __neg__(self):
        return mul(-1, self)

    def __sub__(self, other):
        return add(self, -other)

    def __rsub__(self, other):
        return add(other, -self)

    @_dispatch(int)
    def __pow__(self, power, modulo=None):
        if power < 0:
            raise ValueError('Cannot raise to a negative power.')
        elif power == 0:
            return 1
        else:
            k = self
            for _ in range(power - 1):
                k *= self
        return k

    @property
    def num_terms(self):
        """Number of terms"""
        return 1

    def term(self, i):
        """Get a specific term.

        Args:
            i (int): Index of term.

        Returns:
            :class:`.ring.Element`: The referenced term.
        """
        if i == 0:
            return self
        else:
            raise IndexError('Index out of range.')

    @property
    def num_factors(self):
        """Number of factors"""
        return 1

    def factor(self, i):
        """Get a specific factor.

        Args:
            i (int): Index of factor.

        Returns:
            :class:`.ring.Element`: The referenced factor.
        """
        if i == 0:
            return self
        else:
            raise IndexError('Index out of range.')

    @property
    def __name__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.display()

    def __str__(self):
        return self.display()

    @_dispatch(Formatter)
    def display(self, formatter):
        """Display the element.

        Args:
            formatter (function, optional): Function to format values.

        Returns:
            str: Element as a string.
        """
        return pretty_print(self, formatter)

    @_dispatch()
    def display(self):
        return self.display(lambda x: x)

    def render(self, formatter):
        """Render the element.

        This is the lowest-level operation in  pretty printing an element,
        and should produce a string representation of the element. This
        method should be implemented to determine how to render a custom
        element.

        Args:
            formatter (function, optional): Function to format values.

        Returns:
            str: Rendering of the element.
        """
        return f'{self.__name__}()'


class One(Element):
    """The constant `1`."""
    _dispatch = Dispatcher(in_class=Self)

    def render(self, formatter):
        return '1'

    @_dispatch(Self)
    def __eq__(self, other):
        return True


class Zero(Element):
    """The constant `0`."""
    _dispatch = Dispatcher(in_class=Self)

    def render(self, formatter):
        return '0'

    @_dispatch(Self)
    def __eq__(self, other):
        return True


class Wrapped(Element):
    """A wrapped element.

    Args:
        e (:class:`.ring.Element`): Element to wrap.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e):
        self.e = e

    def __getitem__(self, item):
        if item == 0:
            return self.e
        else:
            raise IndexError('Index out of range.')

    @abstractmethod
    def render_wrap(self, e, formatter):  # pragma: no cover
        pass


class Join(Element):
    """Two wrapped elements.

    Args:
        e1 (:class:`.ring.Element`): First element to wrap.
        e2 (:class:`.ring.Element`): Second element to wrap.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e1, e2):
        self.e1 = e1
        self.e2 = e2

    def __getitem__(self, item):
        if item == 0:
            return self.e1
        elif item == 1:
            return self.e2
        else:
            raise IndexError('Index out of range.')

    @abstractmethod
    def render_join(self, e1, e2, formatter):  # pragma: no cover
        pass


class Scaled(Wrapped):
    """Scaled element.

    Args:
        e (:class:`.ring.Element`): Element to scale.
        scale (tensor): Scale.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, e, scale):
        Wrapped.__init__(self, e)
        self.scale = scale

    @property
    def num_factors(self):
        return self[0].num_factors + 1

    def render_wrap(self, e, formatter):
        return f'{formatter(self.scale)} * {e}'

    def factor(self, i):
        if i >= self.num_factors:
            raise IndexError('Index out of range.')
        else:
            return self.scale if i == 0 else self[0].factor(i - 1)

    @_dispatch(Self)
    def __eq__(self, other):
        return self[0] == other[0] and B.all(self.scale == other.scale)


class Product(Join):
    """Product of elements."""
    _dispatch = Dispatcher(in_class=Self)

    @property
    def num_factors(self):
        return self[0].num_factors + self[1].num_factors

    def factor(self, i):
        if i >= self.num_factors:
            raise IndexError('Index out of range.')
        if i < self[0].num_factors:
            return self[0].factor(i)
        else:
            return self[1].factor(i - self[0].num_factors)

    def render_join(self, e1, e2, formatter):
        return f'{e1} * {e2}'

    @_dispatch(Self)
    def __eq__(self, other):
        return (self[0] == other[0] and self[1] == other[1]) or \
               (self[0] == other[1] and self[1] == other[0])


class Sum(Join):
    """Sum of elements."""
    _dispatch = Dispatcher(in_class=Self)

    @property
    def num_terms(self):
        return self[0].num_terms + self[1].num_terms

    def term(self, i):
        if i >= self.num_terms:
            raise IndexError('Index out of range.')
        if i < self[0].num_terms:
            return self[0].term(i)
        else:
            return self[1].term(i - self[0].num_terms)

    def render_join(self, e1, e2, formatter):
        return f'{e1} + {e2}'

    @_dispatch(Self)
    def __eq__(self, other):
        return (self[0] == other[0] and self[1] == other[1]) or \
               (self[0] == other[1] and self[1] == other[0])


@_dispatch(Element, Formatter)
def pretty_print(el, formatter):
    """Pretty print an element with a minimal number of parentheses.

    Args:
        el (:class:`.field.Element`): Element to print.
        formatter (:class:`.field.Formatter`): Formatter for values.

    Returns:
        str: `el` converted to string prettily.
    """
    return el.render(formatter)


@_dispatch(object, object)
def add(a, b):
    """Add two elements.

    Args:
        a (:class:`.ring.Element`): First element in addition.
        b (:class:`.ring.Element`): Second element in addition.

    Returns:
        :class:`.ring.Element`: Sum of the elements.
    """
    raise NotImplementedError(f'Addition not implemented for '
                              f'"{type(a).__name__}" and "{type(b).__name__}".')


@_dispatch(object, object)
def mul(a, b):
    """Multiply two elements.

    Args:
        a (:class:`.field.Element`): First element in product.
        b (:class:`.field.Element`): Second element in product.

    Returns:
        :class:`.field.Element`: Product of the elements.
    """
    raise NotImplementedError(f'Multiplication not implemented for '
                              f'"{type(a).__name__}" and "{type(b).__name__}".')


@_dispatch(object)
def get_ring(a):
    """Get the ring of an element.

    Args:
        a (:class:`.ring.Element`): Element to get ring of.

    Returns:
        type: Ring of `a`.
    """
    raise RuntimeError(f'Could not determine ring type of '
                       f'"{type(a).__name__}".')


# Register the default ring.
@_dispatch(Element)
def get_ring(a):
    return Element


new_cache = {}  #: Cache for `.ring.new`.


def new(a, t):
    """Create a new specialised type.

    Args:
        a (:class:`.ring.Element`): Element to create new type for.
        t (type): Type to create.

    Returns:
        type: Specialisation of `t` appropriate for `a`.
    """
    try:
        return new_cache[type(a), t]
    except KeyError:
        ring = get_ring(a)

        # Determine candidates.
        ring_types = set(get_subclasses(ring))
        element_types = {t} | set(get_subclasses(t))
        candidates = list(ring_types & element_types)

        # The most specific types are the ones we are looking for.
        candidates = filter_most_specific(candidates)

        # There should only be a single candidate.
        if len(candidates) != 1:
            raise RuntimeError(f'Could not determine "{t.__name__}" for ring '
                               f'"{ring.__name__}".')

        new_cache[type(a), t] = candidates[0]
        return new_cache[type(a), t]


def filter_most_specific(types):
    """From a list of types, determine the most specific ones.

    Args:
        types (list[type]): List of types.

    Returns:
        list[type]: Most specific types in `types`.
    """
    filtered_types = []

    while len(types) > 0:
        t, types = types[0], types[1:]

        # If `t` is a supertype, discard it. Otherwise, keep it.
        if not (
                any(issubclass(u, t) for u in types) or
                any(issubclass(u, t) for u in filtered_types)
        ):
            filtered_types.append(t)

    return filtered_types
