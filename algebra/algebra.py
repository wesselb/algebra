from abc import ABCMeta, abstractmethod

from . import _dispatch
from .util import get_subclasses

__all__ = [
    "proven",
    "Element",
    "One",
    "Zero",
    "Wrapped",
    "Join",
    "pretty_print",
    "add",
    "mul",
    "get_algebra",
    "new",
]

_proven_level = 10  #: Current precedence level for proven methods.


def proven():
    """Generate a method precedence level for proven methods. Proven methods
    should be such that any applicable one gives the same result, and in case
    of ambiguity no particular proven method is preferred.

    Returns:
        int: Precedence level.
    """
    global _proven_level
    _proven_level += 1
    return _proven_level


class Element(metaclass=ABCMeta):
    """An element in a algebra.

    Elements can be added and multiplied.
    """

    def __eq__(self, other):
        return self is other

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

    @_dispatch
    def __pow__(self, power: int, modulo=None):
        if power < 0:
            raise ValueError("Cannot raise to a negative power.")
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
            :class:`.algebra.Element`: The referenced term.
        """
        if i == 0:
            return self
        else:
            raise IndexError("Index out of range.")

    @property
    def num_factors(self):
        """Number of factors"""
        return 1

    def factor(self, i):
        """Get a specific factor.

        Args:
            i (int): Index of factor.

        Returns:
            :class:`.algebra.Element`: The referenced factor.
        """
        if i == 0:
            return self
        else:
            raise IndexError("Index out of range.")

    @property
    def __name__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.display()

    def __str__(self):
        return self.display()

    @_dispatch
    def display(self, formatter):
        """Display the element.

        Args:
            formatter (object, optional): Function to format values.

        Returns:
            str: Element as a string.
        """
        return pretty_print(self, formatter)

    @_dispatch
    def display(self):
        return self.display(lambda x: x)

    def render(self, formatter):
        """Render the element.

        This is the lowest-level operation in pretty printing an element, and should
        produce a string representation of the element. This method should be
        implemented to determine how to render a custom element.

        Args:
            formatter (elements, optional): Function to format values.

        Returns:
            str: Rendering of the element.
        """
        return f"{self.__name__}()"


class One(Element):
    """The constant `1`."""

    def render(self, formatter):
        return "1"

    @_dispatch
    def __eq__(self, other: "One"):
        return True


class Zero(Element):
    """The constant `0`."""

    def render(self, formatter):
        return "0"

    @_dispatch
    def __eq__(self, other: "Zero"):
        return True


class Wrapped(Element):
    """A wrapped element.

    Args:
        e (:class:`.algebra.Element`): Element to wrap.
    """

    def __init__(self, e):
        self.e = e

    def __getitem__(self, item):
        if item == 0:
            return self.e
        else:
            raise IndexError("Index out of range.")

    @abstractmethod
    def render_wrap(self, e, formatter):  # pragma: no cover
        pass


class Join(Element):
    """Two wrapped elements.

    Args:
        e1 (:class:`.algebra.Element`): First element to wrap.
        e2 (:class:`.algebra.Element`): Second element to wrap.
    """

    def __init__(self, e1, e2):
        self.e1 = e1
        self.e2 = e2

    def __getitem__(self, item):
        if item == 0:
            return self.e1
        elif item == 1:
            return self.e2
        else:
            raise IndexError("Index out of range.")

    @abstractmethod
    def render_join(self, e1, e2, formatter):  # pragma: no cover
        pass


@_dispatch
def pretty_print(el: Element, formatter):
    """Pretty print an element with a minimal number of parentheses.

    Args:
        el (:class:`.algebra.Element`): Element to print.
        formatter (object): Formatter for values.

    Returns:
        str: `el` converted to string prettily.
    """
    return el.render(formatter)


@_dispatch
def add(a, b):
    """Add two elements.

    Args:
        a (:class:`.algebra.Element`): First element in addition.
        b (:class:`.algebra.Element`): Second element in addition.

    Returns:
        :class:`.algebra.Element`: Sum of the elements.
    """
    raise NotImplementedError(
        f"Addition not implemented for "
        f'"{type(a).__name__}" and "{type(b).__name__}".'
    )


@_dispatch
def mul(a, b):
    """Multiply two elements.

    Args:
        a (:class:`.algebra.Element`): First element in product.
        b (:class:`.algebra.Element`): Second element in product.

    Returns:
        :class:`.algebra.Element`: Product of the elements.
    """
    raise NotImplementedError(
        f"Multiplication not implemented for "
        f'"{type(a).__name__}" and "{type(b).__name__}".'
    )


@_dispatch
def get_algebra(a):
    """Get the algebra of an element.

    Args:
        a (:class:`.algebra.Element`): Element to get algebra of.

    Returns:
        type: Algebra of `a`.
    """
    raise RuntimeError(f'Could not determine algebra type of "{type(a).__name__}".')


# Register the default algebra.
@_dispatch
def get_algebra(a: Element):
    return Element


new_cache = {}  #: Cache for `.algebra.new`.


def new(a, t):
    """Create a new specialised type.

    Args:
        a (:class:`.algebra.Element`): Element to create new type for.
        t (type): Type to create.

    Returns:
        type: Specialisation of `t` appropriate for `a`.
    """
    try:
        return new_cache[type(a), t]
    except KeyError:
        algebra = get_algebra(a)

        # Determine candidates.
        algebra_types = set(get_subclasses(algebra))
        element_types = {t} | set(get_subclasses(t))
        candidates = list(algebra_types & element_types)

        # The most specific types are the ones we are looking for.
        candidates = filter_most_specific(candidates)

        # There should only be a single candidate.
        if len(candidates) != 1:
            raise RuntimeError(
                f'Could not determine "{t.__name__}" for algebra "{algebra.__name__}".'
            )

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
            any(issubclass(u, t) for u in types)
            or any(issubclass(u, t) for u in filtered_types)
        ):
            filtered_types.append(t)

    return filtered_types
