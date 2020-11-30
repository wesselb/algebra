from numpy.testing import assert_allclose, assert_array_almost_equal
from plum import Dispatcher, Self

from algebra import Element

__all__ = ["allclose", "approx", "a", "b", "c"]

allclose = assert_allclose
approx = assert_array_almost_equal


# Some extra atomic elements to test with:


class A(Element):
    dispatch = Dispatcher(in_class=Self)

    @dispatch(Self)
    def __eq__(self, other):
        return True

    def render(self, formatter):
        return "a"


a = A()


class B(Element):
    dispatch = Dispatcher(in_class=Self)

    @dispatch(Self)
    def __eq__(self, other):
        return True

    def render(self, formatter):
        return "b"


b = B()


class C(Element):
    dispatch = Dispatcher(in_class=Self)

    @dispatch(Self)
    def __eq__(self, other):
        return True

    def render(self, formatter):
        return "c"


c = C()
