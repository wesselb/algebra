from numpy.testing import assert_allclose
from plum import dispatch

from algebra import Element

__all__ = ["approx", "a", "b", "c"]

approx = assert_allclose


# Some extra atomic elements to test with:


class A(Element):
    @dispatch
    def __eq__(self, other: "A"):
        return True

    def render(self, formatter):
        return "a"


a = A()


class B(Element):
    @dispatch
    def __eq__(self, other: "B"):
        return True

    def render(self, formatter):
        return "b"


b = B()


class C(Element):
    @dispatch
    def __eq__(self, other: "C"):
        return True

    def render(self, formatter):
        return "c"


c = C()
