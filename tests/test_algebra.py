import pytest
from plum import NotFoundLookupError

from algebra import Element, One, Zero, Scaled, Product, Sum, add, mul, get_algebra, new
from algebra.algebra import filter_most_specific
from .util import a, b, c


def test_equality_element():
    e1 = Element()
    e2 = Element()
    assert e1 == e1
    assert e1 != e2


def test_equality_one_zero():
    assert One() == One()
    assert One() != Zero()
    assert Zero() == Zero()


def test_equality_scaled():
    assert Scaled(One(), 1) == Scaled(One(), 1)
    assert Scaled(One(), 2) != Scaled(One(), 1)
    assert Scaled(Zero(), 1) != Scaled(One(), 1)


def test_equality_product():
    assert Product(One(), Zero()) == Product(One(), Zero())
    assert Product(One(), Zero()) == Product(Zero(), One())
    assert Product(One(), Zero()) != Product(One(), One())


def test_equality_sum():
    assert Sum(One(), Zero()) == Sum(One(), Zero())
    assert Sum(One(), Zero()) == Sum(Zero(), One())
    assert Sum(One(), Zero()) != Sum(One(), One())


def test_addition():
    assert str(a + 1) == "a + 1"
    assert str(1 + a) == "1 + a"


def test_subtraction():
    assert str(1 - a) == "1 + -1 * a"
    assert str(a - 1) == "a + -1 * 1"
    assert str(a - a) == "0"
    assert str(b - a) == "b + -1 * a"


def test_negation():
    assert str(-a) == "-1 * a"


def test_name():
    assert a.__name__ == "A"


def test_str():
    assert str(1 - a) == "1 + -1 * a"


def test_repr():
    assert repr(1 - a) == "1 + -1 * a"


def test_power():
    with pytest.raises(ValueError):
        a ** -1
    with pytest.raises(NotFoundLookupError):
        a ** 0.5
    assert str(a ** 0) == "1"
    assert str(a ** 1) == "a"
    assert str(a ** 2) == "a * a"
    assert str(a ** 3) == "a * a * a"


def test_terms():
    e = a + a * b + c * c + b
    assert e.num_terms == 4
    assert str(e.term(0)) == "a"
    assert str(e.term(1)) == "a * b"
    assert str(e.term(2)) == "c * c"
    assert str(e.term(3)) == "b"
    with pytest.raises(IndexError):
        e.term(4)
    with pytest.raises(IndexError):
        a.term(1)


def test_factors():
    e = a * b
    assert e.num_factors == 2
    assert str(e.factor(0)) == "a"
    assert str(e.factor(1)) == "b"
    with pytest.raises(IndexError):
        e.factor(2)

    e = (a + a) * c * (b + c)
    assert e.num_factors == 4
    assert str(e.factor(0)) == "2"
    assert str(e.factor(1)) == "a"
    assert str(e.factor(2)) == "c"
    assert str(e.factor(3)) == "b + c"
    with pytest.raises(IndexError):
        e.factor(4)
    with pytest.raises(IndexError):
        a.factor(1)


def test_indexing_wrapped():
    e = 5 * a
    assert str(e[0]) == "a"
    with pytest.raises(IndexError):
        e[1]


def test_indexing_sum():
    e = a + b
    assert str(e[0]) == "a"
    assert str(e[1]) == "b"
    with pytest.raises(IndexError):
        e[2]


def test_indexing_product():
    e = a * b
    assert str(e[0]) == "a"
    assert str(e[1]) == "b"
    with pytest.raises(IndexError):
        e[2]


def test_display_formatter():
    assert (3 * (Element() + 4)).display(lambda x: x ** 2) == "9 * (Element() + 16 * 1)"


def test_add_fallback():
    with pytest.raises(RuntimeError):
        add("1", "2")


def test_mul_fallback():
    with pytest.raises(RuntimeError):
        mul("1", "2")


def test_get_algebra():
    for x in [Element(), One(), Zero()]:
        assert get_algebra(x) == Element

    with pytest.raises(RuntimeError):
        get_algebra(1)


def test_new():
    class Kernel(Element):
        pass

    class SumKernel(Kernel, Sum):
        pass

    @get_algebra.dispatch
    def _get_algebra(e: Kernel):
        return Kernel

    # Test that the algebra is correctly registered.
    assert get_algebra(SumKernel(Kernel(), Kernel())) == Kernel

    # Test the creation of a sum type, which should succeed.
    assert new(Kernel(), Sum) == SumKernel

    # Test the creation of a sum type, which should fail.
    with pytest.raises(RuntimeError):
        new(Kernel(), Product)


def test_filter_most_specific():
    class T1:
        pass

    class T2(T1):
        pass

    assert set(filter_most_specific([int, str])) == {int, str}
    assert set(filter_most_specific([int, str, object])) == {int, str}
    assert set(filter_most_specific([int, object, str])) == {int, str}
    assert set(filter_most_specific([T1, T2])) == {T2}
    assert set(filter_most_specific([T2, T1])) == {T2}
    assert set(filter_most_specific([int, T1, object, str, T2])) == {int, str, T2}
