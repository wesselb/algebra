from ring import (
    Element,
    One,
    Zero,
    Scaled,
    Product,
    Sum
)


def test_equality_element():
    assert Element() != Element()


def test_equality_one_zero():
    assert One() == One()
    assert One() != Zero()
    assert Zero() == Zero()


def test_equality_scaled():
    assert Scaled(One(), 1) != Scaled(One(), 1)
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
