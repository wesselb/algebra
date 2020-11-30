from algebra import One, Zero

from .util import a, b


def test_mul_zero():
    assert str(0 * a) == "0"
    assert str(a * 0) == "0"

    assert str(0 * Zero()) == "0"
    assert str(Zero() * 0) == "0"

    assert str(0 * One()) == "0"
    assert str(One() * 0) == "0"


def test_mul_zero_object():
    assert str(Zero() * a) == "0"
    assert str(a * Zero()) == "0"

    assert str(Zero() * Zero()) == "0"
    assert str(Zero() * Zero()) == "0"

    assert str(Zero() * One()) == "0"
    assert str(One() * Zero()) == "0"


def test_mul_one():
    assert str(1 * a) == "a"
    assert str(a * 1) == "a"

    assert str(1 * Zero()) == "0"
    assert str(Zero() * 1) == "0"

    assert str(1 * One()) == "1"
    assert str(One() * 1) == "1"


def test_mul_one_object():
    assert str(One() * a) == "a"
    assert str(a * One()) == "a"

    assert str(One() * Zero()) == "0"
    assert str(Zero() * One()) == "0"

    assert str(One() * One()) == "1"
    assert str(One() * One()) == "1"


def test_mul_two():
    assert str(2 * a) == "2 * a"
    assert str(a * 2) == "2 * a"

    assert str(2 * Zero()) == "0"
    assert str(Zero() * 2) == "0"

    assert str(2 * One()) == "2 * 1"
    assert str(One() * 2) == "2 * 1"


def test_mul_a():
    assert str(a * a) == "a * a"

    assert str(a * Zero()) == "0"
    assert str(Zero() * a) == "0"

    assert str(a * One()) == "a"
    assert str(One() * a) == "a"


def test_mul_b():
    assert str(b * a) == "b * a"
    assert str(a * b) == "a * b"

    assert str(b * Zero()) == "0"
    assert str(Zero() * b) == "0"

    assert str(b * One()) == "b"
    assert str(One() * b) == "b"


def test_grouping():
    assert str(2 * (2 * a)) == "4 * a"
    assert str((2 * a) * 2) == "4 * a"

    assert str(a * (2 * a)) == "2 * a * a"
    assert str((2 * a) * a) == "2 * a * a"

    assert str(a * (2 * b)) == "2 * a * b"
    assert str((2 * b) * a) == "2 * b * a"

    assert str((2 * a) * (2 * a)) == "4 * a * a"
    assert str((2 * a) * (2 * b)) == "4 * a * b"
