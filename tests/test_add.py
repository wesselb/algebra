from algebra import One, Zero

from .util import a, b


def test_add_zero():
    assert str(0 + a) == "a"
    assert str(a + 0) == "a"

    assert str(0 + Zero()) == "0"
    assert str(Zero() + 0) == "0"

    assert str(0 + One()) == "1"
    assert str(One() + 0) == "1"


def test_add_zero_object():
    assert str(Zero() + a) == "a"
    assert str(a + Zero()) == "a"

    assert str(Zero() + Zero()) == "0"

    assert str(Zero() + One()) == "1"
    assert str(One() + Zero()) == "1"


def test_add_one():
    assert str(1 + a) == "1 + a"
    assert str(a + 1) == "a + 1"

    assert str(1 + Zero()) == "1"
    assert str(Zero() + 1) == "1"

    assert str(1 + One()) == "2 * 1"
    assert str(One() + 1) == "2 * 1"


def test_add_one_object():
    assert str(One() + a) == "1 + a"
    assert str(a + One()) == "a + 1"

    assert str(One() + Zero()) == "1"
    assert str(Zero() + One()) == "1"

    assert str(One() + One()) == "2 * 1"
    assert str(One() + One()) == "2 * 1"


def test_add_two():
    assert str(2 + a) == "2 * 1 + a"
    assert str(a + 2) == "a + 2 * 1"

    assert str(2 + Zero()) == "2 * 1"
    assert str(Zero() + 2) == "2 * 1"

    assert str(2 + One()) == "3 * 1"
    assert str(One() + 2) == "3 * 1"


def test_add_a():
    assert str(a + a) == "2 * a"

    assert str(a + Zero()) == "a"
    assert str(Zero() + a) == "a"

    assert str(a + One()) == "a + 1"
    assert str(One() + a) == "1 + a"


def test_add_b():
    assert str(b + a) == "b + a"
    assert str(a + b) == "a + b"

    assert str(b + Zero()) == "b"
    assert str(Zero() + b) == "b"

    assert str(b + One()) == "b + 1"
    assert str(One() + b) == "1 + b"


def test_grouping():
    assert str(2 * a + b) == "2 * a + b"
    assert str(b + 2 * a) == "b + 2 * a"

    assert str(2 * a + a) == "3 * a"
    assert str(a + 2 * a) == "3 * a"

    assert str(2 * a + 2 * a) == "4 * a"
    assert str(2 * a + 2 * b) == "2 * a + 2 * b"
