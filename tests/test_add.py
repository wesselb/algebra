from ring import One, Zero

from .util import A, B


def test_add_zero():
    assert str(0 + A()) == 'A()'
    assert str(A() + 0) == 'A()'

    assert str(0 + Zero()) == '0'
    assert str(Zero() + 0) == '0'

    assert str(0 + One()) == '1'
    assert str(One() + 0) == '1'


def test_add_zero_object():
    assert str(Zero() + A()) == 'A()'
    assert str(A() + Zero()) == 'A()'

    assert str(Zero() + Zero()) == '0'

    assert str(Zero() + One()) == '1'
    assert str(One() + Zero()) == '1'


def test_add_one():
    assert str(1 + A()) == '1 + A()'
    assert str(A() + 1) == 'A() + 1'

    assert str(1 + Zero()) == '1'
    assert str(Zero() + 1) == '1'

    assert str(1 + One()) == '2 * 1'
    assert str(One() + 1) == '2 * 1'


def test_add_one_object():
    assert str(One() + A()) == '1 + A()'
    assert str(A() + One()) == 'A() + 1'

    assert str(One() + Zero()) == '1'
    assert str(Zero() + One()) == '1'

    assert str(One() + One()) == '2 * 1'
    assert str(One() + One()) == '2 * 1'


def test_add_two():
    assert str(2 + A()) == '2 * 1 + A()'
    assert str(A() + 2) == 'A() + 2 * 1'

    assert str(2 + Zero()) == '2 * 1'
    assert str(Zero() + 2) == '2 * 1'

    assert str(2 + One()) == '3 * 1'
    assert str(One() + 2) == '3 * 1'


def test_add_a():
    assert str(A() + A()) == '2 * A()'

    assert str(A() + Zero()) == 'A()'
    assert str(Zero() + A()) == 'A()'

    assert str(A() + One()) == 'A() + 1'
    assert str(One() + A()) == '1 + A()'


def test_add_b():
    assert str(B() + A()) == 'B() + A()'
    assert str(A() + B()) == 'A() + B()'

    assert str(B() + Zero()) == 'B()'
    assert str(Zero() + B()) == 'B()'

    assert str(B() + One()) == 'B() + 1'
    assert str(One() + B()) == '1 + B()'


def test_grouping():
    assert str(2 * A() + B()) == '2 * A() + B()'
    assert str(B() + 2 * A()) == 'B() + 2 * A()'

    assert str(2 * A() + A()) == '3 * A()'
    assert str(A() + 2 * A()) == '3 * A()'

    assert str(2 * A() + 2 * A()) == '4 * A()'
    assert str(2 * A() + 2 * B()) == '2 * A() + 2 * B()'
