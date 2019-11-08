from ring import One, Zero

from .util import A, B


def test_mul_zero():
    assert str(0 * A()) == '0'
    assert str(A() * 0) == '0'

    assert str(0 * Zero()) == '0'
    assert str(Zero() * 0) == '0'

    assert str(0 * One()) == '0'
    assert str(One() * 0) == '0'


def test_mul_zero_object():
    assert str(Zero() * A()) == '0'
    assert str(A() * Zero()) == '0'

    assert str(Zero() * Zero()) == '0'
    assert str(Zero() * Zero()) == '0'

    assert str(Zero() * One()) == '0'
    assert str(One() * Zero()) == '0'


def test_mul_one():
    assert str(1 * A()) == 'A()'
    assert str(A() * 1) == 'A()'

    assert str(1 * Zero()) == '0'
    assert str(Zero() * 1) == '0'

    assert str(1 * One()) == '1'
    assert str(One() * 1) == '1'


def test_mul_one_object():
    assert str(One() * A()) == 'A()'
    assert str(A() * One()) == 'A()'

    assert str(One() * Zero()) == '0'
    assert str(Zero() * One()) == '0'

    assert str(One() * One()) == '1'
    assert str(One() * One()) == '1'


def test_mul_two():
    assert str(2 * A()) == '2 * A()'
    assert str(A() * 2) == '2 * A()'

    assert str(2 * Zero()) == '0'
    assert str(Zero() * 2) == '0'

    assert str(2 * One()) == '2 * 1'
    assert str(One() * 2) == '2 * 1'


def test_mul_a():
    assert str(A() * A()) == 'A() * A()'

    assert str(A() * Zero()) == '0'
    assert str(Zero() * A()) == '0'

    assert str(A() * One()) == 'A()'
    assert str(One() * A()) == 'A()'


def test_mul_b():
    assert str(B() * A()) == 'B() * A()'
    assert str(A() * B()) == 'A() * B()'

    assert str(B() * Zero()) == '0'
    assert str(Zero() * B()) == '0'

    assert str(B() * One()) == 'B()'
    assert str(One() * B()) == 'B()'


def test_grouping():
    assert str(2 * (2 * A())) == '4 * A()'
    assert str((2 * A()) * 2) == '4 * A()'

    assert str(A() * (2 * A())) == '2 * A() * A()'
    assert str((2 * A()) * A()) == '2 * A() * A()'

    assert str(A() * (2 * B())) == '2 * A() * B()'
    assert str((2 * B()) * A()) == '2 * B() * A()'

    assert str((2 * A()) * (2 * A())) == '4 * A() * A()'
    assert str((2 * A()) * (2 * B())) == '4 * A() * B()'
