import operator

import numpy as np
import pytest

from algebra.util import squeeze, get_subclasses, broadcast, tuple_equal, to_tensor


def test_squeeze():
    assert squeeze([1, 2]) == [1, 2]
    assert squeeze([1]) == 1


def test_get_subclasses():
    class A:
        pass

    class B1(A):
        pass

    class B2(A):
        pass

    class C(B1):
        pass

    assert set(get_subclasses(A)) == {B1, B2, C}
    assert set(get_subclasses(B1)) == {C}
    assert set(get_subclasses(B2)) == set()


def test_broadcast():
    assert broadcast(operator.add, (1, 2, 3), (2, 3, 4)) == (3, 5, 7)
    assert broadcast(operator.add, (1,), (2, 3, 4)) == (3, 4, 5)
    assert broadcast(operator.add, (1, 2, 3), (2,)) == (3, 4, 5)
    with pytest.raises(ValueError):
        broadcast(operator.add, (1, 2), (1, 2, 3))


def test_to_tensor():
    assert isinstance(to_tensor(np.array([1, 2])), np.ndarray)
    assert isinstance(to_tensor([1, 2]), np.ndarray)


def test_tuple_equal():
    assert tuple_equal((1,), (1,))
    assert tuple_equal((1, [1]), (1, [1]))
    assert not tuple_equal((1,), ([1],))
    assert not tuple_equal(([1],), ([1], [1]))
    assert tuple_equal(([1],), ([1],))
    assert not tuple_equal(([1],), ([2],))
    assert not tuple_equal(([1], [1, 2]), ([1, 2], [1, 2]))
    assert not tuple_equal(([1], [1, 2]), ([1], [1, 3]))
