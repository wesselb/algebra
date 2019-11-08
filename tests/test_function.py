import pytest

from plum import Self, Dispatcher
from ring.function import (
    Function,
    OneFunction,
    ZeroFunction,
    TensorProductFunction,

    stretch,
    shift,
    select,
    transform,
    differentiate
)


class F(Function):
    dispatch = Dispatcher(in_class=Self)

    @dispatch(Self)
    def __eq__(self, other):
        return True

    def render(self, formatter):
        return 'f'


class G(Function):
    dispatch = Dispatcher(in_class=Self)

    @dispatch(Self)
    def __eq__(self, other):
        return True

    def render(self, formatter):
        return 'g'


f = F()
g = G()
one = OneFunction()
zero = ZeroFunction()


def check_equality(f_method, g_method, arg1, arg2):
    assert f_method(arg1) == f_method(arg1)
    assert f_method(arg1) != f_method(arg1, arg1)
    assert f_method(arg1) != f_method(arg1, arg2)

    assert f_method(arg1) != g_method(arg1)
    assert f_method(arg1) != g_method(arg1, arg1)
    assert f_method(arg1) != g_method(arg1, arg2)


def test_stretch():
    with pytest.raises(RuntimeError):
        stretch(1, 1)

    assert str(f.stretch(5)) == 'f > 5'
    assert str(f.stretch(5, 6)) == 'f > (5, 6)'
    assert str(f > 5) == 'f > 5'

    check_equality(f.stretch, g.stretch, 4, 5)

    # Test grouping.
    assert str(f.stretch(2).stretch(3)) == 'f > 6'
    assert str(f.stretch(2, 3).stretch(4)) == 'f > (8, 12)'
    assert str(f.stretch(4).stretch(2, 3)) == 'f > (8, 12)'

    # Test cancellation
    assert str(one.stretch(5)) == '1'
    assert str(zero.stretch(5)) == '0'


def test_shift():
    with pytest.raises(RuntimeError):
        shift(1, 1)

    assert str(f.shift(5)) == 'f shift 5'
    assert str(f.shift(5, 6)) == 'f shift (5, 6)'

    check_equality(f.shift, g.shift, 4, 5)

    # Test grouping.
    assert str(f.shift(2).shift(3)) == 'f shift 5'
    assert str(f.shift(2, 3).shift(4)) == 'f shift (6, 7)'
    assert str(f.shift(4).shift(2, 3)) == 'f shift (6, 7)'

    # Test cancellation
    assert str(one.shift(5)) == '1'
    assert str(zero.shift(5)) == '0'


def test_select():
    with pytest.raises(RuntimeError):
        select(1, 1)

    assert str(f.select(1)) == 'f : [1]'
    assert str(f.select([1])) == 'f : [1]'
    assert str(f.select([1], 2, [3, 4, 5])) == 'f : ([1], [2], [3, 4, 5])'
    assert str(f.select(1, 2)) == 'f : ([1], [2])'

    check_equality(f.select, g.select, 4, 5)

    # Test cancellation
    assert str(one.select(5)) == '1'
    assert str(zero.select(5)) == '0'

    # Check that the indices given can be converted to a list.
    with pytest.raises(ValueError):
        f.select([[1]])


def test_transform():
    def f1():
        pass

    def f2():
        pass

    with pytest.raises(RuntimeError):
        transform(1, 1)

    assert str(f.transform(f1)) == 'f transform f1'
    assert str(f.transform(None, f2)) == 'f transform (None, f2)'
    assert str(f.transform(f1, f2)) == 'f transform (f1, f2)'

    check_equality(f.transform, g.transform, f1, f2)

    # Test cancellation
    assert str(one.transform(f1)) == '1'
    assert str(zero.transform(f1)) == '0'


def test_differentiate():
    with pytest.raises(RuntimeError):
        differentiate(1, 1)

    assert str(f.diff(0)) == 'd(0) f'
    assert str(f.diff(0, 1)) == 'd(0, 1) f'

    check_equality(f.diff, g.diff, 4, 5)

    # Test cancellation
    assert str(one.diff(0)) == '0'
    assert str(zero.diff(0)) == '0'


def test_tensor_product():
    def f1():
        pass

    def f2():
        pass

    assert str(TensorProductFunction(f1)) == 'f1'
    assert str(TensorProductFunction(f1, f2)) == 'f1 x f2'

    assert str(2 * TensorProductFunction(f1, f2)) == '2 * (f1 x f2)'
    assert str(1 + TensorProductFunction(f1, f2)) == '1 + (f1 x f2)'

    assert str(2 * TensorProductFunction(f1)) == '2 * f1'
    assert str(1 + TensorProductFunction(f1)) == '1 + f1'

    # Check equality.
    assert TensorProductFunction(f1) == TensorProductFunction(f1)
    assert TensorProductFunction(f1) != TensorProductFunction(f2)
    assert TensorProductFunction(f1) != TensorProductFunction(f1, f1)
    assert TensorProductFunction(f1) != TensorProductFunction(f1, f2)
    assert TensorProductFunction(f1, f1) == TensorProductFunction(f1, f1)
    assert TensorProductFunction(f1, f2) == TensorProductFunction(f1, f2)


def test_function_conversion():
    def f1():
        pass

    assert str(f + f1) == 'f + f1'
    assert str(f1 + f) == 'f1 + f'

    assert str(f * f1) == 'f * f1'
    assert str(f1 * f) == 'f1 * f'
