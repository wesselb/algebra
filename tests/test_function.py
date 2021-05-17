import pytest
from plum import dispatch

from algebra import (
    Function,
    OneFunction,
    ZeroFunction,
    TensorProductFunction,
    ReversedFunction,
    stretch,
    shift,
    select,
    transform,
    diff,
    reverse,
)


class F(Function):
    @dispatch
    def __eq__(self, other: "F"):
        return True

    def render(self, formatter):
        return "f"


class G(Function):
    @dispatch
    def __eq__(self, other: "G"):
        return True

    def render(self, formatter):
        return "g"


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

    assert str(f.stretch(5)) == "f > 5"
    assert str(f.stretch(5, 6)) == "f > (5, 6)"

    check_equality(f.stretch, g.stretch, 4, 5)

    # Test grouping.
    assert str(f.stretch(2).stretch(3)) == "f > 6"
    assert str(f.stretch(2, 3).stretch(4)) == "f > (8, 12)"
    assert str(f.stretch(4).stretch(2, 3)) == "f > (8, 12)"

    # Test cancellation
    assert str(one.stretch(5)) == "1"
    assert str(zero.stretch(5)) == "0"


def test_shift():
    with pytest.raises(RuntimeError):
        shift(1, 1)

    assert str(f.shift(5)) == "f shift 5"
    assert str(f.shift(5, 6)) == "f shift (5, 6)"

    check_equality(f.shift, g.shift, 4, 5)

    # Test grouping.
    assert str(f.shift(2).shift(3)) == "f shift 5"
    assert str(f.shift(2, 3).shift(4)) == "f shift (6, 7)"
    assert str(f.shift(4).shift(2, 3)) == "f shift (6, 7)"

    # Test cancellation
    assert str(one.shift(5)) == "1"
    assert str(zero.shift(5)) == "0"


def test_select():
    with pytest.raises(RuntimeError):
        select(1, 1)

    assert str(f.select(1)) == "f : [1]"
    assert str(f.select([1])) == "f : [1]"
    assert str(f.select([1], 2, [3, 4, 5])) == "f : ([1], [2], [3, 4, 5])"
    assert str(f.select(1, 2)) == "f : ([1], [2])"

    check_equality(f.select, g.select, 4, 5)

    # Test cancellation
    assert str(one.select(5)) == "1"
    assert str(zero.select(5)) == "0"

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

    assert str(f.transform(f1)) == "f transform f1"
    assert str(f.transform(None, f2)) == "f transform (None, f2)"
    assert str(f.transform(f1, f2)) == "f transform (f1, f2)"

    check_equality(f.transform, g.transform, f1, f2)

    # Test cancellation
    assert str(one.transform(f1)) == "1"
    assert str(zero.transform(f1)) == "0"


def test_differentiate():
    with pytest.raises(RuntimeError):
        diff(1, 1)

    assert str(f.diff(0)) == "d(0) f"
    assert str(f.diff(0, 1)) == "d(0, 1) f"

    check_equality(f.diff, g.diff, 4, 5)

    # Test cancellation
    assert str(one.diff(0)) == "0"
    assert str(zero.diff(0)) == "0"


def test_tensor_product():
    def f1():
        pass

    def f2():
        pass

    assert str(TensorProductFunction(f1)) == "f1"
    assert str(TensorProductFunction(f1, f2)) == "f1 x f2"

    assert str(2 * TensorProductFunction(f1, f2)) == "2 * (f1 x f2)"
    assert str(1 + TensorProductFunction(f1, f2)) == "1 + (f1 x f2)"

    assert str(2 * TensorProductFunction(f1)) == "2 * f1"
    assert str(1 + TensorProductFunction(f1)) == "1 + f1"

    # Check equality.
    assert TensorProductFunction(f1) == TensorProductFunction(f1)
    assert TensorProductFunction(f1) != TensorProductFunction(f2)
    assert TensorProductFunction(f1) != TensorProductFunction(f1, f1)
    assert TensorProductFunction(f1) != TensorProductFunction(f1, f2)
    assert TensorProductFunction(f1, f1) == TensorProductFunction(f1, f1)
    assert TensorProductFunction(f1, f2) == TensorProductFunction(f1, f2)


def test_reverse():
    def f1():
        pass

    def f2():
        pass

    with pytest.raises(RuntimeError):
        reverse(1)

    assert str(reverse(f)) == "Reversed(f)"
    assert str(reversed(f)) == "Reversed(f)"

    assert reverse(f) == reverse(f)
    assert reverse(f) != reverse(g)

    # Test cancellation.
    assert str(reverse(one)) == "1"
    assert str(reverse(zero)) == "0"

    # Test propagation.
    assert str(reverse(f + g)) == "Reversed(f) + Reversed(g)"
    assert str(reverse(f * g)) == "Reversed(f) * Reversed(g)"
    assert str(reverse(2 * f)) == "2 * Reversed(f)"

    # Test synergy with other types.
    assert str(reverse(reverse(f))) == "f"
    assert str(reverse(f.stretch(1, 2))) == "Reversed(f) > (2, 1)"
    assert str(reverse(f.shift(1, 2))) == "Reversed(f) shift (2, 1)"
    assert str(reverse(f.select(1, 2))) == "Reversed(f) : ([2], [1])"
    assert str(reverse(f.transform(f1, f2))) == "Reversed(f) transform (f2, f1)"
    assert str(reverse(f.diff(0, 1))) == "d(1, 0) Reversed(f)"
    assert str(reverse(TensorProductFunction(f1, f2))) == "f2 x f1"

    # Test parentheses.
    assert str(ReversedFunction(ReversedFunction(f))) == "Reversed(Reversed(f))"
    assert str(ReversedFunction(2 * f)) == "Reversed(2 * f)"
    assert str(ReversedFunction(f * g)) == "Reversed(f * g)"
    assert str(ReversedFunction(f + g)) == "Reversed(f + g)"
    assert str(ReversedFunction(f) * g) == "Reversed(f) * g"


def test_function_conversion():
    def f1():
        pass

    assert str(f + f1) == "f + f1"
    assert str(f1 + f) == "f1 + f"

    assert str(f * f1) == "f * f1"
    assert str(f1 * f) == "f1 * f"
