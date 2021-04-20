from typing import Union

import lab as B
from plum import Dispatcher

__all__ = ["squeeze", "get_subclasses", "broadcast", "identical", "to_tensor"]

_dispatch = Dispatcher()


def squeeze(xs):
    """Squeeze a sequence if it only contains a single element.

    Args:
        xs (sequence): Sequence to squeeze.

    Returns:
        object: `xs[0]` if `xs` consists of a single element and `xs` otherwise.
    """
    return xs[0] if len(xs) == 1 else xs


def get_subclasses(c):
    """Get all subclasses of a class.

    Args:
        c (type): Class to get subclasses of.

    Returns:
        list[type]: List of subclasses of `c`.
    """
    scs = c.__subclasses__()
    return scs + [x for sc in scs for x in get_subclasses(sc)]


def broadcast(op, xs, ys):
    """Perform a binary operation `op` on elements of `xs` and `ys`. If `xs` or
    `ys` has length 1, then it is repeated sufficiently many times to match the
    length of the other.

    Args:
        op (function): Binary operation.
        xs (sequence): First sequence.
        ys (sequence): Second sequence.

    Returns:
        tuple: Result of applying `op` to every element of `zip(xs, ys)` after
            broadcasting appropriately.
    """
    if len(xs) == 1 and len(ys) > 1:
        # Broadcast `xs`.
        xs = xs * len(ys)
    elif len(ys) == 1 and len(xs) > 1:
        # Broadcast `ys.
        ys = ys * len(xs)

    # Check that `xs` and `ys` are compatible now.
    if len(xs) != len(ys):
        raise ValueError(f'Inputs "{xs}" and "{ys}" could not be broadcasted.')

    # Perform operation.
    return tuple(op(x, y) for x, y in zip(xs, ys))


@_dispatch
def identical(x, y):
    """Check if two objects `x` are `y` are identical for the purpose of algebraic
    simplification.

    Args:
        x (object): First object.
        y (object): Second object.

    Returns:
        bool: `x` and `y` are identical.
    """
    return x is y


@_dispatch
def identical(x: Union[int, float], y: Union[int, float]):
    return x == y


@_dispatch.multi((tuple, tuple), (list, list))
def identical(x: Union[tuple, list], y: Union[tuple, list]):
    return len(x) == len(y) and all([identical(xi, yi) for xi, yi in zip(x, y)])


@_dispatch
def to_tensor(x: B.Numeric):
    """Convert object to tensor.

    Args:
        x (object): Object to convert to tensor.

    Returns:
        tensor: `x` as a tensor.
    """
    return x


@_dispatch
def to_tensor(x: Union[tuple, list]):
    return B.stack(*x, axis=0)
