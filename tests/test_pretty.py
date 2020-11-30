import pytest

from algebra import Scaled, Product

from .util import a, b


@pytest.mark.parametrize(
    "e, result",
    [
        (a + a * b, "a + a * b"),
        (a + (a * b + 2 * a), "a + a * b + 2 * a"),
        (a + 4 * (a * b + 2 * a), "a + 4 * (a * b + 2 * a)"),
        (2 * (a + b), "2 * (a + b)"),
        ((a + b) * (a + b), "(a + b) * (a + b)"),
        ((a + b) * (a + b) + a, "(a + b) * (a + b) + a"),
        # I'm not sure if the below can ever be constructed, but it should be
        # handled sensibly.
        (Product(Scaled(a, 2), a * b), "2 * a * a * b"),
    ],
)
def test_pretty_printing(e, result):
    assert str(e) == result
