import pytest

from ring import Scaled, Product

from .util import A, B


@pytest.mark.parametrize('e, result', [
    (A() + A() * B(),
     'A() + A() * B()'),

    (A() + (A() * B() + 2 * A()),
     'A() + A() * B() + 2 * A()'),

    (A() + 4 * (A() * B() + 2 * A()),
     'A() + 4 * (A() * B() + 2 * A())'),

    (2 * (A() + B()),
     '2 * (A() + B())'),

    ((A() + B()) * (A() + B()),
     '(A() + B()) * (A() + B())'),

    ((A() + B()) * (A() + B()) + A(),
     '(A() + B()) * (A() + B()) + A()'),

    # I'm not sure if the below can ever be constructed, but it should be
    # handled sensibly.
    (Product(Scaled(A(), 2), A() * B()),
     '2 * A() * A() * B()')
])
def test_pretty_printing(e, result):
    assert str(e) == result
