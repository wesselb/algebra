# [Algebra](http://github.com/wesselb/algebra)

[![CI](https://github.com/wesselb/algebra/workflows/CI/badge.svg?branch=master)](https://github.com/wesselb/algebra/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/algebra/badge.svg?branch=master&service=github)](https://coveralls.io/github/wesselb/algebra?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://user.github.io/algebra)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Algebraic structures

*Note:* Algebra requires Python 3.6 or higher.

## Requirements and Installation

```bash
pip install algebra
```

See also [the instructions here](https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc).

## Algebra

This package provides an algebra where the elements can be manipulated 
in a natural way, with basic algebraic simplifications happening automatically.
It also support equality checking, which is conservative:
if `x == y`, then `x` is equal to `y`;
but if `x != y`, then either `x` is different from `y`, or it could not be 
proven that `x` is equal to `y`.

As an example, let's create numbered elements.

```python
from algebra import Element


class Numbered(Element):
    total = 0
    
    def __init__(self):
        self.num = Numbered.total
        Numbered.total += 1
    
    def render(self, formatter):
        return f'x{self.num}'
```

Then instances of `Numbered` can be manipulated as follows.

```python
>>> x0 = Numbered()

>>> x1 = Numbered()

>>> x0 == x0
True

>>> x0 == x1
False

>>> x0 + x1
x0 + x1

>>> x0 + x0
2 * x0

>>> x0 + x1 == x1 + x0
True

>>> x0 - x0
0

>>> 2 + x0
2 * 1 + x0

>>> (2 + x0) * x1
(2 * 1 + x0) * x1

>>> (2 + x0) * x1 * 0
0
```


## Create Your Own Algebra

Coming soon.

## Function Algebra

Coming soon.

## Create Your Own Function Algebra

Coming soon.


