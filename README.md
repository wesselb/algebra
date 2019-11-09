# [Algebra](http://github.com/wesselb/algebra)

[![Build](https://travis-ci.org/wesselb/algebra.svg?branch=master)](https://travis-ci.org/wesselb/algebra)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/algebra/badge.svg?branch=master&service=github)](https://coveralls.io/github/wesselb/algebra?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://user.github.io/algebra)

Algebraic structures

*Note:* Algebra requires Python 3.6 or higher.

## Installation

Before installing the package, please ensure that `gcc` and `gfortran` are 
available.
On OS X, these are both installed with `brew install gcc`;
users of Anaconda may want to instead consider `conda install gcc`.
On Linux, `gcc` is most likely already available, and `gfortran` can be
installed with `apt-get install gfortran`.
Then simply

```bash
pip install algebra
```

## Algebra

This package provides an algebra where the elements manipulated as expected,
with basic algebraic simplifications done automatically.

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


