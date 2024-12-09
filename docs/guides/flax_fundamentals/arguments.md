# Dealing with Flax Module arguments

## Introduction

In Flax Linen we can define `Module` arguments either as dataclass attributes or as arguments to methods (usually `__call__`).
Typically the distinction is clear:
* Completely fixed properties, such as the choice of kernel initializer or number of output features, are hyperparameters and should be defined as dataclass attributes. Typically two Module instances with different hyperparamaters cannot share in a meaningful way.
* Dynamic properties, such as input data and top-level "mode switches" like `train=True/False`, should be passed as arguments to `__call__` or another method.

Some cases are however less clear cut. Take for example the `Dropout` module.
We have a number of clear hyperparameters:

1. The dropout rate
2. The axes for which a dropout mask is generated

And some clear call time arguments:

1. The input that should be masked using dropout
2. The (optional) rng used to sample the random mask

There is however one property that is ambiguous -- the `deterministic` property in a Dropout module.

If `deterministic` is `True` no dropout mask is sampled. This is typically used during model evaluation.
However, if we pass `eval=True` or `train=False` to a top-level Module, the `deterministic` argument needs
to be applied everywhere and the boolean argument needs to be passed down to all the layers that might use `Dropout`.
If instead `deterministic` is a dataclass attribute, we might do the following:

```python
from functools import partial
from flax import linen as nn

class ResidualModel(nn.Module):
  drop_rate: float

  @nn.compact
  def __call__(self, x, *, train):
    dropout = partial(nn.Dropout, rate=self.drop_rate, deterministic=not train)
    for i in range(10):
      x += ResidualBlock(dropout=dropout, ...)(x)
```

It makes sense to pass `determinstic` to the constructor here because this way we can pass the dropout template to the sub-modules.
Now the sub-module no longer needs to take care of train vs eval mode and can simply use the `dropout` argument.
Note that because the dropout layer can only be constructed in the sub-module we can only partially apply `deterministic` to the constructor but not to `__call__`.

However, if `deterministic` is a dataclass attribute we run into trouble when using the setup pattern. We would **want** to write our module code like this:

```python
class SomeModule(nn.Module):
  drop_rate: float

  def setup(self):
    self.dropout = nn.Dropout(rate=self.drop_rate)

  @nn.compact
  def __call__(self, x, *, train):
    # ...
    x = self.dropout(x, deterministic=not train)
    # ...
```

But, as defined above, `deterministic` would be an attribute, so this doesn't work.
Here it makes sense to pass `deterministic` during `__call__` because it depends on the `train` argument.

## Solution

We can support both use cases described before by allowing certain properties to be passed
as dataclass attributes or as method argument (but not both!).
This can be implemented as follows:
```python
class MyDropout(nn.Module):
  drop_rate: float
  deterministic: Optional[bool] = None

  @nn.compact
  def __call__(self, x, deterministic=None):
    deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)
    # ...
```

In this example `nn.merge_param` will ensure that either `self.deterministic` or `deterministic` is set but not both.
An error is raised if both values are `None` or both values are not `None`.
This avoids confusing behavior where 2 different parts of the code set the same parameter and one is overruled by the other.
It also avoids a default value which would probably cause either the train step or eval step of a training procedure to be broken by default.



## Functional Core

Functional core defines functions rather than classes.
Therefore, there is no clear distinction between hyperparameters and call-time arguments.
The only way to pre-determine the hyperparameters is by using `partial`.
On the upside, there are no ambiguous cases where method arguments could also be attributes.
