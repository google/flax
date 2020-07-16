from dataclasses import dataclass
import jax
from jax import numpy as jnp, random, lax, jit
from flax import nn
from flax.core.scope import Scope
from flax.nn import initializers
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.linen import Module, MultiModule
import numpy as np
from dense import Dense
from flax.core.frozen_dict import freeze, unfreeze, FrozenDict

def standardize(x, axis, eps=1e-8):
  x = x - jnp.mean(x, axis=axis, keepdims=True)
  x = x / jnp.sqrt(jnp.mean(jnp.square(x), axis=axis, keepdims=True) + eps)
  return x

# A wrapper that calls through a simple module with standardized parameters.
#
# Note that StdWeight is /not/ a module, hence it doesn't add another layer
# of depth in the variable dict (i.e. this is a "transparent module")
@dataclass
class StdWeight:
  module: Module

  def __call__(self, x):
    # TODO: Think about how this modifies other state
    if not 'param' in self.module.variables:
      # initialize parameters
      self.module(x)

    param = self.module.variables['param']
    # Make a copy because `param` is (and should be) frozen. We're only transforming
    # the parameters, not mutating them.
    std_param = param.copy(kernel=standardize(param['kernel'], axis=[0, 1]))
    return self.module.detached().attached(variables={"param": std_param})(x)

class MyModule(Module):
  def __call__(self, x):
    module = Dense(self, 3)
    std_module = StdWeight(module)
    return std_module(x)

m = MyModule(parent=None).initialized({'param': jax.random.PRNGKey(10)}, jnp.ones((1, 4)))
print(m.variables)
