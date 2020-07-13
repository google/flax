from dataclasses import dataclass
import jax
from jax import numpy as jnp, random, lax, jit
from flax import nn
from flax.core.scope import Scope
from flax.nn import initializers
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.linen import Module, MultiModule
import numpy as np


from flax.core.frozen_dict import freeze, unfreeze, FrozenDict

def standardize(x, axis, eps=1e-8):
  x = x - jnp.mean(x, axis=axis, keepdims=True)
  x = x / jnp.sqrt(jnp.mean(jnp.square(x), axis=axis, keepdims=True) + eps)
  return x
  
class Dense(Module):
  features: int
  def __call__(self, x):
    kernel = self.param('kernel', initializers.lecun_normal(), (x.shape[-1], self.features))
    return jnp.dot(x, kernel) + self.param('bias', initializers.zeros, (self.features,))




@dataclass
class StdWeight:
  module: Module
  
  def __call__(self, x):
    if not 'param' in self.module.vars:
      # initialize parameters
      self.module(x)

    param = self.module.variables['param']
    # Need to make a copy because `param` is (and should be) frozen. We're only transforming
    # the parameters, not mutating them.
    std_param = param.copy(kernel=standardize(param['kernel'], axis=[0, 1]))
    scope = Scope(variables={"param": std_param})
    return self.module.clone(parent=scope)(x)

class MyModule(Module):
  def __call__(self, x):
    module = Dense(self, 3)
    std_module = StdWeight(module)
    return std_module(x)  # parameters

m = MyModule(parent=None).initialized({'param': jax.random.PRNGKey(10)}, jnp.ones((1, 4)))
print(m.variables)
