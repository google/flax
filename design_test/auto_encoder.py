from flax.core.scope import Scope, init, apply
from jax import numpy as jnp, random

from flax import nn, struct

from jax.scipy.linalg import expm

from dataclasses import dataclass, InitVar
from typing import Any, Callable, Sequence, NamedTuple, Any


Initializer = Callable[..., Any]
Array = Any


def mlp(scope: Scope, x: Array, hidden: int, out: int):
  x = scope.child(nn.dense, 'hidden')(x, hidden)
  x = nn.relu(x)
  return scope.child(nn.dense, 'out')(x, out)


@dataclass
class AutoEncoder:

  latents: int
  features: int
  hidden: int

  def __call__(self, scope, x):
    z = self.encode(scope, x)
    return self.decode(scope, z)

  def encode(self, scope, x):
    return scope.child(mlp, 'encoder')(x, self.hidden, self.latents)

  def decode(self, scope, z):
    return scope.child(mlp, 'decoder')(z, self.hidden, self.features)



ae = AutoEncoder(latents=2, features=4, hidden=3)
x = jnp.ones((1, 3))

x_r, params = init(ae)(random.PRNGKey(0), x)

print(x, x_r)
print(params)

print('AutoEncoder with scope')

def module_method(fn, name=None):
  if name is None:
    name = fn.__name__ if hasattr(fn, '__name__') else None

  def wrapper(self, *args, **kwargs):
    scope = self.scope.rewinded()  
    mod_fn = lambda scope: fn(self, scope, *args, **kwargs)
    return scope.child(mod_fn, name)()
  return wrapper

@dataclass
class AutoEncoder2:
  scope: Scope
  latents: int
  features: int
  hidden: int

  def __call__(self, x):
    z = self.encode(x)
    return self.decode(z)

  @module_method
  def encode(self, scope, x):
    return mlp(scope, x, self.hidden, self.latents)

  @module_method
  def decode(self, scope, z):
    return mlp(scope, z, self.hidden, self.features)

ae = lambda scope, x: AutoEncoder2(scope, latents=2, features=4, hidden=3)(x)
x = jnp.ones((1, 3))

x_r, params = init(ae)(random.PRNGKey(0), x)

print(x, x_r)
print(params)