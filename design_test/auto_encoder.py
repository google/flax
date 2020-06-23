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


@struct.dataclass
class AutoEncoder:

  latents: int = struct.field(False)
  features: int = struct.field(False)
  hidden: int = struct.field(False)

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
