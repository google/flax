from flax.core import Scope, init, apply, lift
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
class TiedAutoEncoder:

  latents: int = struct.field(False)
  features: int = struct.field(False)

  def __call__(self, scope, x):
    z = self.encode(scope, x)
    return self.decode(scope, z)

  def encode(self, scope, x):
    assert x.shape[-1] == self.features
    return self._tied(nn.dense)(scope, x, self.latents, bias=False)

  def decode(self, scope, z):
    assert z.shape[-1] == self.latents
    return self._tied(nn.dense, transpose=True)(scope, z, self.features, bias=False)
  
  def _tied(self, fn, transpose=False):
    if not transpose:
      return fn

    def tie(params):
      params['kernel'] = params['kernel'].T
      print(params)
      return params
    
    return lift.transform_module(tie, fn)




ae = TiedAutoEncoder(latents=2, features=4)
x = jnp.ones((1, ae.features))

x_r, params = init(ae)(random.PRNGKey(0), x)

print(x, x_r)
print(params)
