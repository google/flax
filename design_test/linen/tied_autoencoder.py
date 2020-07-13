import jax
from jax import numpy as jnp, random, lax
from flax import nn
from flax.nn import initializers
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.linen import Module, MultiModule
import numpy as np

class Dense(Module):
  features: int
  def __call__(self, x):
    kernel = self.param('kernel', initializers.lecun_normal(), (x.shape[-1], self.features))
    return jnp.dot(x, kernel)

# TODO: What does this look like with bias as well?
class TiedAutoEncoder(Module):
  def setup(self):
    self.encoder = Dense(self, features=4)

  @property
  def decoder(self):
    return self.encoder.scoped_clone(variables={
      "param": {"kernel": self.encoder.variables['param']['kernel'].T}})

  def __call__(self, x):
    z = self.encoder(x)
    x = self.decoder(z)
    return x

tae = TiedAutoEncoder(parent=None)
tae = tae.initialized(
  {'param': random.PRNGKey(42)},
  jnp.ones((1, 16)))
print("reconstruct", jnp.shape(tae(jnp.ones((1, 16)))))
print("var shapes", jax.tree_map(jnp.shape, tae.variables))



