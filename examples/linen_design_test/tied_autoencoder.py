import jax
from jax import numpy as jnp, random, lax
from flax import linen as nn
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.linen import Module, compact
import numpy as np
from dense import Dense

# TODO(avital, levskaya): resurrect this example once interactive api is restored.

# class TiedAutoEncoder(Module):
#   def setup(self):
#     self.encoder = Dense(features=4, use_bias=False)

#   @property
#   def decoder(self):
#     return self.encoder.detached().attached(variables={
#       "param": {"kernel": self.encoder.variables['param']['kernel'].T}})

#   def __call__(self, x):
#     z = self.encoder(x)
#     x = self.decoder(z)
#     return x

# tae = TiedAutoEncoder(parent=None)
# tae = tae.initialized(
#   {'param': random.PRNGKey(42)},
#   jnp.ones((1, 16)))
# print("reconstruct", jnp.shape(tae(jnp.ones((1, 16)))))
# print("var shapes", jax.tree_map(jnp.shape, tae.variables))
