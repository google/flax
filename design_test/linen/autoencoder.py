import jax
from jax import numpy as jnp, random, lax
from flax import nn
from flax.nn import initializers
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.linen import Module, MultiModule
import numpy as np


class Dense(Module):
  features: int
  kernel_init: Callable = initializers.lecun_normal()
  bias_init: Callable = initializers.zeros

  def __call__(self, x):
    kernel = self.param('kernel', self.kernel_init, (x.shape[-1], self.features))
    return jnp.dot(x, kernel) + self.param('bias', self.bias_init, (self.features,))


class MLP(Module):
  widths: Iterable
  
  def __call__(self, x):
    for width in self.widths[:-1]:
      x = nn.relu(Dense(self, width)(x))
    return Dense(self, self.widths[-1])(x)


class AutoEncoder(MultiModule):
  encoder_widths: Iterable
  decoder_widths: Iterable
  input_shape: Tuple = None

  def setup(self):
    self._encoder = MLP(self, self.encoder_widths)
    self._decoder = MLP(self, self.decoder_widths)
    self._decoder_final = Dense(self, np.prod(self.input_shape))

  def __call__(self, x):
    return self.decode(self.encode(x))

  def encode(self, x):
    assert x.shape[-len(self.input_shape):] == self.input_shape
    return self._encoder(jnp.reshape(x, [-1]))

  def decode(self, z):
    z = self._decoder(z)
    z = nn.relu(z)
    z = self._decoder_final(z)
    x = nn.sigmoid(z)
    x = jnp.reshape(x, (-1, ) + self.input_shape)
    return x
    
  
ae = AutoEncoder(
  parent=None,
  encoder_widths=(32, 32, 32),
  decoder_widths=(32, 32, 32),
  input_shape=(28, 28, 1))
ae = ae.initialized(
  jnp.ones((1, 28, 28, 1)),
  rngs={'param': random.PRNGKey(42)})
print(ae.variables)
print("reconstruct", ae(jnp.ones((1, 28, 28, 1))))
print("encoder", ae.encode(jnp.ones((1, 28, 28, 1))))
print("var shapes", jax.tree_map(jnp.shape, ae.variables))
print(ae.variables)
