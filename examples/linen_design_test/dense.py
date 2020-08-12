import jax
from jax import numpy as jnp, random, lax
from flax.linen import initializers
from typing import Callable
from flax.linen import Module, compact

class Dense(Module):
  features: int
  kernel_init: Callable = initializers.lecun_normal()
  bias_init: Callable = initializers.zeros
  use_bias: bool = True

  @compact
  def __call__(self, inputs):
    kernel = self.param('kernel', self.kernel_init,
                        (inputs.shape[-1], self.features))
    y = lax.dot_general(inputs, kernel,
                        (((inputs.ndim - 1,), (0,)), ((), ())),)
    if self.use_bias:
      bias = self.param('bias', self.bias_init, (self.features,))
      y = y + bias
    return y
