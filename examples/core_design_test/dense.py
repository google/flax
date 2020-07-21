from flax.core import Scope, init, apply, nn
from jax import numpy as jnp, random

from flax import struct

from jax.scipy.linalg import expm

from dataclasses import dataclass, InitVar
from typing import Any, Callable, Sequence, NamedTuple, Any, Optional

from dataclasses import dataclass

Initializer = Callable[..., Any]
Array = Any



@dataclass
class Dense:
  features: int
  bias: bool = True
  kernel_init: Initializer = nn.linear.default_kernel_init
  bias_init: Initializer = nn.initializers.zeros

  def __call__(self, scope, x):
    kernel = scope.param('kernel', self.kernel_init, (x.shape[-1], self.features))
    y = x @ kernel
    if self.bias:
      y += scope.param('bias', self.bias_init, (self.features,))
    return y


@struct.dataclass
class ExplicitDense:
  kernel: Array
  bias: Optional[Array]

  # a fully explicit "scope free" version
  @staticmethod
  def create(rng, in_size, out_size, bias=True,
             kernel_init=nn.linear.default_kernel_init,
             bias_init=nn.initializers.zeros):
    k1, k2 = random.split(rng, 2)
    kernel = kernel_init(k1, (in_size, out_size))
    if bias:
      bias = bias_init(k2, (out_size,))
    else:
      bias = None
    return ExplicitDense(kernel, bias)

  # a semi-explicit version where a scope is used to create explicit params
  @staticmethod
  def create_in_scope(scope, in_size, out_size, bias=True,
                      kernel_init=nn.linear.default_kernel_init,
                      bias_init=nn.initializers.zeros):
    kernel = scope.param('kernel', kernel_init, (in_size, out_size))
    if bias:
      bias = scope.param('bias', bias_init, (out_size,))
    else:
      bias = None
    return ExplicitDense(kernel, bias)

  def __call__(self, x):
    y = x @ self.kernel
    if self.bias is not None:
      y += self.bias
    return y

def explicit_mlp(scope, x, sizes=(3, 1)):
  for i, size in enumerate(sizes):
    dense = scope.param(f'dense_{i}', ExplicitDense.create, x.shape[-1], size)
    x = dense(x)
    if i + 1 < len(sizes):
      x = nn.relu(x)
  return x

def semi_explicit_mlp(scope, x, sizes=(3, 1)):
  for i, size in enumerate(sizes):
    dense = scope.child(ExplicitDense.create_in_scope, prefix='dense_')(x.shape[-1], size)
    x = dense(x)
    if i + 1 < len(sizes):
      x = nn.relu(x)
  return x

if __name__ == "__main__":
  model = Dense(features=4)
  x = jnp.ones((1, 3))

  y, params = init(model)(random.PRNGKey(0), x)

  print(y)
  print(params)


  print('explicit dense:')
  y, params = init(explicit_mlp)(random.PRNGKey(0), x)

  print(y)
  print(params)

  print('semi-explicit dense:')
  y, params = init(semi_explicit_mlp)(random.PRNGKey(0), x)

  print(y)
  print(params)