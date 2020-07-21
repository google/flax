from flax.core import Scope, init, apply, nn
from jax import numpy as jnp, random

from jax.scipy.linalg import expm

from dataclasses import dataclass
from typing import Any, Callable, Sequence, NamedTuple, Any


Initializer = Callable[..., Any]
Array = Any
Flow = Any



@dataclass
class DenseFlow:
  kernel_init: Initializer = nn.linear.default_kernel_init
  bias_init: Initializer = nn.initializers.zeros

  def params(self, scope: Scope, features: int):
    kernel = scope.param('kernel', self.kernel_init, (features, features))
    bias = scope.param('bias', self.bias_init, (features,))
    return kernel, bias

  def forward(self, scope: Scope, x: Array):
    kernel, bias = self.params(scope, x.shape[-1])
    return jnp.dot(x, expm(kernel)) + bias

  def backward(self, scope: Scope, y: Array):
    kernel, bias = self.params(scope, y.shape[-1])
    return jnp.dot(y - bias, expm(-kernel))


@dataclass
class StackFlow:
  flows: Sequence[Flow]

  def forward(self, scope: Scope, x: Array):
    for i, f in enumerate(self.flows):
      x = scope.child(f.forward, name=str(i))(x)
    return x

  def backward(self, scope: Scope, x: Array):
    for i, f in reversed(tuple(enumerate(self.flows))):
      x = scope.child(f.backward, name=str(i))(x)
    return x

if __name__ == "__main__":
  flow = StackFlow((DenseFlow(),) * 3)
  # forward and backward are interchangeable here
  # so shape inference and initialization can be done on the forward and backward pass of the flow
  y, params = init(flow.forward)(random.PRNGKey(0), jnp.ones((1, 3)))
  print(params)
  x_restore = apply(flow.backward)(params, y)
  print(x_restore)