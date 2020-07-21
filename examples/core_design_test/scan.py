
from flax.core import Scope, init, apply, unfreeze, lift, nn
from typing import Sequence, Callable


import jax
from jax import lax, random, numpy as jnp

from typing import Any
from functools import partial


Array = Any

def mlp_scan(scope: Scope, xs: Array):
  def body_fn(scope, c, x):
    counter = scope.variable('counter', 'i', jnp.zeros, ())
    counter.value += 1
    x = scope.child(nn.dense)(x, 1)
    c *= 2
    return c, x

  print('starting with scan', xs.shape)
  counter = scope.variable('counter', 'i', jnp.zeros, ())
  print('counter before scan', counter.value)

  carry, ys = lift.scan(
      body_fn, scope, jnp.array([1]), xs,
      variable_modes={'param': 'broadcast', 'counter': 'carry'},
      split_rngs={'param': False})

  print('output carry', carry)
  
  print('counter after scan', counter.value)

  # output layer
  return carry, ys

if __name__ == "__main__":
  x = random.normal(random.PRNGKey(0), (1, 4))
  x = jnp.concatenate([x, x], 0)
  
  print(x.shape)

  # print('unshared params: (outputs should be different, parameters has extra dim)')
  # y, variables = init(mlp_scan)(random.PRNGKey(1), x, share_params=False)
  # print(y)
  # print(unfreeze(variables))

  # print('shared params: (outputs should be the same)')
  y, variables = init(mlp_scan)(random.PRNGKey(1), x)
  # print(y)
  # print(unfreeze(variables))