
from flax.core import Scope, init, apply, unfreeze, lift, nn
from typing import Sequence, Callable


import jax
from jax import lax, random, numpy as jnp

from typing import Any
from functools import partial


Array = Any

def mlp_scan(scope: Scope, xs: Array,
             share_params: bool = False):

  scope.variable('counter', 'i', jnp.zeros, ())
  def body_fn(scope, c, x):
    counter = scope.variable('counter', 'i', jnp.zeros, ())
    counter.value += 1
    x = scope.child(nn.dense)(x, 1)
    return c, x

  if share_params:
    carry, ys = lift.scan(
        body_fn, scope, (), xs,
        variable_modes={'param': 'broadcast', 'counter': 'carry'},
        split_rngs={'param': False})
  else:
    carry, ys = lift.scan(
        body_fn, scope, (), xs,
        variable_modes={'param': 'scan', 'counter': 'carry'},
        split_rngs={'param': True})

  # output layer
  return carry, ys

if __name__ == "__main__":
  x = random.normal(random.PRNGKey(0), (1, 4))
  x = jnp.concatenate([x, x], 0)

  print('unshared params: (outputs should be different, parameters has extra dim)')
  y, variables = init(mlp_scan)(random.PRNGKey(1), x, share_params=False)
  print(y)
  print(unfreeze(variables))

  print('shared params: (outputs should be the same)')
  y, variables = init(mlp_scan)(random.PRNGKey(1), x, share_params=True)
  print(y)
  print(unfreeze(variables))