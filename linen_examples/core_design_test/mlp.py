
from flax.core import Scope, init, apply, unfreeze, nn
from typing import Sequence, Callable

import jax
from jax import lax, random, numpy as jnp

from typing import Any
from functools import partial


Array = Any

def mlp(scope: Scope, x: Array,
        sizes: Sequence[int] = (2, 4, 1),
        act_fn: Callable[[Array], Array] = nn.relu):
  # hidden layers
  for size in sizes[:-1]:
    x = scope.child(nn.dense)(x, size)
    x = act_fn(x)
  # output layer
  return scope.child(nn.dense, 'out')(x, sizes[-1])



if __name__ == "__main__":
  x = random.normal(random.PRNGKey(0), (1, 4,))
  y, params = init(mlp)(random.PRNGKey(1), x)
  print(y.shape)
  print(jax.tree_map(jnp.shape, unfreeze(params)))
