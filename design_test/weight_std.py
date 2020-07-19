
from flax.core import Scope, init, apply, unfreeze, lift, nn
from typing import Any, Sequence, Callable

import jax
from jax import lax, random, numpy as jnp

from functools import partial

Array = Any


def weight_std(fn, kernel_name='kernel', eps=1e-8):
  def std(variables):
    params = variables['param']
    assert kernel_name in params
    kernel = params[kernel_name]
    redux = tuple(range(kernel.ndim - 1))
    norm = jnp.square(kernel).sum(redux, keepdims=True)
    std_kernel = kernel / jnp.sqrt(norm + eps)
    params[kernel_name] = std_kernel
    return variables

  # transform handles a few of nasty edge cases here...
  # the transformed kind will be immutable inside fn
  # this way we avoid lost mutations to param
  # transform also avoids accidental reuse of rngs
  # and it makes sure that other state is updated correctly (not twice during init!)
  return lift.transform_module(fn, trans_in_fn=std)

def mlp(scope: Scope, x: Array,
        sizes: Sequence[int] = (2, 4, 1),
        act_fn: Callable[[Array], Array] = nn.relu):
  std_dense = weight_std(partial(nn.dense, kernel_init=nn.initializers.normal(stddev=1e5)))
  # hidden layers
  for size in sizes[:-1]:
    x = scope.child(std_dense, prefix='hidden_')(x, size)
    # x = act_fn(x)

  # output layer
  return scope.child(nn.dense, 'out')(x, sizes[-1])

if __name__ == "__main__":
  x = random.normal(random.PRNGKey(0), (1, 4,))
  y, params = init(mlp)(random.PRNGKey(1), x)
  print(y)
  print(jax.tree_map(jnp.shape, unfreeze(params)))