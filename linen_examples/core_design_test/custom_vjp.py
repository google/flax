
from flax.core import Scope, init, apply, unfreeze, lift, nn
from typing import Any, Sequence, Callable

import jax
from jax import lax, random, numpy as jnp

from functools import partial


Array = Any

def mlp_custom_grad(scope: Scope, x: Array,
             sizes: Sequence[int] = (8, 1),
             act_fn: Callable[[Array], Array] = nn.relu):

  def fwd(scope, x, features):
    y = nn.dense(scope, x, features)
    return y, x

  def bwd(features, scope_fn, params, res, g):
    x = res
    fn = lambda params, x: nn.dense(scope_fn(params), x, features)
    _, pullback = jax.vjp(fn, params, x)
    g_param, g_x = pullback(g)
    g_param = jax.tree_map(jnp.sign, g_param)
    return g_param, g_x

  dense_custom_grad = lift.custom_vjp(fwd, backward_fn=bwd, nondiff_argnums=(2,))

  # hidden layers
  for size in sizes[:-1]:
    x = scope.child(dense_custom_grad)(x, size)
    x = act_fn(x)

  # output layer
  return scope.child(dense_custom_grad)(x, sizes[-1])

if __name__ == "__main__":
  x = random.normal(random.PRNGKey(0), (1, 4))
  x = jnp.concatenate([x, x], 0)

  print('shared params: (same inputs, same outputs)')
  y, params = init(mlp_custom_grad)(random.PRNGKey(1), x)
  print(y)
  print(jax.tree_map(jnp.shape, unfreeze(params)))

  print(jax.grad(lambda params, x: jnp.mean(apply(mlp_custom_grad)(params, x) ** 2))(params, x))