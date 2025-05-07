# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import optax

from absl.testing import absltest
import numpy as np

from flax import linen
from flax import nnx


class TrainState(nnx.TrainState):
  batch_stats: nnx.State


class TestHelpers(absltest.TestCase):
  def test_train_state(self):
    m = nnx.Dict(a=nnx.Param(1), b=nnx.BatchStat(2))

    graphdef, params, batch_stats = nnx.split(m, nnx.Param, nnx.BatchStat)

    state = TrainState.create(
      graphdef,
      params=params,
      tx=optax.sgd(1.0),
      batch_stats=batch_stats,
    )

    leaves = jax.tree_util.tree_leaves(state)

  def test_train_state_methods(self):
    class Foo(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 4, rngs=rngs)
        self.batch_norm = nnx.BatchNorm(4, rngs=rngs)

      def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        x = self.linear(x)
        x = self.batch_norm(x, use_running_average=not train)
        return x

    module = Foo(rngs=nnx.Rngs(0))
    graphdef, params, batch_stats = nnx.split(module, nnx.Param, nnx.BatchStat)

    state = TrainState.create(
      graphdef,
      params=params,
      tx=optax.sgd(1.0),
      batch_stats=batch_stats,
    )

    x = jax.numpy.ones((1, 2))
    y, _updates = state.apply('params', 'batch_stats')(x, train=True)

    assert y.shape == (1, 4)

    # fake gradient
    grads = jax.tree.map(jnp.ones_like, state.params)
    # test apply_gradients
    state = state.apply_gradients(grads)

  def test_nnx_linen_sequential_equivalence(self):
    key1, key2 = jax.random.split(jax.random.key(0), 2)
    rngs = nnx.Rngs(0)
    x = jax.random.uniform(key1, (3, 1, 5))

    model_nnx = nnx.Sequential(
      nnx.Linear(5, 4, rngs=rngs), nnx.Linear(4, 2, rngs=rngs)
    )
    model = linen.Sequential([linen.Dense(4), linen.Dense(2)])

    variables = model.init(key2, x)
    for layer_index in range(2):
      for param in ('kernel', 'bias'):
        variables['params'][f'layers_{layer_index}'][param] = getattr(
          model_nnx.layers[layer_index], param
        ).value
    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    np.testing.assert_array_equal(out, out_nnx)

    variables = model.init(key2, x)
    for layer_index in range(2):
      for param in ('kernel', 'bias'):
        getattr(model_nnx.layers[layer_index], param).value = variables[
          'params'
        ][f'layers_{layer_index}'][param]
    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    np.testing.assert_array_equal(out, out_nnx)


if __name__ == '__main__':
  absltest.main()

