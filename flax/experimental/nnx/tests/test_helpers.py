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

from flax.experimental import nnx


class TestHelpers:
  def test_train_state(self):
    m = nnx.Dict(a=nnx.Param(1), b=nnx.BatchStat(2))

    params, batch_stats, graphdef = m.split(nnx.Param, nnx.BatchStat)

    state = nnx.TrainState(
      graphdef,
      params=params,
      tx=optax.sgd(1.0),
      batch_stats=nnx.TreeNode(batch_stats),
      other=nnx.Variable(100),
      int=200,
    )

    leaves = jax.tree_util.tree_leaves(state)

    assert 1 in leaves
    assert 2 in leaves
    assert 100 in leaves
    assert 200 not in leaves

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
    params, batch_stats, graphdef = module.split(nnx.Param, nnx.BatchStat)

    state = nnx.TrainState(
      graphdef,
      params=params,
      tx=optax.sgd(1.0),
      batch_stats=batch_stats,
    )

    x = jax.numpy.ones((1, 2))
    y, _updates = state.apply('params', 'batch_stats')(x, train=True)

    assert y.shape == (1, 4)

    # fake gradient
    grads = jax.tree_map(jnp.ones_like, state.params)
    # test apply_gradients
    state = state.apply_gradients(grads)
