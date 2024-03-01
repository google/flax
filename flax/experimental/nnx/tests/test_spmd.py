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
from jax._src import test_util as jtu
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec

from flax.experimental import nnx


class TestSPMD:
  @jtu.skip_on_devices('cpu', 'gpu')
  def test_init(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.w = nnx.Param(
          nnx.with_partitioning(
            lambda: jnp.ones((8, 2)),
            sharding=('model', 'data'),
          )()
        )

      def __call__(self, x):
        return x @ self.w

    @jax.jit
    def create_module():
      return Foo().split()

    mesh = Mesh(mesh_utils.create_device_mesh((2, 2)), ('model', 'data'))

    with mesh:
      m: Foo = nnx.merge(create_module())

    assert m.w.shape == (8, 2)
    assert m.w.sharding.shard_shape(m.w.shape) == (4, 1)

  def test_get_partition_spec(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.w = nnx.Param(
          nnx.with_partitioning(
            lambda: jnp.ones((8, 2)),
            sharding=('row', 'col'),
          )()
        )

      def __call__(self, x):
        return x @ self.w

    params, graphdef = Foo().split()
    state = nnx.TrainState(
      graphdef,
      params=params,
      tx=optax.adam(1e-3),
    )
    state_spec = nnx.get_partition_spec(state)

    assert state_spec.params['w'].raw_value == PartitionSpec('row', 'col')
    assert state_spec.opt_state[0].mu['w'].raw_value == PartitionSpec(
      'row', 'col'
    )
    assert state_spec.opt_state[0].nu['w'].raw_value == PartitionSpec(
      'row', 'col'
    )
