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

from absl.testing import absltest
import flax
from flax import nnx
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
import optax


class TestSPMD(absltest.TestCase):
  def test_init(self):
    if jax.device_count() < 4:
      self.skipTest('At least 4 devices required')
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
      return nnx.split(Foo())

    mesh = Mesh(mesh_utils.create_device_mesh((2, 2)), ('model', 'data'))

    with mesh:
      m: Foo = nnx.merge(*create_module())  # type: ignore[invalid-annotation]

    assert m.w.shape == (8, 2)
    assert m.w.sharding.shard_shape(m.w.shape) == (4, 1)

  def test_init_all_devices(self):
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
      return nnx.split(Foo())

    mesh = Mesh(mesh_utils.create_device_mesh((1, 1)), ('model', 'data'))

    with mesh:
      m: Foo = nnx.merge(*create_module())  # type: ignore[invalid-annotation]

    assert m.w.value.shape == (8, 2)
    assert m.w.value.sharding.shard_shape(m.w.value.shape) == (8, 2)

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

    graphdef, params = nnx.split(Foo())
    state = nnx.TrainState.create(
      graphdef,
      params=params,
      tx=optax.adam(1e-3),
    )
    state_spec = nnx.get_partition_spec(state)

    assert state_spec.params['w'].value == PartitionSpec('row', 'col')
    assert state_spec.opt_state[0].mu['w'].value == PartitionSpec('row', 'col')
    assert state_spec.opt_state[0].nu['w'].value == PartitionSpec('row', 'col')

  def test_add_remove_axis_in_transform(self):
    test = self
    kadds, kremoves, badds, bremoves = [], [], [], []
    class MLP(nnx.Module):

      @nnx.split_rngs(splits=5)
      @nnx.vmap(
          in_axes=(0, 0),
          transform_metadata={nnx.PARTITION_NAME: 'layers', 'nickname': 'nick'},
      )
      def __init__(self, rngs: nnx.Rngs):
        self.linear = nnx.Linear(
          3,
          3,
          kernel_init=nnx.with_metadata(
            nnx.initializers.lecun_normal(),
            sharding=('din', 'dout'),
            nickname=('in', 'out'),
            on_add_axis=lambda _, idx, name: kadds.append((idx, name)),
            on_remove_axis=lambda _, idx, name: kremoves.append((idx, name)),
          ),
          bias_init=nnx.with_metadata(
            nnx.initializers.zeros_init(),  # no sharding annotation here!
            on_add_axis=lambda _, idx, name: badds.append((idx, name)),
            on_remove_axis=lambda _, idx, name: bremoves.append((idx, name)),
          ),
          rngs=rngs,
        )

      @nnx.scan(
          in_axes=(0, nnx.Carry),
          transform_metadata={nnx.PARTITION_NAME: 'layers'}
      )
      def __call__(self, x: jax.Array):
        x = self.linear(x)
        # test sharding layer axes is not present inside scan
        test.assertEqual(self.linear.kernel.shape, (3, 3))
        test.assertEqual(self.linear.kernel.sharding, ('din', 'dout'))
        # at least a remove_axis was already called to remove the layer axis
        test.assertEqual(kremoves[-1], (0, 'layers'))
        test.assertEqual(bremoves[-1], (0, 'layers'))
        return x, None

    m = MLP(rngs=nnx.Rngs(0))
    self.assertEqual(m.linear.kernel.shape, (5, 3, 3))
    self.assertEqual(m.linear.kernel.sharding, ('layers', 'din', 'dout'))
    self.assertEqual(m.linear.kernel.nickname, ('nick', 'in', 'out'))
    self.assertEqual(m.linear.bias.shape, (5, 3))
    # One add_axis called to add the `nnx.vmap` dimension
    self.assertEqual(kadds, [(0, 'layers')])
    self.assertEqual(kremoves, [])
    self.assertEqual(badds, [(0, 'layers')])
    self.assertEqual(bremoves, [])

    # One remove_axis and one add_axis called when in and out of `nnx.scan`
    y = m(jnp.ones((5, 3)))
    self.assertEqual(kadds, [(0, 'layers'), (0, 'layers')])
    self.assertEqual(kremoves, [(0, 'layers')])
    self.assertEqual(badds, [(0, 'layers'), (0, 'layers')])
    self.assertEqual(bremoves, [(0, 'layers')])

  def test_logical_rules(self):
    class Foo(nnx.Module):

      def __init__(self):
        self.w = nnx.Param(
            nnx.with_partitioning(
                lambda: jnp.ones((8, 2)),
                sharding=('row-alias', 'col-alias'),
                sharding_rules=(('row-alias', 'row'),),
            )()
        )
        self.b = nnx.Param(
            nnx.with_partitioning(
                lambda: jnp.zeros((2,)), sharding=('col-alias',)
            )()
        )

      def __call__(self, x):
        return x @ self.w + self.b

    graphdef, params = nnx.split(Foo())
    state = nnx.TrainState.create(
        graphdef,
        params=params,
        tx=optax.adam(1e-3),
    )
    with flax.core.spmd.logical_axis_rules((('col-alias', 'col'),)):
      state_spec = nnx.get_partition_spec(state)

    assert state_spec.params['w'].value == PartitionSpec('row', 'col')
    assert state_spec.opt_state[0].mu['w'].value == PartitionSpec('row', 'col')
    assert state_spec.opt_state[0].nu['w'].value == PartitionSpec('row', 'col')


if __name__ == '__main__':
  absltest.main()
