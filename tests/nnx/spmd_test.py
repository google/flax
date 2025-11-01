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

import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
import optax


class TestSPMD(parameterized.TestCase):

  def setUp(self):
    if jax.device_count() < 4:
      self.skipTest('At least 4 devices required')

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
      return nnx.split(Foo())

    mesh = jax.make_mesh((2, 2), ('model', 'data'))

    with jax.set_mesh(mesh):
      m: Foo = nnx.merge(*create_module())  # type: ignore[invalid-annotation]
      x = jax.device_put(jnp.zeros((4, 8)), P(None, 'model'))
      y = m(x)

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

    mesh = jax.make_mesh((1, 1), ('model', 'data'))

    with mesh:
      m: Foo = nnx.merge(*create_module())  # type: ignore[invalid-annotation]

    assert m.w.shape == (8, 2)
    assert m.w.sharding.shard_shape(m.w.shape) == (8, 2)

  def test_shard_optimizer_state(self):
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

    mesh = jax.make_mesh(((2, 2)), ('row', 'col'))
    with jax.set_mesh(mesh):
      graphdef, params = nnx.split(Foo())
      state = nnx.TrainState.create(
        graphdef,
        params=params,
        tx=optax.adam(1e-3),
      )

    assert state.params['w'].sharding.is_equivalent_to(
      NamedSharding(mesh, P('row', 'col')), ndim=2)
    assert state.opt_state[0].mu['w'].sharding.is_equivalent_to(
      NamedSharding(mesh, P('row', 'col')), ndim=2)
    assert state.opt_state[0].nu['w'].sharding.is_equivalent_to(
      NamedSharding(mesh, P('row', 'col')), ndim=2)

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
          4,
          4,
          kernel_init=nnx.with_metadata(
            nnx.initializers.lecun_normal(),
            sharding_names=('din', 'dout'),
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
        test.assertEqual(self.linear.kernel.shape, (4, 4))
        test.assertEqual(self.linear.kernel.sharding_names, ('din', 'dout'))
        # at least a remove_axis was already called to remove the layer axis
        test.assertEqual(kremoves[-1], (0, 'layers'))
        test.assertEqual(bremoves[-1], (0, 'layers'))
        return x, None

    mesh = jax.make_mesh(((1, 2, 2)), ('layers', 'din', 'dout'))
    with jax.set_mesh(mesh):
      m = MLP(rngs=nnx.Rngs(0))
    self.assertEqual(m.linear.kernel.shape, (5, 4, 4))
    self.assertEqual(m.linear.kernel.sharding_names, ('layers', 'din', 'dout'))
    self.assertEqual(m.linear.kernel.nickname, ('nick', 'in', 'out'))
    self.assertEqual(m.linear.bias.shape, (5, 4))
    # One add_axis called to add the `nnx.vmap` dimension
    self.assertEqual(kadds, [(0, 'layers')])
    self.assertEqual(kremoves, [])
    self.assertEqual(badds, [(0, 'layers')])
    self.assertEqual(bremoves, [])

    # One remove_axis and one add_axis called when in and out of `nnx.scan`
    with jax.set_mesh(mesh):
       _ = m(jnp.ones((5, 4)))
    self.assertEqual(kadds, [(0, 'layers'), (0, 'layers')])
    self.assertEqual(kremoves, [(0, 'layers')])
    self.assertEqual(badds, [(0, 'layers'), (0, 'layers')])
    self.assertEqual(bremoves, [(0, 'layers')])

  @parameterized.product(use_ref=[True, False])
  def test_logical_rules(self, use_ref):
    self.enter_context(nnx.use_refs(use_ref))
    class Foo(nnx.Module):

      def __init__(self):
        self.w = nnx.Param(
            nnx.with_partitioning(
                lambda: jnp.ones((8, 2)),
                sharding=('row-alias', 'col-alias'),
            )()
        )
        self.b = nnx.Param(
            nnx.with_partitioning(
                lambda: jnp.zeros((2,)), sharding=('col-alias2',),
                sharding_rules=(('col-alias2', 'col'),),
            )()
        )

      def __call__(self, x):
        return x @ self.w + self.b

    mesh = jax.make_mesh(((1, 2, 2)), ('layers', 'row', 'col'))
    global_rule = (('row-alias', 'row'),('col-alias', 'col'),)
    with jax.set_mesh(mesh), nnx.logical_axis_rules(global_rule):
      model = Foo()
      optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)


    assert model.w.sharding.is_equivalent_to(
      NamedSharding(mesh, P('row', 'col')), ndim=2)
    assert optimizer.opt_state[0].mu['w'].sharding.is_equivalent_to(
      NamedSharding(mesh, P('row', 'col')), ndim=2)
    assert optimizer.opt_state[0].nu['w'].sharding.is_equivalent_to(
      NamedSharding(mesh, P('row', 'col')), ndim=2)

  def test_get_abstract_model(self):
    class Foo(nnx.Module):
      def __init__(self, rngs):
        self.linear = nnx.Linear(
          8, 8, rngs=rngs, use_bias=False,
          kernel_init=nnx.with_partitioning(
            nnx.initializers.lecun_normal(), (None, 'model')))
        self.shared = self.linear.kernel

    mesh = jax.make_mesh(((2, 2)), ('batch', 'model'))
    gdef, abs_state = nnx.get_abstract_model(lambda: Foo(nnx.Rngs(0)), mesh)
    assert len(jax.tree.leaves(abs_state)) == 1
    assert jax.tree.leaves(abs_state)[0].sharding.is_equivalent_to(
      NamedSharding(mesh, P(None, 'model')), ndim=2)


if __name__ == '__main__':
  absltest.main()
