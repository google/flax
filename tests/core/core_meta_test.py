# Copyright 2022 The Flax Authors.
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

from flax import errors
from flax.core import init, lift, meta, nn
import jax
from jax import numpy as jnp
from jax import random
from jax import sharding
from jax.experimental import mesh_utils


class MetaTest(absltest.TestCase):

  def test_boxed_param(self):
    def f(scope, xs):
      def g(scope, x):
        kernel_init = meta.with_partitioning(nn.initializers.ones_init(),
                                             ('in', 'out'))
        kernel = scope.param('kernel', kernel_init, (x.shape[-1], 2))
        kernel_box = scope.get_variable('params', 'kernel')
        self.assertIsInstance(kernel_box, meta.Partitioned)
        self.assertEqual(kernel_box.names, ('in', 'out'))
        return x @ kernel

      lift.vmap(
          g, in_axes=0,
          variable_axes={'params': 0}, split_rngs={'params': True},
          metadata_params={meta.PARTITION_NAME: 'batch'})(scope, xs)

    _, variables = init(f)(random.PRNGKey(0), jnp.zeros((8, 3)))
    self.assertEqual(variables['params']['kernel'].names,
                     ('batch', 'in', 'out'))

  def test_boxed_variable(self):
    def f(scope, xs):
      def g(scope, x):
        kernel_init = meta.with_partitioning(nn.initializers.ones_init(),
                                             ('in', 'out'))
        kernel = scope.variable('params', 'kernel', kernel_init,
                                scope.make_rng('params'), (x.shape[-1], 2))
        kernel.value += 1.
        self.assertEqual(kernel.value.sum(), kernel.value.size * 2)
        kernel_box = scope.get_variable('params', 'kernel')
        self.assertIsInstance(kernel_box, meta.Partitioned)
        self.assertEqual(kernel_box.names, ('in', 'out'))
        return x @ kernel.value

      lift.vmap(
          g, in_axes=0,
          variable_axes={'params': 0}, split_rngs={'params': True},
          metadata_params={meta.PARTITION_NAME: 'batch'})(scope, xs)

    _, variables = init(f)(random.PRNGKey(0), jnp.zeros((8, 3)))
    self.assertEqual(variables['params']['kernel'].names,
                     ('batch', 'in', 'out'))

  def test_partition_axis_unspecified(self):
    def f(scope, xs):
      def g(scope, x):
        kernel_init = meta.with_partitioning(nn.initializers.ones_init(),
                                             ('in', 'out'))
        scope.param('kernel', kernel_init, (3, 2))
        return x

      with self.assertRaises(errors.PartitioningUnspecifiedError):
        lift.vmap(
            g, in_axes=0,
            variable_axes={'params': 0}, split_rngs={'params': True},
            metadata_params={})(scope, xs)
    init(f)(random.PRNGKey(0), jnp.zeros((8, 3)))

  def test_unbox(self):
    xs = {'kernel': meta.Partitioned(jnp.zeros((3, 2)), ('in', 'out')),
          'complex': meta.Partitioned(
              {'K': jnp.zeros((3, 2)), 'b': jnp.zeros((3,))}, ('data',))}
    unboxed = meta.unbox(xs)
    unboxed_shapes = jax.tree_map(jnp.shape, unboxed)
    self.assertEqual(unboxed_shapes, {
        'kernel': (3, 2),
        'complex': {
            'K': (3, 2), 'b': (3,),
        }
    })

  def test_scan_over_layers(self):
    def f(scope, x):
      def body(scope, x):
        kernel_init = meta.with_partitioning(nn.initializers.ones_init(),
                                             ('in', 'out'))
        y = nn.dense(scope, x, 3, kernel_init=kernel_init)
        return y, ()

      c, _ = lift.scan(
          body,
          variable_axes={'params': 0}, split_rngs={'params': True},
          length=8,
          metadata_params={meta.PARTITION_NAME: 'layers'})(scope, x)
      return c

    _, variables = init(f)(random.PRNGKey(0), jnp.zeros((8, 3)))
    boxed_shapes = jax.tree_map(jnp.shape, variables['params'])
    self.assertEqual(boxed_shapes, {
        'kernel': meta.Partitioned((8, 3, 3), ('layers', 'in', 'out')),
        'bias': (8, 3),
    })

  def test_get_partition_spec(self):
    xs = {'kernel': meta.Partitioned(jnp.zeros((8, 3, 3)),
                                     ('layers', 'in', 'out')),
          'bias': jnp.zeros((8, 3))}
    ps = meta.get_partition_spec(xs)
    self.assertEqual(
        ps,
        {
            'kernel': jax.sharding.PartitionSpec('layers', 'in', 'out'),
            'bias': None,
        },
    )

  def test_boxed_param_with_mesh(self):
    devices = mesh_utils.create_device_mesh((jax.local_device_count(), 1))
    mesh = sharding.Mesh(devices, ('in', 'out'))

    def f(scope, x):
        kernel_init = meta.with_partitioning(
          nn.initializers.ones_init(),('in', 'out'), mesh=mesh)
        kernel = scope.param('kernel', kernel_init, (x.shape[-1], 2))
        kernel_box = scope.get_variable('params', 'kernel')
        self.assertIsInstance(kernel_box, meta.Partitioned)
        self.assertEqual(kernel_box.names, ('in', 'out'))
        return x @ kernel

    @jax.jit
    def create_state():
      y, variables = init(f)(random.PRNGKey(0), jnp.zeros((8, 4)))
      spec = meta.get_partition_spec(variables)
      shardings = jax.tree_map(lambda s: sharding.NamedSharding(mesh, s), spec)
      variables = jax.lax.with_sharding_constraint(variables, shardings)
      return variables


    variables = create_state()
    self.assertEqual(variables['params']['kernel'].names,
                     ('in', 'out'))
    self.assertIs(variables['params']['kernel'].mesh, mesh)

if __name__ == '__main__':
  absltest.main()
