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

"""Tests for linen_meta."""

from absl.testing import absltest
from flax import linen as nn
import jax
from jax import numpy as jnp
from jax import random
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit
from jax.sharding import Mesh
from jax.sharding import PartitionSpec


class LinenMetaTest(absltest.TestCase):

  def test_boxed_param(self):
    class Bar(nn.Module):

      @nn.compact
      def __call__(mdl_self, x):  # pylint: disable=no-self-argument
        kernel_init = nn.with_partitioning(nn.initializers.ones_init(),
                                           ('in', 'out'))
        kernel = mdl_self.param('kernel', kernel_init, (x.shape[-1], 2))
        kernel_box = mdl_self.get_variable('params', 'kernel')
        self.assertIsInstance(kernel_box, nn.Partitioned)
        self.assertEqual(kernel_box.names, ('in', 'out'))
        return x @ kernel

    class Foo(nn.Module):

      @nn.compact
      def __call__(self, xs):
        return nn.vmap(
            Bar, in_axes=0,
            variable_axes={'params': 0}, split_rngs={'params': True},
            metadata_params={nn.PARTITION_NAME: 'batch'})(name='bar')(xs)

    m = Foo()
    variables = m.init(random.PRNGKey(0), jnp.zeros((8, 3)))
    self.assertEqual(variables['params']['bar']['kernel'].names,
                     ('batch', 'in', 'out'))


  def test_boxed_variable(self):
    class Bar(nn.Module):

      @nn.compact
      def __call__(mdl_self, x):  # pylint: disable=no-self-argument
        kernel_init = nn.with_partitioning(nn.initializers.ones_init(),
                                           ('in', 'out'))
        kernel = mdl_self.variable(
            'params', 'kernel', kernel_init,
            mdl_self.make_rng('params'), (x.shape[-1], 2))
        kernel.value += 1.
        self.assertEqual(kernel.value.sum(), kernel.value.size * 2)
        kernel_box = mdl_self.get_variable('params', 'kernel')
        self.assertIsInstance(kernel_box, nn.Partitioned)
        self.assertEqual(kernel_box.names, ('in', 'out'))
        return x @ kernel.value

    class Foo(nn.Module):

      @nn.compact
      def __call__(self, xs):
        return nn.vmap(
            Bar, in_axes=0,
            variable_axes={'params': 0}, split_rngs={'params': True},
            metadata_params={nn.PARTITION_NAME: 'batch'})(name='bar')(xs)

    m = Foo()
    variables = m.init(random.PRNGKey(0), jnp.zeros((8, 3)))
    self.assertEqual(variables['params']['bar']['kernel'].names,
                     ('batch', 'in', 'out'))


  # def test_boxed_variable(self):
  #   def f(scope, xs):
  #     def g(scope, x):
        # kernel_init = nn.with_partitioning(nn.initializers.ones_init(),
        #                                      ('in', 'out'))
        # kernel = scope.variable('params', 'kernel', kernel_init,
        #                         scope.make_rng('params'), (x.shape[-1], 2))
        # kernel.value += 1.
        # self.assertEqual(kernel.value.sum(), kernel.value.size * 2)
        # kernel_box = scope.get_variable('params', 'kernel')
        # self.assertIsInstance(kernel_box, nn.Partitioned)
        # self.assertEqual(kernel_box.names, ('in', 'out'))
        # return x @ kernel.value

  #     nn.vmap(
  #         g, in_axes=0,
  #         variable_axes={'params': 0}, split_rngs={'params': True},
  #         metadata_params={nn.PARTITION_NAME: 'batch'})(scope, xs)

  #   _, variables = init(f)(random.PRNGKey(0), jnp.zeros((8, 3)))
  #   self.assertEqual(variables['params']['kernel'].names,
  #                    ('batch', 'in', 'out'))

  def test_pjit_scan_over_layers(self):
    class MLP(nn.Module):
      hidden_size: int

      @nn.compact
      def __call__(self, x):
        ki = nn.linear.default_kernel_init
        h = nn.Dense(
            self.hidden_size,
            kernel_init=nn.with_partitioning(ki, ('data', 'model')))(x)
        h = nn.relu(h)
        return nn.Dense(
            x.shape[-1],
            kernel_init=nn.with_partitioning(ki, ('model', 'data')))(h)

    class Model(nn.Module):

      @nn.compact
      def __call__(self, x):
        def body(_, c):
          c = MLP(512)(c)
          return c, ()
        c, _ = nn.scan(
            body, variable_axes={'params': 0}, split_rngs={'params': 0},
            length=8, metadata_params={nn.PARTITION_NAME: None})(
                self, x)
        return c

    devs = mesh_utils.create_device_mesh((jax.device_count(), 1))
    mesh = Mesh(devs, ['data', 'model'])
    model = Model()
    x = jnp.ones((8, 128))
    spec = nn.get_partition_spec(
        jax.eval_shape(model.init, random.PRNGKey(0), x))
    self.assertEqual(spec, {
        'params': {
            'MLP_0': {
                'Dense_0': {
                    'bias': None,
                    'kernel': PartitionSpec(None, 'data', 'model'),
                },
                'Dense_1': {
                    'bias': None,
                    'kernel': PartitionSpec(None, 'model', 'data'),
                },
            },
        },
    })
    init_fn = mesh(pjit(model.init, (
        None, PartitionSpec('data', 'model')), spec))
    variables = init_fn(random.PRNGKey(0), x)
    apply_fn = mesh(pjit(model.apply, (
        spec, PartitionSpec('data', 'model')), PartitionSpec('data', 'model')))
    y = apply_fn(variables, x)
    self.assertEqual(y.shape, (8, 128))

if __name__ == '__main__':
  absltest.main()
