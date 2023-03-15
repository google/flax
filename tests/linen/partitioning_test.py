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

"""Tests for flax.linen.partitioning."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from flax.core import freeze, unfreeze
from flax.linen import partitioning
from jax.experimental import mesh_utils
from jax import sharding
import jax
from jax import random
import jax.numpy as jnp


mock = absltest.mock

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()

# Testing constants.
AXIS_RULES_1 = (('foo', 'data'), ('bar', 'model'), ('baz', None))
AXIS_RULES_2 = (('foo', 'model'), ('bar', None), ('baz', 'data'))


class PartitioningTest(parameterized.TestCase):

  def test_axis_rules(self):
    self.assertEqual(partitioning._axis_rules.rules, ())
    partitioning.set_axis_rules(AXIS_RULES_1)
    self.assertEqual(partitioning._axis_rules.rules, AXIS_RULES_1)
    self.assertEqual(partitioning.get_axis_rules(), AXIS_RULES_1)
    partitioning.set_axis_rules(())

  def test_axis_rules_context(self):
    partitioning.set_axis_rules(AXIS_RULES_1)
    self.assertEqual(partitioning.get_axis_rules(), AXIS_RULES_1)
    with partitioning.axis_rules(AXIS_RULES_2):
      self.assertEqual(partitioning.get_axis_rules(), AXIS_RULES_2)
    self.assertEqual(partitioning.get_axis_rules(), AXIS_RULES_1)

  def test_logical_to_mesh_axes(self):
    axes_0 = ('foo', 'bar')
    # direct rule assignment
    self.assertEqual(
        partitioning.logical_to_mesh_axes(axes_0, rules=AXIS_RULES_1),
        ('data', 'model'))
    # axis rules context
    with partitioning.axis_rules(AXIS_RULES_1):
      self.assertEqual(
          partitioning.logical_to_mesh_axes(axes_0), ('data', 'model'))
      # nested context
      with partitioning.axis_rules(AXIS_RULES_2):
        self.assertEqual(
            partitioning.logical_to_mesh_axes(axes_0), ('model', None))
    # duplicated logical names
    with partitioning.axis_rules(AXIS_RULES_1):
      with self.assertRaises(ValueError):
        partitioning.logical_to_mesh_axes(('foo', 'foo', 'baz'))

  def test_logical_to_mesh_axes_priorities(self):
    p_rules = (
        ('foo', 'model'),
        ('bar', 'model'),
        ('baz', 'data'))
    with partitioning.axis_rules(p_rules):
      self.assertEqual(
          partitioning.logical_to_mesh_axes(('foo', 'bar', 'baz')),
          ('model', None, 'data'))
      self.assertEqual(
          partitioning.logical_to_mesh_axes(('bar', 'foo', 'baz')),
          (None, 'model', 'data'))
      self.assertEqual(
          partitioning.logical_to_mesh_axes(('baz', 'bar', 'foo')),
          ('data', None, 'model'))
      self.assertEqual(
          partitioning.logical_to_mesh_axes(
              ('baz', 'bar', 'foo', 'unassigned')),
          ('data', None, 'model', None))

  @parameterized.parameters(
      dict(rules=(('a', ('model', 'data')), ('b', 'data')),
           axes=('a', 'b'),
           expected=(('model', 'data'), None)),
      dict(rules=(('a', ('model', 'replica')), ('b', 'data')),
           axes=('a', 'b'),
           expected=(('model', 'replica'), 'data')),
      dict(rules=(('a', ('model', 'replica')), ('b', ('data', 'model'))),
           axes=('a', 'b'),
           expected=(('model', 'replica'), None)),
      dict(rules=(('a', ('model', 'replica')), ('b', 'model')),
           axes=('a', 'b', 'c'),
           expected=(('model', 'replica'), None, None)),
      dict(rules=(),
           axes=('a', 'b', 'c'),
           expected=(None, None, None)),
      dict(rules=(('a', None), ('a', 'model')),
           axes=('a', 'b'),
           expected=(None, None)),
      dict(rules=(('baz', 'data'),
                  ('bar', None),
                  ('foo', 'model'),
                  ('foo', 'data')),
           axes=('baz', 'bar', 'foo'),
           expected=('data', None, 'model')),
  )
  def test_logical_to_mesh_axes_cases(self, rules, axes, expected):
    with partitioning.axis_rules(rules):
      result = partitioning.logical_to_mesh_axes(axes)
    self.assertEqual(result, expected)

  @mock.patch('flax.linen.spmd._with_sharding_constraint')
  def test_with_sharding_constraint(self, wsc_fn):
    arr = jnp.ones((2, 2))
    axes = ('foo', 'bar')
    partitioning.set_axis_rules(())
    _ = partitioning.with_sharding_constraint(arr, axes)
    wsc_fn.assert_not_called()
    with partitioning.axis_rules(AXIS_RULES_1):
      _ = partitioning.with_sharding_constraint(arr, None)
      wsc_fn.assert_not_called()
      _ = partitioning.with_sharding_constraint(arr, axes)
      wsc_fn.assert_called_with(
          arr, jax.sharding.PartitionSpec('data', 'model'), mesh=None
      )

  @mock.patch('flax.linen.spmd._with_sharding_constraint')
  def test_with_sharding_constraint_fallback(self, wsc_fn):
    arr = jnp.ones((2, 2))
    with partitioning.axis_rules(AXIS_RULES_1):
      _ = partitioning.with_sharding_constraint(arr, ('foo', 'not_recognized'))
      wsc_fn.assert_called_with(arr, jax.sharding.PartitionSpec('data', None), mesh=None)
      wsc_fn.reset_mock()
      _ = partitioning.with_sharding_constraint(
          arr, ('foo', 'not_recognized'),
          fallback=partitioning.RulesFallback.AXIS_IS_UNSHARDED)
      wsc_fn.assert_called_with(arr, jax.sharding.PartitionSpec('data', None), mesh=None)
      wsc_fn.reset_mock()
      with self.assertRaises(ValueError):
        _ = partitioning.with_sharding_constraint(
            arr, ('foo', 'not_recognized'),
            fallback=partitioning.RulesFallback.RAISE_ERROR)
      wsc_fn.assert_not_called()
      _ = partitioning.with_sharding_constraint(
          arr, ('foo', 'not_recognized'),
          fallback=partitioning.RulesFallback.NO_CONSTRAINT)
      wsc_fn.assert_not_called()

  @parameterized.parameters(dict(axes_spec=None), dict(axes_spec=()))
  def test_param_with_axes_no_axes(self, axes_spec):
    class ParamTest(nn.Module):

      @nn.compact
      def __call__(self, x):
        foo = partitioning.param_with_axes(
            'foo', lambda k, s, d: jnp.zeros(s, d),
            (2, 2), x.dtype, axes=axes_spec)
        return x + foo

    k = random.PRNGKey(0)
    x = jnp.ones((2, 2))
    _ = ParamTest().init(k, x)

  def test_param_with_axes(self):
    class ParamTest(nn.Module):

      @nn.compact
      def __call__(self, x):
        foo = partitioning.param_with_axes(
            'foo', lambda k, s, d: jnp.zeros(s, d),
            (2, 2), x.dtype, axes=('foo', 'bar'))
        return x + foo

    p_rules = (
        ('foo', 'model'),
        ('bar', 'data'),
        ('baz', None))
    k = random.PRNGKey(0)
    x = jnp.ones((2, 2))
    with partitioning.axis_rules(p_rules):
      variables = ParamTest().init(k, x)
    self.assertIn('params', variables)
    self.assertIn('params_axes', variables)
    self.assertEqual(variables['params_axes']['foo_axes'],
                     partitioning.AxisMetadata(names=('foo', 'bar')))
    logical_axis_names = partitioning.get_axis_names(variables['params_axes'])
    self.assertEqual(logical_axis_names,
                     {'foo': jax.sharding.PartitionSpec('foo', 'bar')})

  def test_param_pytree_with_axes(self):
    def init_fn(k, s, d):
      del k
      return {'a': jnp.zeros(s, d), 'b': (jnp.zeros(s, d), jnp.zeros(s, d))}
    axes = {'a': ('foo', 'bar'), 'b': (('foo', 'bar'), ('bar', 'foo'))}
    class ParamTest(nn.Module):

      @nn.compact
      def __call__(self, x):
        foo = partitioning.param_with_axes(
            'foo', init_fn, (2, 2), x.dtype, axes=axes)
        return x + foo['a']

    p_rules = (
        ('foo', 'model'),
        ('bar', 'data'),
        ('baz', None))
    k = random.PRNGKey(0)
    x = jnp.ones((2, 2))
    with partitioning.axis_rules(p_rules):
      variables = ParamTest().init(k, x)
    self.assertIn('params', variables)
    self.assertIn('params_axes', variables)
    self.assertEqual(variables['params_axes']['foo_axes'],
                     partitioning.AxisMetadata(names=axes))
    logical_axis_names = partitioning.get_axis_names(variables['params_axes'])
    expected = freeze(
        {'foo':
             {'a': jax.sharding.PartitionSpec('foo', 'bar'),
              'b': (jax.sharding.PartitionSpec('foo', 'bar'),
                    jax.sharding.PartitionSpec('bar', 'foo'))}})
    self.assertEqual(logical_axis_names, expected)

  @parameterized.parameters(dict(axes_spec=None), dict(axes_spec=()))
  def test_variable_with_axes_no_axes(self, axes_spec):
    class VarTest(nn.Module):

      @nn.compact
      def __call__(self, x):
        foo = partitioning.variable_with_axes(
            'test', 'foo', jnp.zeros, (2, 2), x.dtype, axes=axes_spec)
        return x + foo.value

    k = random.PRNGKey(0)
    x = jnp.ones((2, 2))
    _ = VarTest().init(k, x)

  def test_variable_with_empty_tuple_has_empty_axes(self):

    class VarTest(nn.Module):

      @nn.compact
      def __call__(self, x):
        foo = partitioning.variable_with_axes(
            'test', 'foo', jnp.zeros, (2, 2), x.dtype, axes=())
        return x + foo.value

    k = random.PRNGKey(0)
    x = jnp.ones((2, 2))
    variables = VarTest().init(k, x)
    logical_axis_names = partitioning.get_axis_names(variables['test_axes'])
    self.assertEqual(logical_axis_names, {'foo': jax.sharding.PartitionSpec()})

  def test_variable_with_axes(self):
    class VarTest(nn.Module):

      @nn.compact
      def __call__(self, x):
        foo = partitioning.variable_with_axes(
            'test', 'foo', jnp.zeros, (2, 2), x.dtype, axes=('foo', 'bar'))
        return x + foo.value

    p_rules = (
        ('foo', 'model'),
        ('bar', 'data'),
        ('baz', None))
    k = random.PRNGKey(0)
    x = jnp.ones((2, 2))
    with partitioning.axis_rules(p_rules):
      variables = VarTest().init(k, x)
    self.assertIn('test', variables)
    self.assertIn('test_axes', variables)
    self.assertEqual(variables['test_axes']['foo_axes'],
                     partitioning.AxisMetadata(names=('foo', 'bar')))
    logical_axis_names = partitioning.get_axis_names(variables['test_axes'])
    self.assertEqual(logical_axis_names,
                     {'foo': jax.sharding.PartitionSpec('foo', 'bar')})

  @mock.patch('flax.linen.partitioning._with_sharding_constraint')
  def test_variable_with_axes_fallback(self, wsc_fn):
    class VarTest(nn.Module):

      @nn.compact
      def __call__(self, x):
        foo = partitioning.variable_with_axes(
            'test', 'foo', jnp.zeros, (2, 2), x.dtype, axes=('foo', 'bar'),
            fallback=partitioning.RulesFallback.NO_CONSTRAINT)
        return x + foo.value

    p_rules = (
        # No rule for 'foo':
        ('bar', 'data'),
        ('baz', None))
    k = random.PRNGKey(0)
    x = jnp.ones((2, 2))
    with partitioning.axis_rules(p_rules):
      variables = VarTest().init(k, x)

    wsc_fn.assert_not_called()
    self.assertIn('test', variables)
    self.assertIn('test_axes', variables)
    self.assertEqual(variables['test_axes']['foo_axes'],
                     partitioning.AxisMetadata(names=('foo', 'bar')))
    logical_axis_names = partitioning.get_axis_names(variables['test_axes'])
    self.assertEqual(logical_axis_names,
                     {'foo': jax.sharding.PartitionSpec('foo', 'bar')})

  def test_scan_with_axes(self):
    # MLP Hparams
    B, L, E = 8, 4, 32  # pylint: disable=invalid-name
    # fake inputs
    x = jnp.ones((B, E))
    k = random.PRNGKey(0)

    class SinDot(nn.Module):
      depth: int

      @nn.compact
      def __call__(self, x):
        W1 = partitioning.param_with_axes(  # pylint: disable=invalid-name
            'W1',
            nn.initializers.xavier_normal(),
            (x.shape[-1], self.depth),
            axes=('emb', 'mlp'))
        W2 = partitioning.param_with_axes(  # pylint: disable=invalid-name
            'W2',
            nn.initializers.xavier_normal(),
            (self.depth, x.shape[-1]),
            axes=('mlp', 'emb'))
        y = jnp.dot(jnp.sin(jnp.dot(x, W1)), W2)
        _ = partitioning.variable_with_axes(
            'stats', 'y_st', lambda: y, axes=('batch', 'emb'))
        # scan expects a (carry, out) return signature.
        return y, None

    class Scanned(nn.Module):
      num_layers: int
      depth: int

      @nn.compact
      def __call__(self, x):
        scanned_sindot = partitioning.scan_with_axes(
            SinDot,
            in_axes=(),
            variable_axes={'params': 0, 'stats': 1},
            split_rngs={'params': True},
            axis_name='layer',
            axes_collections=('params', 'stats'),
            length=self.num_layers)(self.depth,
                                    name='scanned_layer')
        y, _ = scanned_sindot(x)
        # test calling again to test metadata compatibility across calls
        _, _ = scanned_sindot(x)
        return y

    p_rules = (('emb', 'data'), ('mlp', 'model'), ('batch', 'data'))
    with partitioning.axis_rules(p_rules):
      variables = Scanned(L, E).init(k, x)

      # Ensure that the module can be called when 'params_axes' is not mutable.
      Scanned(L, E).apply(variables, x)
    self.assertIn('params', variables)
    self.assertIn('params_axes', variables)
    self.assertIn('stats', variables)
    self.assertIn('stats_axes', variables)
    self.assertEqual(
        variables['params_axes']['scanned_layer']['W1_axes'],
        partitioning.AxisMetadata(names=('layer', 'emb', 'mlp')))
    logical_axis_names = partitioning.get_axis_names(variables['params_axes'])
    self.assertEqual(
        logical_axis_names,
        {'scanned_layer': {
            'W1': jax.sharding.PartitionSpec('layer', 'emb', 'mlp'),
            'W2': jax.sharding.PartitionSpec('layer', 'mlp', 'emb')}})
    logical_axis_names = partitioning.get_axis_names(variables['stats_axes'])
    self.assertEqual(
        logical_axis_names,
        {'scanned_layer': {
            'y_st': jax.sharding.PartitionSpec('batch', 'layer', 'emb')}})

  def test_vmap_with_axes(self):

    class Foo(nn.Module):

      @nn.compact
      def __call__(self, x):
        return partitioning.param_with_axes(
            'w', jax.nn.initializers.uniform(), [4, 3], axes=('out', 'in')) @ x

    class Vmapped(nn.Module):

      @nn.compact
      def __call__(self, x):
        FooVmapped = partitioning.vmap_with_axes(  # pylint: disable=invalid-name
            Foo,
            variable_axes={
                'params': 1,
            },
            split_rngs={'params': True},
            partitioning_axis_names={'params': 'vmap_axis'})
        return FooVmapped(name='foo_vmapped')(x)

    p_rules = (('out', None), ('in', 'data'), ('vmap_axis', 'model'))

    # check that regular Food module is correct
    with partitioning.axis_rules(p_rules):
      variables = Foo().init(jax.random.PRNGKey(0), jnp.array([1, 2, 3]))
    variables = unfreeze(variables)
    variables['params'] = jax.tree_util.tree_map(lambda x: x.shape, variables['params'])
    self.assertDictEqual(
        variables, {
            'params': {
                'w': (4, 3)
            },
            'params_axes': {
                'w_axes': partitioning.AxisMetadata(names=('out', 'in'))
            }
        })

    # check that FooVmapped adds 'vmap_axis' to axis 1
    with partitioning.axis_rules(p_rules):
      variables = Vmapped().init(
          jax.random.PRNGKey(0), jnp.array([[1, 2, 3], [4, 5, 6]]))
    variables = unfreeze(variables)
    variables['params'] = jax.tree_util.tree_map(lambda x: x.shape, variables['params'])
    self.assertDictEqual(
        variables, {
            'params': {
                'foo_vmapped': {
                    'w': (4, 2, 3)
                }
            },
            'params_axes': {
                'foo_vmapped': {
                    'w_axes':
                        partitioning.AxisMetadata(
                            names=('out', 'vmap_axis', 'in'))
                }
            }
        })

  def test_logical_with_mesh_and_rules(self):
    devices = mesh_utils.create_device_mesh((jax.local_device_count(), 1))
    mesh = sharding.Mesh(devices, ('in', 'out'))
    test = self
    rules = (('a', 'in'), ('b', 'out'))

    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        kernel_init = nn.with_logical_partitioning(
          nn.initializers.ones_init(), ('a', 'b'), mesh=mesh, rules=rules)
        kernel = self.param('kernel', kernel_init, (x.shape[-1], 2))
        kernel_box = self.get_variable('params', 'kernel')
        test.assertIsInstance(kernel_box, nn.Partitioned)
        test.assertEqual(kernel_box.names, ('a', 'b'))
        return x @ kernel

    @jax.jit
    def create_state():
      module = Foo()
      variables = module.init(random.PRNGKey(0), jnp.zeros((8, 4)))
      logical_spec = nn.get_partition_spec(variables)
      spec = nn.logical_to_mesh(logical_spec, rules)
      shardings = jax.tree_map(lambda s: sharding.NamedSharding(mesh, s), spec)
      variables = jax.lax.with_sharding_constraint(variables, shardings)
      return variables


    variables = create_state()
    self.assertEqual(variables['params']['kernel'].names,
                     ('a', 'b'))
    self.assertIs(variables['params']['kernel'].mesh, mesh)
    self.assertEqual(variables['params']['kernel'].rules, rules)

if __name__ == '__main__':
  absltest.main()
