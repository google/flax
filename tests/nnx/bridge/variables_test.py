# Copyright 2026 The Flax Authors.
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
from unittest import mock

os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

from absl.testing import absltest, parameterized
from flax import errors, linen as nn, nnx
from flax.nnx import spmd
from flax.nnx.bridge import variables as bv
import jax
import jax.numpy as jnp
import numpy as np


class AxisMetadataTest(parameterized.TestCase):
  @parameterized.named_parameters(
    ('ordinary', False, False), ('ref', False, True), ('hijax', True, False)
  )
  def test_variable_kind_preserved(self, hijax, ref):
    variable = nnx.Param(
      jnp.ones((2,)),
      out_sharding=('in',),
      eager_sharding=False,
      hijax=hijax,
      ref=ref,
    )
    original = bv.to_linen_var(variable)
    self.assertIs(original.var_type, nnx.Param)
    params = {nn.meta.PARTITION_NAME: 'layers'}
    added = original.add_axis(0, params)
    self.assertEqual(added.metadata['out_sharding'], ('layers', 'in'))
    self.assertEqual(added.metadata['hijax'], hijax)
    self.assertEqual(added.metadata['ref'], ref)
    self.assertIs(added.value, original.value)
    self.assertEqual(added.remove_axis(0, params).metadata, original.metadata)

  @parameterized.product(axis=(0, 1, 2), axis_name=('layers', None))
  def test_axis_roundtrip(self, axis, axis_name):
    variable = nnx.Param(
      jnp.ones((2, 3)), out_sharding=('in', 'out'), eager_sharding=False
    )
    original = bv.to_linen_var(variable)
    params = {nn.meta.PARTITION_NAME: axis_name}
    added = original.add_axis(axis, params)
    expected = ['in', 'out']
    expected.insert(axis, axis_name)
    self.assertEqual(added.metadata['out_sharding'], tuple(expected))
    self.assertEqual(original.metadata['out_sharding'], ('in', 'out'))
    self.assertIs(added.value, original.value)
    self.assertIsNot(added, original)
    self.assertEqual(params, {nn.meta.PARTITION_NAME: axis_name})
    removed = added.remove_axis(axis, params)
    self.assertEqual(removed.metadata, original.metadata)
    self.assertIs(removed.value, original.value)

  def test_remove_existing_axis(self):
    original = bv.to_linen_var(
      nnx.Param(
        jnp.ones((3, 2, 4)),
        out_sharding=('layers', 'in', 'out'),
        eager_sharding=False,
      )
    )
    removed = original.remove_axis(0, {nn.meta.PARTITION_NAME: 'layers'})
    self.assertEqual(removed.metadata['out_sharding'], ('in', 'out'))
    self.assertEqual(original.metadata['out_sharding'], ('layers', 'in', 'out'))
    with self.assertRaisesRegex(ValueError, 'Expected to remove'):
      original.remove_axis(0, {nn.meta.PARTITION_NAME: 'wrong'})

  @parameterized.parameters('add_axis', 'remove_axis')
  def test_partition_name_required(self, method):
    original = bv.to_linen_var(
      nnx.Param(
        jnp.ones((2,)),
        out_sharding=('in',),
        eager_sharding=False,
      )
    )
    with self.assertRaises(errors.PartitioningUnspecifiedError):
      getattr(original, method)(0, {})

  @parameterized.named_parameters(('empty', ()), ('nonempty', ('in', 'out')))
  def test_additional_axis_metadata(self, names):
    original = bv.to_linen_var(
      nnx.Param(
        jnp.array(1.0),
        nickname=names,
        unrelated='unchanged',
      )
    )
    # No partition_name is necessary for an unsharded variable.
    params = {'nickname': 'layer'}
    added = original.add_axis(0, params)
    self.assertEqual(added.metadata['nickname'], ('layer', *names))
    self.assertEqual(added.metadata['unrelated'], 'unchanged')
    self.assertEqual(added.remove_axis(0, params).metadata, original.metadata)
    self.assertEqual(params, {'nickname': 'layer'})

  @parameterized.parameters(False, True)
  def test_empty_sharding(self, through_bridge):
    variable = nnx.Param(jnp.array(1.0), out_sharding=(), eager_sharding=False)
    params = {nn.meta.PARTITION_NAME: 'layers'}
    if through_bridge:
      original = bv.to_linen_var(variable)
      added = original.add_axis(0, params)
      self.assertEqual(added.metadata['out_sharding'], ('layers',))
      self.assertEqual(added.remove_axis(0, params).metadata, original.metadata)
    else:
      spmd.add_axis(variable, 0, params)
      self.assertEqual(variable.out_sharding, ('layers',))
      spmd.remove_axis(variable, 0, params)
      self.assertEqual(variable.out_sharding, ())

  @parameterized.product(named=(False, True), tuple_hooks=(False, True))
  def test_hooks(self, named, tuple_hooks):
    add, remove = mock.Mock(), mock.Mock()
    with self.assertWarnsRegex(UserWarning, 'Variable hooks are deprecated'):
      variable = nnx.Param(
        jnp.ones((2,)),
        on_add_axis=(add,) if tuple_hooks else add,
        on_remove_axis=(remove,) if tuple_hooks else remove,
      )
    original = bv.to_linen_var(variable)
    params = {nn.meta.PARTITION_NAME: 'layers'} if named else {}
    added = original.add_axis(0, params)
    added.remove_axis(0, params)
    self.assertEqual(add.call_count, 1)
    self.assertEqual(remove.call_count, 1)
    expected_name = 'layers' if named else None
    self.assertEqual(add.call_args.args[1:], (0, expected_name))
    self.assertEqual(remove.call_args.args[1:], (0, expected_name))

  def test_custom_variable_axis_methods(self):
    class CustomParam(nnx.Param):
      def add_axis(self, index, name):
        names = list(self.names)
        names.insert(index, name)
        self.set_metadata(names=tuple(names))

      def remove_axis(self, index, name):
        names = list(self.names)
        assert names.pop(index) == name
        self.set_metadata(names=tuple(names))

    original = bv.to_linen_var(CustomParam(jnp.ones((2,)), names=('in',)))
    params = {nn.meta.PARTITION_NAME: 'layers'}
    added = original.add_axis(0, params)
    self.assertIs(added.var_type, CustomParam)
    self.assertEqual(added.metadata['names'], ('layers', 'in'))
    self.assertEqual(original.metadata['names'], ('in',))
    self.assertEqual(added.remove_axis(0, params).metadata, original.metadata)

  def test_no_initialization_or_value_hooks(self):
    create = mock.Mock(side_effect=lambda v, x: x + 1)
    get = mock.Mock(side_effect=lambda v, x: x + 2)
    set_value = mock.Mock(side_effect=lambda v, x: x)
    with self.assertWarnsRegex(UserWarning, 'Variable hooks are deprecated'):
      original = bv.to_linen_var(
        nnx.Param(
          jnp.ones((2,)),
          on_create_value=create,
          on_get_value=get,
          on_set_value=set_value,
          out_sharding=('in',),
          eager_sharding=False,
        )
      )
    create.reset_mock()
    get.reset_mock()
    params = {nn.meta.PARTITION_NAME: 'layers'}
    with mock.patch.object(nnx.Param, '__init__', side_effect=AssertionError):
      result = original.add_axis(0, params).remove_axis(0, params)
    create.assert_not_called()
    get.assert_not_called()
    set_value.assert_not_called()
    self.assertIs(result.value, original.value)


class Dense(nnx.Module):
  def __init__(self, *, rngs, sharding=('in', 'out')):
    self.kernel = nnx.Param(
      jax.random.normal(rngs.params(), (4, 4)), out_sharding=sharding
    )

  def __call__(self, x):
    return x @ self.kernel[...]


class Cell(Dense):
  def __call__(self, carry, x):
    y = super().__call__(carry + x)
    return y, y


class AxisTransformTest(parameterized.TestCase):
  @parameterized.product(kind=('vmap', 'scan'), axis=(0, 1, 2, -1, -2, -3))
  def test_transform_with_mesh_and_jit(self, kind, axis):
    if jax.device_count() < 4:
      self.skipTest('At least 4 devices required')
    mesh = jax.sharding.Mesh(
      np.array(jax.devices()[:4]).reshape(1, 2, 2), ('layers', 'in', 'out')
    )
    options = dict(
      variable_axes={'params': axis},
      split_rngs={'params': True},
      metadata_params={nn.meta.PARTITION_NAME: 'layers'},
    )
    if kind == 'vmap':
      model = nn.vmap(nnx.bridge.ToLinen, **options)(Dense)
      args = (jnp.ones((3, 4)),)
    else:
      model = nn.scan(nnx.bridge.ToLinen, length=3, **options)(Cell)
      args = (jnp.ones((4,)), jnp.ones((3, 4)))
    with jax.set_mesh(mesh):
      output, variables = model.init_with_output(jax.random.key(0), *args)
      actual = jax.jit(model.apply)(variables, *args)
      spec = nn.get_partition_spec(variables)['params']['kernel']
    kernel = variables['params']['kernel']
    expected_names = ['in', 'out']
    expected_names.insert(axis % 3, 'layers')
    expected_shape = [4, 4]
    expected_shape.insert(axis % 3, 3)
    self.assertEqual(kernel.value.shape, tuple(expected_shape))
    self.assertEqual(kernel.metadata['out_sharding'], tuple(expected_names))
    self.assertEqual(spec, jax.sharding.PartitionSpec(*expected_names))
    weights = jnp.moveaxis(kernel.value, axis, 0)
    if kind == 'vmap':

      def reference(w):
        return jnp.einsum('bi,bij->bj', args[0], w)
    else:

      def reference(w):
        carry = args[0]
        ys = []
        for i in range(3):
          carry = (carry + args[1][i]) @ w[i]
          ys.append(carry)
        return carry, jnp.stack(ys)

    expected = reference(weights)
    for result in (output, actual):
      for a, b in zip(jax.tree.leaves(result), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)

    def total(output):
      return sum(jnp.sum(leaf) for leaf in jax.tree.leaves(output))

    with jax.set_mesh(mesh):
      gradients = jax.jit(jax.grad(lambda vs: total(model.apply(vs, *args))))(
        variables
      )
    gradient = gradients['params']['kernel']
    expected_gradient = jax.grad(lambda w: total(reference(w)))(weights)
    self.assertEqual(gradient.metadata, kernel.metadata)
    np.testing.assert_allclose(
      jnp.moveaxis(gradient.value, axis, 0),
      expected_gradient,
      atol=1e-5,
      rtol=1e-5,
    )

  def test_scalar_parameter_vmap(self):
    class Scalar(nnx.Module):
      def __init__(self, *, rngs):
        self.weight = nnx.Param(
          jax.random.normal(rngs.params(), ()),
          out_sharding=(),
          eager_sharding=False,
        )

      def __call__(self, x):
        return x * self.weight[...]

    model = nn.vmap(
      nnx.bridge.ToLinen,
      variable_axes={'params': 0},
      split_rngs={'params': True},
      metadata_params={nn.meta.PARTITION_NAME: 'layers'},
    )(Scalar)
    x = jnp.arange(3, dtype=jnp.float32)
    variables = model.init(jax.random.key(0), x)
    weight = variables['params']['weight']
    self.assertEqual(weight.value.shape, (3,))
    self.assertEqual(weight.metadata['out_sharding'], ('layers',))
    self.assertEqual(
      nn.get_partition_spec(variables)['params']['weight'],
      jax.sharding.PartitionSpec('layers'),
    )
    np.testing.assert_allclose(
      jax.jit(model.apply)(variables, x), x * weight.value
    )

  def test_nested_vmap(self):
    inner = nn.vmap(
      nnx.bridge.ToLinen,
      variable_axes={'params': 0},
      split_rngs={'params': True},
      metadata_params={nn.meta.PARTITION_NAME: 'inner'},
    )
    outer = nn.vmap(
      inner,
      variable_axes={'params': 0},
      split_rngs={'params': True},
      metadata_params={nn.meta.PARTITION_NAME: 'outer'},
    )
    model = outer(Dense)
    x = jnp.ones((2, 3, 4))
    with nnx.use_eager_sharding(False):
      variables = model.init(jax.random.key(0), x)
      y = jax.jit(model.apply)(variables, x)
    kernel = variables['params']['kernel']
    self.assertEqual(kernel.value.shape, (2, 3, 4, 4))
    self.assertEqual(
      kernel.metadata['out_sharding'], ('outer', 'inner', 'in', 'out')
    )
    np.testing.assert_allclose(
      y, jnp.einsum('abi,abij->abj', x, kernel.value), atol=1e-5, rtol=1e-5
    )


if __name__ == '__main__':
  absltest.main()
