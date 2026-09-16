# Copyright 2023 The Flax Authors.
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

import dataclasses
import types
from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from flax.linen.experimental.params import param
import jax
from jax import numpy as jnp
import numpy as np


class SchemaTest(absltest.TestCase):

  def test_no_extra(self):
    s = param.param_schema(
        first=param.Attr, other=param.Given
    )

    self.assertEqual(dataclasses.MISSING, s.default)
    self.assertEqual(param.Param(), s.default_factory())
    expected_schema = types.MappingProxyType(
        {'first': param.Attr(), 'other': param.Given()}
    )

    self.assertEqual(expected_schema, param.get_param_schema(s))

  def test_custom_init(self):
    def custom_init(rng, shape, dtype):
      raise RuntimeError('I should never be called.')

    s = param.param_schema(
        custom_init, seq_len=param.Given, d=param.Attr('d_model')
    )

    self.assertEqual(dataclasses.MISSING, s.default)
    self.assertEqual(
        param.Param(initializer=custom_init), s.default_factory()
    )
    expected_schema = types.MappingProxyType(
        {'seq_len': param.Given(), 'd': param.Attr('d_model')}
    )
    self.assertEqual(expected_schema, param.get_param_schema(s))

  def test_custom_spec(self):
    s = param.param_schema(
        param.Param(dtype=jnp.bfloat16),
        features=param.Attr,
        input=param.Given,
    )

    self.assertEqual(dataclasses.MISSING, s.default)
    self.assertEqual(
        param.Param(dtype=jnp.bfloat16), s.default_factory()
    )
    expected_schema = types.MappingProxyType(
        {'features': param.Attr(), 'input': param.Given()}
    )
    self.assertEqual(expected_schema, param.get_param_schema(s))

  def test_custom_field(self):
    s = param.param_schema(
        dataclasses.field(kw_only=True, repr=False),
        qkv=param.Const(3),
        input_dim=param.Given,
        num_heads=param.Attr,
        dim_per_head=param.Attr('d_model'),
    )

    self.assertEqual(dataclasses.MISSING, s.default)
    self.assertTrue(s.kw_only)
    self.assertFalse(s.repr)
    self.assertEqual(param.Param(), s.default_factory())

    expected_schema = types.MappingProxyType({
        'qkv': param.Const(3),
        'input_dim': param.Given(),
        'num_heads': param.Attr(),
        'dim_per_head': param.Attr('d_model'),
    })
    self.assertEqual(expected_schema, param.get_param_schema(s))

  def test_custom_field_with_custom_spec(self):
    s = param.param_schema(
        dataclasses.field(
            default=param.Param(dtype=jnp.bfloat16), kw_only=True
        ),
        features=param.Attr,
    )

    self.assertEqual(dataclasses.MISSING, s.default)
    self.assertTrue(s.kw_only)
    self.assertEqual(
        param.Param(dtype=jnp.bfloat16), s.default_factory()
    )

  def test_schema_int(self):
    s = param.param_schema(k_x=3, k_y=3)

    expected_schema = types.MappingProxyType(
        {'k_x': param.Const(3), 'k_y': param.Const(3)}
    )
    self.assertEqual(expected_schema, param.get_param_schema(s))

  def test_schema_unexpected_value(self):
    with self.assertRaises(TypeError):
      param.param_schema(custom_dim='weird value')  # pytype: disable=wrong-arg-types

  def test_schema_negative_int(self):
    with self.assertRaises(ValueError):
      param.param_schema(k_x=-1)

    with self.assertRaises(ValueError):
      param.param_schema(k_y=param.Const(-42))


class SimpleModule(nn.Module):
  features: int = 42
  simple: param.Param = param.param_schema(
      features=param.Attr,
      input=param.Given,
  )

  def __call__(self, x):
    simple = self.simple(input=x.shape[-1])
    return simple


class ParamSpecTest(absltest.TestCase):
  def test_simple_schema(self):
    model = SimpleModule()
    params = model.init(jax.random.PRNGKey(42), jnp.zeros((10, 128)))
    self.assertEqual((42, 128), params['params']['simple']['w'].shape)
    self.assertEqual((42, 128), model.apply(params, jnp.zeros((10, 128))).shape)

    model = SimpleModule(features=24)
    params = model.init(jax.random.PRNGKey(42), jnp.zeros((18, 256)))
    self.assertEqual((24, 256), params['params']['simple']['w'].shape)
    self.assertEqual((24, 256), model.apply(params, jnp.zeros((18, 256))).shape)


class SequenceDiffTest(absltest.TestCase):

  def test_equivalent_sequences(self):
    self.assertIsNone(
        param._sequence_diff(set(['a', 'b']), set(['b', 'a']))
    )
    self.assertIsNone(param._sequence_diff(set(['a']), set(['a'])))
    self.assertIsNone(param._sequence_diff(set(), set()))

  def test_simple_extra(self):
    expected = param.SequenceDifferences(extra=('extra',), missing=())
    self.assertEqual(
        expected,
        param._sequence_diff(expected=set(), actual=set(['extra'])),
    )

  def test_simple_missing(self):
    expected = param.SequenceDifferences(extra=(), missing=('missing',))
    self.assertEqual(
        expected,
        param._sequence_diff(expected=set(['missing']), actual=set()),
    )

  def test_typo(self):
    expected = param.SequenceDifferences(
        extra=('typo',), missing=('typ',)
    )
    self.assertEqual(
        expected,
        param._sequence_diff(
            expected=set(['common', 'also', 'typ', 'shared']),
            actual=set(['common', 'typo', 'also', 'shared']),
        ),
    )


def eye_initializer(
    rng: jax.random.KeyArray, shape: Sequence[int], dtype
) -> jax.Array:
  del rng
  assert len(shape) == 2, f'{shape}'
  assert shape[0] == shape[1], f'{shape}'
  return jnp.eye(shape[0], dtype=dtype)


class EinsumTest(parameterized.TestCase):

  @parameterized.parameters(
      ('ab,bc->ac', {'b': -1}),
      ('ab,cb->ac', {'b': -1}),
      ('ba,cb->ac', {'b': -2}),
      ('abc,bcd->ad', {'b': -2, 'c': -1}),
      ('...td,cdD->c...tD', {'d': -1}),
      ('...td,chdD->c...thD', {'d': -1}),
      ('...thd,hdD->...tD', {'d': -1, 'h': -2}),
  )
  def test_simple_names_to_input_axes(
      self, computation: str, expected_axis: dict[str, int]
  ):
    self.assertEqual(
        param._names_to_input_axes(computation), expected_axis
    )

  def test_simple_block(self):
    class SimpleBlock(nn.Module):
      eye: param.Einsum = param.einsum(
          'ab,bc->ac',
          eye_initializer,
          c=2)

      def __call__(self, x):
        return self.eye(x)

    block = SimpleBlock()
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    variables = block.init(jax.random.PRNGKey(42), x)

    self.assertEqual((2, 2), variables['params']['eye']['w'].shape)

    y = block.apply(variables, x)
    np.testing.assert_allclose(x, np.array(y))

  def test_attr_access(self):
    class SimpleBlock(nn.Module):
      features: int
      eye: param.Einsum = param.einsum(
          'ab,bc->ac',
          eye_initializer,
          c=param.Attr(from_name='features')
      )

      def __call__(self, x):
        return self.eye(x)

    block = SimpleBlock(features=5)
    batch_size = 10
    input_features_size = 5
    x = np.zeros((batch_size, input_features_size))
    variables = block.init(jax.random.PRNGKey(42), x)

    self.assertEqual((5, 5), variables['params']['eye']['w'].shape)
    self.assertEqual((batch_size, 5), block.apply(variables, x).shape)

    # Switch up the # features.
    block = SimpleBlock(features=10)
    batch_size = 8
    input_features_size = 10
    x = np.zeros((batch_size, input_features_size))
    variables = block.init(jax.random.PRNGKey(43), x)

    self.assertEqual((10, 10), variables['params']['eye']['w'].shape)
    self.assertEqual((batch_size, 10), block.apply(variables, x).shape)


if __name__ == '__main__':
  absltest.main()
