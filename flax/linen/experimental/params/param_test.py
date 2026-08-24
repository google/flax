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

from absl.testing import absltest
from flax import linen as nn
from flax.linen.experimental.params import param
from jax import numpy as jnp
import jax
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


if __name__ == '__main__':
  absltest.main()
