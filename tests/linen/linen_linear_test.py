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

"""Tests for flax.linen.linear."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax import random
from jax.nn import initializers

from flax import linen as nn

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class LinearTest(parameterized.TestCase):
  def test_dense(self):
    rng = dict(params=random.key(0))
    x = jnp.ones((1, 3))
    dense_module = nn.Dense(
      features=4,
      kernel_init=initializers.ones,
      bias_init=initializers.ones,
    )
    y, _ = dense_module.init_with_output(rng, x)
    self.assertEqual(y.shape, (1, 4))
    self.assertEqual(y.dtype, jnp.float32)
    np.testing.assert_allclose(y, np.full((1, 4), 4.0))

  def test_dense_extra_batch_dims(self):
    rng = dict(params=random.key(0))
    x = jnp.ones((1, 2, 3))
    dense_module = nn.Dense(
      features=4,
      kernel_init=initializers.ones,
      bias_init=initializers.ones,
    )
    y, _ = dense_module.init_with_output(rng, x)
    np.testing.assert_allclose(y, np.full((1, 2, 4), 4.0))

  def test_dense_no_bias(self):
    rng = dict(params=random.key(0))
    x = jnp.ones((1, 3))
    dense_module = nn.Dense(
      features=4,
      use_bias=False,
      kernel_init=initializers.ones,
    )
    y, _ = dense_module.init_with_output(rng, x)
    np.testing.assert_allclose(y, np.full((1, 4), 3.0))

  def test_dense_is_dense_general(self):
    x = jax.random.normal(random.key(0), (5, 3))
    dense_module = nn.Dense(
      features=4,
      use_bias=True,
      bias_init=initializers.normal(),
    )
    y1, _ = dense_module.init_with_output(dict(params=random.key(1)), x)
    dg_module = nn.DenseGeneral(
      features=4,
      use_bias=True,
      bias_init=initializers.normal(),
    )
    y2, _ = dg_module.init_with_output(dict(params=random.key(1)), x)

    np.testing.assert_allclose(y1, y2)

  def test_dense_general_batch_dim_raises(self):
    rng = dict(params=random.key(0))
    x = jnp.ones((1, 3, 2, 5))
    with self.assertRaises(ValueError):
      dg_module = nn.DenseGeneral(
        features=4,
        batch_dims=(0, 2),
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
      )
      dg_module.init_with_output(rng, x)

  def test_dense_general_two_out(self):
    rng = dict(params=random.key(0))
    x = jnp.ones((1, 3))
    dg_module = nn.DenseGeneral(
      features=(2, 2),
      kernel_init=initializers.ones,
      bias_init=initializers.ones,
    )
    y, _ = dg_module.init_with_output(rng, x)
    np.testing.assert_allclose(y, np.full((1, 2, 2), 4.0))

  def test_dense_general_two_in(self):
    rng = dict(params=random.key(0))
    x = jnp.ones((1, 2, 2))
    dg_module = nn.DenseGeneral(
      features=3,
      axis=(-2, 2),
      kernel_init=initializers.ones,
      bias_init=initializers.ones,
    )
    y, _ = dg_module.init_with_output(rng, x)
    np.testing.assert_allclose(y, np.full((1, 3), 5.0))

  def test_dense_general_batch_dim(self):
    rng = dict(params=random.key(0))
    x = jnp.ones((2, 1, 3, 5))

    state = {'counter': 0.0}

    def _counter_init(rng, shape, dtype, state):
      del rng, dtype
      state['counter'] += 1.0
      return jnp.full(shape, state['counter'])

    counter_init = functools.partial(_counter_init, state=state)

    dg_module = nn.DenseGeneral(
      features=7,
      axis=(3, -2),
      batch_dims=0,
      bias_init=initializers.ones,
      kernel_init=counter_init,
    )
    y, _ = dg_module.init_with_output(rng, x)
    target = np.full((2, 1, 7), 16.0)
    np.testing.assert_allclose(y, target)

  @parameterized.parameters(
    [
      ((-2, 3), (), 'bijk,jklm->bilm'),
      ((3, -2), (), 'bijk,jklm->bilm'),
      ((-2, 3), (0,), 'bijk,bjklm->bilm'),
    ]
  )
  def test_dense_general_vs_numpy(self, axis, batch_dims, einsum_expr):
    rng = dict(params=random.key(0))
    x = jnp.ones((16, 8, 9, 10))

    dg_module = nn.DenseGeneral(
      features=(11, 12),
      axis=axis,
      batch_dims=batch_dims,
      bias_init=initializers.ones,
      kernel_init=initializers.normal(),
    )
    y, initial_params = dg_module.init_with_output(rng, x)
    target = np.einsum(einsum_expr, x, initial_params['params']['kernel']) + 1.0
    np.testing.assert_allclose(y, target, atol=1e-6)

  def test_complex_params_dense(self):
    dense = nn.Dense(features=2, param_dtype=jnp.complex64)
    x = jnp.ones((1, 2), jnp.float32)
    variables = dense.init(random.key(0), x)
    self.assertEqual(variables['params']['kernel'].dtype, jnp.complex64)
    self.assertEqual(variables['params']['bias'].dtype, jnp.complex64)
    y = dense.apply(variables, x)
    self.assertEqual(y.dtype, jnp.complex64)

  def test_complex_input_dense(self):
    dense = nn.Dense(features=2)
    x = jnp.ones((1, 2), jnp.complex64)
    variables = dense.init(random.key(0), x)
    self.assertEqual(variables['params']['kernel'].dtype, jnp.float32)
    self.assertEqual(variables['params']['bias'].dtype, jnp.float32)
    y = dense.apply(variables, x)
    self.assertEqual(y.dtype, jnp.complex64)

  @parameterized.parameters(
    [
      (
        'abc,cde->abde',
        (3, 4, 5),
        (5, 6, 7),
        (3, 4, 6, 7),
        (6, 7),
        (1, 1, 6, 7),
      ),
      (
        'abcd,abcd->abcd',
        (3, 4, 5, 6),
        (3, 4, 5, 6),
        (3, 4, 5, 6),
        (3, 4, 5, 6),
        (3, 4, 5, 6),
      ),
      (
        'abcd,abcd->abd',
        (3, 4, 5, 6),
        (3, 4, 5, 6),
        (3, 4, 6),
        (3, 4, 6),
        (3, 4, 6),
      ),
      (
        'abcd,cdef->abef',
        (3, 4, 5, 6),
        (5, 6, 7, 8),
        (3, 4, 7, 8),
        (7, 8),
        (1, 1, 7, 8),
      ),
      (
        'abcd,eafc->bfed',
        (3, 4, 5, 6),
        (7, 3, 8, 5),
        (4, 8, 7, 6),
        (8, 7),
        (1, 8, 7, 1),
      ),
      (
        'abcd,cbedf->abfe',
        (3, 4, 5, 6),
        (5, 4, 7, 6, 8),
        (3, 4, 8, 7),
        (4, 8, 7),
        (1, 4, 8, 7),
      ),
      (
        'ab...,bc...->ac...',
        (3, 4, 6),
        (4, 5, 6),
        (3, 5, 6),
        (5, 6),
        (1, 5, 6),
      ),
      (
        'd...ab,bc...->ad...c',
        (8, 6, 7, 3, 4),
        (4, 5, 6, 7),
        (3, 8, 6, 7, 5),
        (6, 7, 5),
        (1, 1, 6, 7, 5),
      ),
      (
        'd...ab,bc...->adc',
        (8, 6, 7, 3, 4),
        (4, 5, 6, 7),
        (3, 8, 5),
        (5,),
        (1, 1, 5),
      ),
      (
        'abd...,bc...->ac...',
        (3, 4, 6),
        (4, 5, 6),
        (3, 5, 6),
        (5, 6),
        (1, 5, 6),
      ),
      (
        'a...d,ej...f->adef',
        (3, 4, 5, 6),
        (7, 4, 5, 8),
        (3, 6, 7, 8),
        (7, 8),
        (1, 1, 7, 8),
      ),
      (
        'ab...d,ej...f->ad...f',
        (3, 4, 5, 6),
        (7, 4, 5, 8),
        (3, 6, 5, 8),
        (5, 8),
        (1, 1, 5, 8),
      ),
    ]
  )
  def test_einsum_init_apply(
    self,
    einsum_str,
    lhs_shape,
    rhs_shape,
    expected_result_shape,
    expected_bias_shape,
    bias_broadcast_shape,
  ):
    layer = nn.Einsum(rhs_shape, einsum_str, bias_init=nn.initializers.normal())
    x = jax.random.normal(jax.random.key(0), lhs_shape)

    v = layer.init(jax.random.key(1), x)
    self.assertEqual(rhs_shape, v['params']['kernel'].shape)
    self.assertEqual(expected_bias_shape, v['params']['bias'].shape)

    out = layer.apply(v, x)
    self.assertEqual(out.shape, expected_result_shape)
    expected_out = jnp.einsum(einsum_str, x, v['params']['kernel']) + v[
      'params'
    ]['bias'].reshape(bias_broadcast_shape)
    np.testing.assert_allclose(out, expected_out)

  @parameterized.parameters(
    [
      (
        ('abd,bce->ace', 'abd,bc...->ac...', 'abd...,bc...->ac...'),
        (3, 4, 6),
        (4, 5, 6),
      ),
      (
        (
          'abcd,ejcf->adef',
          'ab...d,ej...f->adef',
          'ab...d,e...f->adef',
          'a...d,ej...f->adef',
        ),
        (3, 4, 5, 6),
        (7, 4, 5, 8),
      ),
      (
        ('abcd,ejcf->adcf', 'ab...d,ej...f->ad...f'),
        (3, 4, 5, 6),
        (7, 4, 5, 8),
      ),
    ]
  )
  def test_einsum_ellipsis_equivalence(
    self, einsum_str_list, lhs_shape, rhs_shape
  ):
    x = jax.random.uniform(jax.random.key(0), lhs_shape)
    layer = nn.Einsum(
      rhs_shape, einsum_str_list[0], bias_init=nn.initializers.normal()
    )
    v = layer.init(jax.random.key(1), x)
    out = layer.apply(v, x)

    for einsum_str in einsum_str_list[1:]:
      layer2 = nn.Einsum(
        rhs_shape, einsum_str, bias_init=nn.initializers.normal()
      )
      v2 = layer2.init(jax.random.key(1), x)
      np.testing.assert_allclose(v['params']['kernel'], v2['params']['kernel'])
      np.testing.assert_allclose(v['params']['bias'], v2['params']['bias'])
      np.testing.assert_allclose(out, layer2.apply(v2, x))

  def test_einsum_str_arg(self):
    einsum_str = 'abc,cde->abde'
    x = jax.random.normal(jax.random.key(0), (3, 4, 5))

    constructed_layer = nn.Einsum(
      (5, 6, 7), einsum_str, bias_init=nn.initializers.normal()
    )
    constructed_v = constructed_layer.init(jax.random.key(1), x)
    constructed_out = constructed_layer.apply(constructed_v, x)

    called_layer = nn.Einsum((5, 6, 7), bias_init=nn.initializers.normal())
    called_v = called_layer.init(jax.random.key(1), x, einsum_str)
    called_out = called_layer.apply(called_v, x, einsum_str)

    np.testing.assert_allclose(
      constructed_v['params']['kernel'], called_v['params']['kernel']
    )
    np.testing.assert_allclose(
      constructed_v['params']['bias'], called_v['params']['bias']
    )
    np.testing.assert_allclose(constructed_out, called_out)

    with self.assertRaisesWithLiteralMatch(
      ValueError,
      'Parameter "einsum_str" was passed to the constructor and at call time. Should be passed just once.',
    ):
      constructed_layer.init(jax.random.key(1), x, einsum_str)
    with self.assertRaisesWithLiteralMatch(
      ValueError,
      'Parameter "einsum_str" must be passed to the constructor or at call time.',
    ):
      called_layer.init(jax.random.key(1), x)

  def test_einsum_space_str(self):
    x = jax.random.normal(jax.random.key(0), (3, 4, 5))

    layer1 = nn.Einsum(
      (5, 6, 7), 'abc,cde->abde', bias_init=nn.initializers.normal()
    )
    v1 = layer1.init(jax.random.key(1), x)
    out1 = layer1.apply(v1, x)

    layer2 = nn.Einsum(
      (5, 6, 7),
      '  ab c  , c d e    -  >a b d  e  ',
      bias_init=nn.initializers.normal(),
    )
    v2 = layer2.init(jax.random.key(1), x)
    out2 = layer2.apply(v1, x)

    np.testing.assert_allclose(v1['params']['kernel'], v2['params']['kernel'])
    np.testing.assert_allclose(v1['params']['bias'], v2['params']['bias'])
    np.testing.assert_allclose(out1, out2)

  @parameterized.parameters(
    [
      ('abc,cde', '`einsum_str` equation must be explicit and include "->".'),
      (
        'abc->',
        f'`einsum_str` equation must have exactly two operands and therefore, exactly one comma character, instead of 0',
      ),
      (
        'abc,cde,efg->abdfg',
        f'`einsum_str` equation must have exactly two operands and therefore, exactly one comma character, instead of 2',
      ),
    ]
  )
  def test_einsum_error(self, einsum_str, error_msg):
    x = jax.random.normal(jax.random.key(0), (3, 4, 5))

    layer = nn.Einsum((5, 6, 7), einsum_str)
    with self.assertRaisesRegex(ValueError, error_msg):
      layer.init(jax.random.key(1), x)

  @parameterized.product(use_bias=(True, False))
  def test_conv(self, use_bias):
    rng = dict(params=random.key(0))
    x = jnp.ones((1, 8, 3))
    conv_module = nn.Conv(
      features=4,
      use_bias=use_bias,
      kernel_size=(3,),
      padding='VALID',
      kernel_init=initializers.ones,
      bias_init=initializers.ones,
    )
    y, initial_params = conv_module.init_with_output(rng, x)
    self.assertEqual(initial_params['params']['kernel'].shape, (3, 3, 4))
    expected = 10.0 if use_bias else 9.0
    np.testing.assert_allclose(y, np.full((1, 6, 4), expected))

  @parameterized.product(use_bias=(True, False))
  def test_multibatch_input_conv(self, use_bias):
    rng = dict(params=random.key(0))
    x = jnp.ones((2, 5, 8, 3))
    conv_module = nn.Conv(
      features=4,
      use_bias=use_bias,
      kernel_size=(3,),
      padding='VALID',
      kernel_init=initializers.ones,
      bias_init=initializers.ones,
    )
    y, initial_params = conv_module.init_with_output(rng, x)
    self.assertEqual(initial_params['params']['kernel'].shape, (3, 3, 4))
    expected = 10.0 if use_bias else 9.0
    np.testing.assert_allclose(y, np.full((2, 5, 6, 4), expected))

  def test_conv_local(self):
    rng = dict(params=random.key(0))
    x = jnp.ones((1, 8, 2))
    conv_module = nn.ConvLocal(
      features=4,
      kernel_size=(3,),
      padding='VALID',
      kernel_init=initializers.ones,
      bias_init=initializers.ones,
    )
    y, initial_params = conv_module.init_with_output(rng, x)
    self.assertEqual(initial_params['params']['kernel'].shape, (6, 3 * 2, 4))
    np.testing.assert_allclose(y, np.full((1, 6, 4), 7.0))

  def test_single_input_conv(self):
    rng = dict(params=random.key(0))
    x = jnp.ones((8, 3))
    conv_module = nn.Conv(
      features=4,
      kernel_size=(3,),
      padding='VALID',
      kernel_init=initializers.ones,
      bias_init=initializers.ones,
    )
    y, initial_params = conv_module.init_with_output(rng, x)
    self.assertEqual(initial_params['params']['kernel'].shape, (3, 3, 4))
    np.testing.assert_allclose(y, np.full((6, 4), 10.0))

  def test_single_input_masked_conv(self):
    rng = dict(params=random.key(0))
    x = jnp.ones((8, 3))
    m = jnp.tril(jnp.ones((3, 3, 4)))
    conv_module = nn.Conv(
      features=4,
      kernel_size=(3,),
      padding='VALID',
      mask=m,
      kernel_init=initializers.ones,
      bias_init=initializers.ones,
    )
    expected = jnp.array(
      [
        [10.0, 7.0, 4.0, 1.0],
        [10.0, 7.0, 4.0, 1.0],
        [10.0, 7.0, 4.0, 1.0],
        [10.0, 7.0, 4.0, 1.0],
        [10.0, 7.0, 4.0, 1.0],
        [10.0, 7.0, 4.0, 1.0],
      ]
    )
    y, initial_params = conv_module.init_with_output(rng, x)
    self.assertEqual(initial_params['params']['kernel'].shape, (3, 3, 4))
    np.testing.assert_allclose(y, expected)

  def test_single_input_conv_local(self):
    rng = dict(params=random.key(0))
    x = jnp.ones((8, 2))
    conv_module = nn.ConvLocal(
      features=4,
      kernel_size=(3,),
      padding='VALID',
      kernel_init=initializers.ones,
      bias_init=initializers.ones,
    )
    y, initial_params = conv_module.init_with_output(rng, x)
    self.assertEqual(initial_params['params']['kernel'].shape, (6, 3 * 2, 4))
    np.testing.assert_allclose(y, np.full((6, 4), 7.0))

  def test_group_conv(self):
    rng = dict(params=random.key(0))
    x = jnp.ones((1, 8, 4))
    conv_module = nn.Conv(
      features=4,
      kernel_size=(3,),
      feature_group_count=2,
      padding='VALID',
      kernel_init=initializers.ones,
      bias_init=initializers.ones,
    )
    y, initial_params = conv_module.init_with_output(rng, x)
    self.assertEqual(initial_params['params']['kernel'].shape, (3, 2, 4))
    np.testing.assert_allclose(y, np.full((1, 6, 4), 7.0))

  @parameterized.product(
    n_batch=(1, 3),
    n_features=(1, 2),
    kernel_size=(1, 2, 3, 9),
    n_input_features=(1, 3),
    input_size=(1, 8, 16),
    module=(nn.Conv, nn.ConvLocal),
  )
  def test_circular_conv_1d_constant(
    self,
    n_batch,
    n_features,
    kernel_size,
    n_input_features,
    input_size,
    module,
  ):
    """
    Test 1D convolution with circular padding: filter with all elements equal
    to 1 applied on an input with all elements equal to 1.
    Result should have the same shape as input (except for the feature
    dimension) and have all elements equal to
    `n_input_features * kernel_lin_size`.
    """
    rng = dict(params=random.key(0))
    x = jnp.ones((n_batch, input_size, n_input_features))
    conv_module = module(
      features=n_features,
      kernel_size=(kernel_size,),
      padding='CIRCULAR',
      kernel_init=initializers.ones,
      bias_init=initializers.zeros,
    )
    y, initial_params = conv_module.init_with_output(rng, x)

    kernel_shape = self._get_kernel_shape(
      x.shape, (kernel_size,), module, n_features
    )

    self.assertEqual(
      initial_params['params']['kernel'].shape,
      kernel_shape,
    )
    correct_ans = np.full(
      (n_batch, input_size, n_features), kernel_size * n_input_features
    )
    np.testing.assert_allclose(y, correct_ans)

  def _get_kernel_shape(self, input_shape, kernel_size, module, n_features):
    if module == nn.Conv:
      kernel_shape = kernel_size + (input_shape[-1], n_features)
    elif module == nn.ConvLocal:
      kernel_shape = input_shape[1:-1] + (
        input_shape[-1] * np.prod(kernel_size),
        n_features,
      )
    else:
      raise ValueError(module)
    return kernel_shape

  @parameterized.product(
    n_batch=(1, 3),
    n_features=(1, 2, 10),
    kernel_lin_size=(1, 2, 3, 9),
    n_input_features=(1, 5),
    input_x_size=(14,),
    input_y_size=(5, 10),
    module=(nn.Conv, nn.ConvLocal),
  )
  def test_circular_conv_2d_constant(
    self,
    n_batch,
    n_features,
    kernel_lin_size,
    n_input_features,
    input_x_size,
    input_y_size,
    module,
  ):
    """
    Test 2D convolution with circular padding: square filter with all elements
    equal to 1 applied on an input with all elements equal to 1.
    Result should have the same shape as input (except for the feature
    dimension) and have all elements equal to
    `n_input_features * kernel_lin_size^2`.
    """
    rng = dict(params=random.key(0))
    x = jnp.ones((n_batch, input_x_size, input_y_size, n_input_features))
    kernel_size = (kernel_lin_size, kernel_lin_size)
    conv_module = module(
      features=n_features,
      kernel_size=kernel_size,
      padding='CIRCULAR',
      kernel_init=initializers.ones,
      bias_init=initializers.zeros,
    )
    y, initial_params = conv_module.init_with_output(rng, x)

    kernel_shape = self._get_kernel_shape(
      x.shape, kernel_size, module, n_features
    )

    self.assertEqual(
      initial_params['params']['kernel'].shape,
      kernel_shape,
    )
    correct_ans = np.full(
      (n_batch, input_x_size, input_y_size, n_features),
      kernel_lin_size * kernel_lin_size * n_input_features,
    )
    np.testing.assert_allclose(y, correct_ans)

  def test_circular_conv_1d_custom(self):
    """Test 1d convolution with circular padding and a stride."""
    rng = dict(params=random.key(0))
    x = np.arange(1, 6)
    x = np.expand_dims(x, (0, 2))
    kernel = np.array((1, 2, 1))
    kernel = np.expand_dims(kernel, (1, 2))

    conv_module = nn.Conv(
      features=1,
      kernel_size=(3,),
      strides=(3,),
      padding='CIRCULAR',
      kernel_init=lambda *_: kernel,
      bias_init=initializers.zeros,
    )
    y, initial_params = conv_module.init_with_output(rng, x)

    self.assertEqual(initial_params['params']['kernel'].shape, (3, 1, 1))
    # Compare with manually computed convolution
    correct_ans = np.array((5 + 2 * 1 + 2, 3 + 2 * 4 + 5))
    correct_ans = np.expand_dims(correct_ans, (0, 2))
    np.testing.assert_allclose(y, correct_ans)

  def test_circular_conv_local_1d_custom(self):
    """
    Test 1d local convolution with circular padding and a stride
    """
    rng = dict(params=random.key(0))
    x = np.arange(1, 6)
    x = np.expand_dims(x, (0, 2))
    kernel = np.array(((-1, 2, 3), (4, 5, 6)))
    kernel = np.expand_dims(kernel, (2,))
    conv_module = nn.ConvLocal(
      features=1,
      kernel_size=(3,),
      strides=(3,),
      padding='CIRCULAR',
      kernel_init=lambda *_: kernel,
      bias_init=initializers.zeros,
    )
    y, initial_params = conv_module.init_with_output(rng, x)

    self.assertEqual(initial_params['params']['kernel'].shape, (2, 3, 1))
    # Compare with manually computed convolution
    correct_ans = np.array((-1 * 5 + 2 * 1 + 3 * 2, 4 * 3 + 5 * 4 + 6 * 5))
    correct_ans = np.expand_dims(correct_ans, (0, 2))
    np.testing.assert_allclose(y, correct_ans)

  def test_circular_conv_1d_dilation(self):
    """Test 1d convolution with circular padding and kernel dilation."""
    rng = dict(params=random.key(0))
    x = np.arange(1, 6)
    x = np.expand_dims(x, (0, 2))
    kernel = np.array((1, 2, 1))
    kernel = np.expand_dims(kernel, (1, 2))

    conv_module = nn.Conv(
      features=1,
      kernel_size=(3,),
      padding='CIRCULAR',
      kernel_init=lambda *_: kernel,
      bias_init=initializers.zeros,
      kernel_dilation=(3,),
    )
    y, initial_params = conv_module.init_with_output(rng, x)

    self.assertEqual(initial_params['params']['kernel'].shape, (3, 1, 1))
    # Compare with manually computed convolution
    correct_ans = np.array(
      (
        3 + 2 * 1 + 4,
        4 + 2 * 2 + 5,
        5 + 2 * 3 + 1,
        1 + 2 * 4 + 2,
        2 + 2 * 5 + 3,
      )
    )
    correct_ans = np.expand_dims(correct_ans, (0, 2))
    np.testing.assert_allclose(y, correct_ans)

  def test_circular_conv_local_1d_dilation(self):
    """
    Test 1d local convolution with circular padding and kernel dilation
    """
    rng = dict(params=random.key(0))
    x = np.arange(1, 6)
    x = np.expand_dims(x, (0, 2))
    kernel = np.array(
      ((1, 2, 1), (3, 4, 5), (-1, 1, 2), (2, 3, 4), (-1, -2, -3))
    )
    kernel = np.expand_dims(kernel, (2,))

    conv_module = nn.ConvLocal(
      features=1,
      kernel_size=(3,),
      padding='CIRCULAR',
      kernel_init=lambda *_: kernel,
      bias_init=initializers.zeros,
      kernel_dilation=(3,),
    )
    y, initial_params = conv_module.init_with_output(rng, x)

    self.assertEqual(initial_params['params']['kernel'].shape, (5, 3, 1))
    # Compare with manually computed convolution
    correct_ans = np.array(
      (
        1 * 3 + 2 * 1 + 1 * 4,
        3 * 4 + 4 * 2 + 5 * 5,
        -1 * 5 + 1 * 3 + 2 * 1,
        2 * 1 + 3 * 4 + 4 * 2,
        -1 * 2 + -2 * 5 + -3 * 3,
      )
    )
    correct_ans = np.expand_dims(correct_ans, (0, 2))
    np.testing.assert_allclose(y, correct_ans)

  def test_circular_conv_2d_custom(self):
    """Test 2d convolution with circular padding on a 3x3 example."""
    rng = dict(params=random.key(0))
    x = np.array(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
    x = np.expand_dims(x, (0, 3))
    kernel = np.array(((0, 1, 0), (1, 2, 1), (0, 1, 0)))
    kernel = np.expand_dims(kernel, (2, 3))

    conv_module = nn.Conv(
      features=1,
      kernel_size=(3, 3),
      padding='CIRCULAR',
      kernel_init=lambda *_: kernel,
      bias_init=initializers.zeros,
    )
    y, initial_params = conv_module.init_with_output(rng, x)

    self.assertEqual(initial_params['params']['kernel'].shape, (3, 3, 1, 1))
    # Compare with manually computed convolution
    correct_ans = np.array(
      (
        (2 * 1 + 7 + 2 + 4 + 3, 2 * 2 + 8 + 3 + 5 + 1, 2 * 3 + 9 + 1 + 6 + 2),
        (2 * 4 + 1 + 5 + 7 + 6, 2 * 5 + 2 + 6 + 8 + 4, 2 * 6 + 3 + 4 + 9 + 5),
        (2 * 7 + 4 + 8 + 1 + 9, 2 * 8 + 5 + 9 + 2 + 7, 2 * 9 + 6 + 7 + 3 + 8),
      )
    )
    correct_ans = np.expand_dims(correct_ans, (0, 3))
    np.testing.assert_allclose(y, correct_ans)

  def test_circular_conv_local_2d_custom(self):
    """
    Test 2d local convolution with circular padding on a 3x3 example
    """
    rng = dict(params=random.key(0))
    x = np.array(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
    x = np.expand_dims(x, (0, 3))
    kernel = np.array(
      (
        (
          ((0, 1, 0), (1, 2, 1), (0, 1, 0)),
          ((0, 1, 0), (1, 3, 1), (0, 1, 0)),
          ((0, 1, 0), (1, 4, 1), (0, 1, 0)),
        ),
        (
          ((0, 1, 0), (1, 5, 1), (0, 1, 0)),
          ((0, 1, 0), (1, 6, 1), (0, 1, 0)),
          ((0, 1, 0), (1, 7, 1), (0, 1, 0)),
        ),
        (
          ((0, 1, 0), (1, 8, 1), (0, 1, 0)),
          ((0, 1, 0), (1, 9, 1), (0, 1, 0)),
          ((0, 1, 0), (1, 10, 1), (0, 1, 0)),
        ),
      )
    )
    kernel = np.expand_dims(kernel, (3,))
    kernel = np.reshape(kernel, (3, 3, 9, 1))

    conv_module = nn.ConvLocal(
      features=1,
      kernel_size=(3, 3),
      padding='CIRCULAR',
      kernel_init=lambda *_: kernel,
      bias_init=initializers.zeros,
    )
    y, initial_params = conv_module.init_with_output(rng, x)

    self.assertEqual(initial_params['params']['kernel'].shape, (3, 3, 9, 1))
    # Compare with manually computed convolution
    correct_ans = np.array(
      (
        (2 * 1 + 7 + 2 + 4 + 3, 3 * 2 + 8 + 3 + 5 + 1, 4 * 3 + 9 + 1 + 6 + 2),
        (5 * 4 + 1 + 5 + 7 + 6, 6 * 5 + 2 + 6 + 8 + 4, 7 * 6 + 3 + 4 + 9 + 5),
        (8 * 7 + 4 + 8 + 1 + 9, 9 * 8 + 5 + 9 + 2 + 7, 10 * 9 + 6 + 7 + 3 + 8),
      )
    )
    correct_ans = np.expand_dims(correct_ans, (0, 3))
    np.testing.assert_allclose(y, correct_ans)

  def test_reflect_conv_1d_custom(self):
    """Test 1d convolution with reflection padding and a stride."""
    rng = dict(params=random.key(0))
    x = np.arange(1, 6)
    x = np.expand_dims(x, (0, 2))
    kernel = np.array((1, 2, 1))
    kernel = np.expand_dims(kernel, (1, 2))

    conv_module = nn.Conv(
      features=1,
      kernel_size=(3,),
      strides=(2,),
      padding='REFLECT',
      kernel_init=lambda *_: kernel,
      bias_init=initializers.zeros,
    )
    y, initial_params = conv_module.init_with_output(rng, x)

    self.assertEqual(initial_params['params']['kernel'].shape, (3, 1, 1))
    # Compare with manually computed convolution
    correct_ans = np.array((2 + 2 * 1 + 2, 2 + 2 * 3 + 4, 4 + 2 * 5 + 4))
    correct_ans = np.expand_dims(correct_ans, (0, 2))
    np.testing.assert_allclose(y, correct_ans)

  def test_reflect_conv_2d_custom(self):
    """Test 2d convolution with reflect padding on a 3x3 example."""
    rng = dict(params=random.key(0))
    x = np.array(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
    x = np.expand_dims(x, (0, 3))
    kernel = np.array(((0, 1, 0), (1, 2, 1), (0, 1, 0)))
    kernel = np.expand_dims(kernel, (2, 3))

    conv_module = nn.Conv(
      features=1,
      kernel_size=(3, 3),
      padding='REFLECT',
      kernel_init=lambda *_: kernel,
      bias_init=initializers.zeros,
    )
    y, initial_params = conv_module.init_with_output(rng, x)

    self.assertEqual(initial_params['params']['kernel'].shape, (3, 3, 1, 1))
    # Compare with manually computed convolution
    correct_ans = np.array(
      (
        (2 * 1 + 4 + 2 + 4 + 2, 2 * 2 + 5 + 3 + 5 + 1, 2 * 3 + 6 + 2 + 6 + 2),
        (2 * 4 + 1 + 5 + 7 + 5, 2 * 5 + 2 + 6 + 8 + 4, 2 * 6 + 3 + 5 + 9 + 5),
        (2 * 7 + 4 + 8 + 8 + 4, 2 * 8 + 5 + 9 + 5 + 7, 2 * 9 + 6 + 8 + 6 + 8),
      )
    )
    correct_ans = np.expand_dims(correct_ans, (0, 3))
    np.testing.assert_allclose(y, correct_ans)

  def test_causal_conv1d(self):
    rng = dict(params=random.key(0))
    x = jnp.ones((1, 8, 4))
    conv_module = nn.Conv(
      features=4,
      kernel_size=(3,),
      padding='CAUSAL',
      kernel_init=initializers.ones,
      bias_init=initializers.ones,
    )
    y, _ = conv_module.init_with_output(rng, x)
    correct_ans = np.array(
      [
        [
          [5.0, 5.0, 5.0, 5.0],
          [9.0, 9.0, 9.0, 9.0],
          [13.0, 13.0, 13.0, 13.0],
          [13.0, 13.0, 13.0, 13.0],
          [13.0, 13.0, 13.0, 13.0],
          [13.0, 13.0, 13.0, 13.0],
          [13.0, 13.0, 13.0, 13.0],
          [13.0, 13.0, 13.0, 13.0],
        ]
      ]
    )
    np.testing.assert_allclose(y, correct_ans)
    np.testing.assert_array_equal(correct_ans.shape, y.shape)

  @parameterized.product(
    use_bias=(True, False),
  )
  def test_conv_transpose(self, use_bias):
    rng = dict(params=random.key(0))
    x = jnp.ones((1, 8, 3))
    conv_transpose_module = nn.ConvTranspose(
      features=4,
      use_bias=use_bias,
      kernel_size=(3,),
      padding='VALID',
      kernel_init=initializers.ones,
      bias_init=initializers.ones,
    )
    y, initial_params = conv_transpose_module.init_with_output(rng, x)
    self.assertEqual(initial_params['params']['kernel'].shape, (3, 3, 4))
    correct_ans = np.array(
      [
        [
          [4.0, 4.0, 4.0, 4.0],
          [7.0, 7.0, 7.0, 7.0],
          [10.0, 10.0, 10.0, 10.0],
          [10.0, 10.0, 10.0, 10.0],
          [10.0, 10.0, 10.0, 10.0],
          [10.0, 10.0, 10.0, 10.0],
          [10.0, 10.0, 10.0, 10.0],
          [10.0, 10.0, 10.0, 10.0],
          [7.0, 7.0, 7.0, 7.0],
          [4.0, 4.0, 4.0, 4.0],
        ]
      ]
    )
    if not use_bias:
      correct_ans -= 1.0
    np.testing.assert_allclose(y, correct_ans)

  @parameterized.product(
    use_bias=(True, False),
  )
  def test_multibatch_input_conv_transpose(self, use_bias):
    rng = dict(params=random.key(0))
    x = jnp.ones((2, 5, 8, 3))
    conv_transpose_module = nn.ConvTranspose(
      features=4,
      use_bias=use_bias,
      kernel_size=(3,),
      padding='VALID',
      kernel_init=initializers.ones,
      bias_init=initializers.ones,
    )
    y, initial_params = conv_transpose_module.init_with_output(rng, x)
    self.assertEqual(initial_params['params']['kernel'].shape, (3, 3, 4))
    correct_ans = np.array(
      [
        [
          [4.0, 4.0, 4.0, 4.0],
          [7.0, 7.0, 7.0, 7.0],
          [10.0, 10.0, 10.0, 10.0],
          [10.0, 10.0, 10.0, 10.0],
          [10.0, 10.0, 10.0, 10.0],
          [10.0, 10.0, 10.0, 10.0],
          [10.0, 10.0, 10.0, 10.0],
          [10.0, 10.0, 10.0, 10.0],
          [7.0, 7.0, 7.0, 7.0],
          [4.0, 4.0, 4.0, 4.0],
        ]
      ]
    )
    correct_ans = np.repeat(correct_ans[None], repeats=2, axis=0)
    correct_ans = np.repeat(correct_ans, repeats=5, axis=1)
    if not use_bias:
      correct_ans -= 1.0
    np.testing.assert_allclose(y, correct_ans)

  def test_single_input_conv_transpose(self):
    rng = dict(params=random.key(0))
    x = jnp.ones((8, 3))
    conv_transpose_module = nn.ConvTranspose(
      features=4,
      kernel_size=(3,),
      padding='VALID',
      kernel_init=initializers.ones,
      bias_init=initializers.ones,
    )
    y, initial_params = conv_transpose_module.init_with_output(rng, x)
    self.assertEqual(initial_params['params']['kernel'].shape, (3, 3, 4))
    correct_ans = np.array(
      [
        [4.0, 4.0, 4.0, 4.0],
        [7.0, 7.0, 7.0, 7.0],
        [10.0, 10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0, 10.0],
        [7.0, 7.0, 7.0, 7.0],
        [4.0, 4.0, 4.0, 4.0],
      ]
    )
    np.testing.assert_allclose(y, correct_ans)

  def test_single_input_masked_conv_transpose(self):
    rng = dict(params=random.key(0))
    x = jnp.ones((8, 3))
    m = jnp.tril(jnp.ones((3, 3, 4)))
    conv_transpose_module = nn.ConvTranspose(
      features=4,
      kernel_size=(3,),
      padding='VALID',
      mask=m,
      kernel_init=initializers.ones,
      bias_init=initializers.ones,
    )
    y, initial_params = conv_transpose_module.init_with_output(rng, x)
    self.assertEqual(initial_params['params']['kernel'].shape, (3, 3, 4))
    correct_ans = np.array(
      [
        [4.0, 3.0, 2.0, 1.0],
        [7.0, 5.0, 3.0, 1.0],
        [10.0, 7.0, 4.0, 1.0],
        [10.0, 7.0, 4.0, 1.0],
        [10.0, 7.0, 4.0, 1.0],
        [10.0, 7.0, 4.0, 1.0],
        [10.0, 7.0, 4.0, 1.0],
        [10.0, 7.0, 4.0, 1.0],
        [7.0, 5.0, 3.0, 1.0],
        [4.0, 3.0, 2.0, 1.0],
      ]
    )
    np.testing.assert_allclose(y, correct_ans)

  @parameterized.product(
    n_batch=(1, 3),
    n_features=(1, 2),
    kernel_size=(1, 2, 3, 9),
    n_input_features=(1, 3),
    input_size=(1, 8, 16),
  )
  def test_circular_conv_transpose_1d_constant(
    self, n_batch, n_features, kernel_size, n_input_features, input_size
  ):
    """
    Test 1D transposed convolution with circular padding: filter with all
    elements equal to 1 applied on an input with all elements equal to 1.
    Result should have the same shape as input (except for the feature
    dimension) and have all elements equal to
    `n_input_features * kernel_lin_size`.
    """
    rng = dict(params=random.key(0))
    x = jnp.ones((n_batch, input_size, n_input_features))
    conv_module = nn.ConvTranspose(
      features=n_features,
      kernel_size=(kernel_size,),
      padding='CIRCULAR',
      kernel_init=initializers.ones,
      bias_init=initializers.zeros,
    )
    y, initial_params = conv_module.init_with_output(rng, x)

    self.assertEqual(
      initial_params['params']['kernel'].shape,
      (kernel_size, n_input_features, n_features),
    )
    correct_ans = np.full(
      (n_batch, input_size, n_features), kernel_size * n_input_features
    )
    np.testing.assert_allclose(y, correct_ans)

  @parameterized.product(
    n_batch=(1, 3),
    n_features=(1, 2, 10),
    kernel_lin_size=(1, 2, 3, 9),
    n_input_features=(1, 5),
    input_x_size=(14,),
    input_y_size=(5, 10),
  )
  def test_circular_conv_transpose_2d_constant(
    self,
    n_batch,
    n_features,
    kernel_lin_size,
    n_input_features,
    input_x_size,
    input_y_size,
  ):
    """
    Test 2D transposed convolution with circular padding: square filter with all
    elements equal to 1 applied on an input with all elements equal to 1.
    Result should have the same shape as input (except for the feature
    dimension) and have all elements equal to
    `n_input_features * kernel_lin_size^2`.
    """
    rng = dict(params=random.key(0))
    x = jnp.ones((n_batch, input_x_size, input_y_size, n_input_features))
    conv_module = nn.ConvTranspose(
      features=n_features,
      kernel_size=(kernel_lin_size, kernel_lin_size),
      padding='CIRCULAR',
      kernel_init=initializers.ones,
      bias_init=initializers.zeros,
    )
    y, initial_params = conv_module.init_with_output(rng, x)

    self.assertEqual(
      initial_params['params']['kernel'].shape,
      (kernel_lin_size, kernel_lin_size, n_input_features, n_features),
    )
    correct_ans = np.full(
      (n_batch, input_x_size, input_y_size, n_features),
      kernel_lin_size * kernel_lin_size * n_input_features,
    )
    np.testing.assert_allclose(y, correct_ans)

  def test_circular_conv_transpose_2d_with_vmap(self):
    layer = nn.ConvTranspose(features=5, kernel_size=(3,), padding='CIRCULAR')

    # this is ok
    sample_input = jnp.ones((1, 32, 2))
    out, vars = layer.init_with_output(jax.random.key(0), sample_input)
    self.assertEqual(out.shape, (1, 32, 5))

    batch_input = jnp.ones((8, 32, 2))
    batch_apply = jax.vmap(layer.apply, in_axes=(None, 0))

    # this breaks with the error provided
    batch_out = batch_apply(vars, batch_input)
    self.assertEqual(batch_out.shape, (8, 32, 5))

  def test_circular_conv_transpose_1d_custom(self):
    """Test 1d transposed convolution with circular padding and a stride."""
    rng = dict(params=random.key(0))
    x = np.arange(1, 6)
    x = np.expand_dims(x, (0, 2))
    kernel = np.array((1, 2, 1))
    kernel = np.expand_dims(kernel, (1, 2))

    conv_module = nn.ConvTranspose(
      features=1,
      kernel_size=(3,),
      strides=(3,),
      padding='CIRCULAR',
      kernel_init=lambda *_: kernel,
      bias_init=initializers.zeros,
    )
    y, initial_params = conv_module.init_with_output(rng, x)

    self.assertEqual(initial_params['params']['kernel'].shape, (3, 1, 1))
    # Compare with manually computed convolution
    correct_ans = np.array(
      (  # pyformat: disable
        1 * 1,
        1 * 2,
        1 * 1,
        2 * 1,
        2 * 2,
        2 * 1,
        3 * 1,
        3 * 2,
        3 * 1,
        4 * 1,
        4 * 2,
        4 * 1,
        5 * 1,
        5 * 2,
        5 * 1,
      )
    )
    correct_ans = np.expand_dims(correct_ans, (0, 2))
    np.testing.assert_allclose(y, correct_ans)

  def test_circular_conv_transpose_2d_custom(self):
    """Test 2d transposed convolution with circular padding on a 3x3 example."""
    rng = dict(params=random.key(0))
    x = np.array(
      (
        (1, 2, 3),
        (4, 5, 6),
        (7, 8, 9),
      )
    )
    x = np.expand_dims(x, (0, 3))
    kernel = np.array(((0, 1, 0), (1, 2, 1), (0, 1, 0)))
    kernel = np.expand_dims(kernel, (2, 3))

    conv_module = nn.ConvTranspose(
      features=1,
      kernel_size=(3, 3),
      padding='CIRCULAR',
      kernel_init=lambda *_: kernel,
      bias_init=initializers.zeros,
    )
    y, initial_params = conv_module.init_with_output(rng, x)

    self.assertEqual(initial_params['params']['kernel'].shape, (3, 3, 1, 1))
    # Compare with manually computed convolution
    correct_ans = np.array(
      (
        (18, 21, 24),
        (27, 30, 33),
        (36, 39, 42),
      )
    )
    correct_ans = np.expand_dims(correct_ans, (0, 3))
    np.testing.assert_allclose(y, correct_ans)

  def test_circular_conv_transpose_2d_custom_bias(self):
    """Test 2d transposed convolution with circular padding on a 2x2 example with bias."""
    rng = dict(params=random.key(0))
    x = np.array(((1, 2), (3, 4)))
    x = np.expand_dims(x, (0, 3))
    kernel = np.array(
      (
        (1, 2),
        (3, 4),
      )
    )
    kernel = np.expand_dims(kernel, (2, 3))

    conv_module = nn.ConvTranspose(
      features=1,
      kernel_size=(2, 2),
      padding='CIRCULAR',
      kernel_init=lambda *_: kernel,
      bias_init=initializers.ones,
    )
    y, initial_params = conv_module.init_with_output(rng, x)

    self.assertEqual(initial_params['params']['kernel'].shape, (2, 2, 1, 1))
    # Compare with manually computed convolution
    correct_ans = np.array(
      (
        (21, 23),
        (29, 31),
      )
    )
    correct_ans = np.expand_dims(correct_ans, (0, 3))
    np.testing.assert_allclose(y, correct_ans)

  @parameterized.product(use_bias=(True, False))
  def test_transpose_kernel_conv_transpose(self, use_bias):
    rng = dict(params=random.key(0))
    x = jnp.ones((1, 15, 15, 3))
    conv_module = nn.ConvTranspose(
      features=4,
      use_bias=use_bias,
      strides=2,
      kernel_size=(6, 6),
      padding='CIRCULAR',
      transpose_kernel=True,
    )
    y, initial_params = conv_module.init_with_output(rng, x)
    self.assertEqual(initial_params['params']['kernel'].shape, (6, 6, 4, 3))
    self.assertEqual(y.shape, (1, 30, 30, 4))

  @parameterized.product(module=(nn.Conv, nn.ConvLocal, nn.ConvTranspose))
  def test_int_kernel_equality(self, module):
    conv_int = module(features=4, kernel_size=3)
    conv_seq = module(features=4, kernel_size=(3,))
    x = jnp.ones((8, 3))
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                lambda x, y: (x == y).all(),
                conv_int.init(random.key(0), x),
                conv_seq.init(random.key(0), x),
            )
        )
    )

  def test_embed(self):
    rng = dict(params=random.key(0))
    x = jnp.arange(4)[None]
    dummy_embedding = jnp.broadcast_to(jnp.arange(4)[..., None], (4, 3)).astype(
      jnp.float32
    )
    embed_module = nn.Embed(
      num_embeddings=4,
      features=3,
      embedding_init=lambda rng, shape, dtype: dummy_embedding,
    )
    y, initial_params = embed_module.init_with_output(rng, x)
    np.testing.assert_allclose(y, dummy_embedding[None])
    z = embed_module.apply(
      initial_params, jnp.ones((3,)), method=embed_module.attend
    )
    np.testing.assert_allclose(z, 3.0 * jnp.arange(4))

  def test_embed_numpy(self):
    rng = dict(params=random.key(0))
    x = jnp.arange(4)[None]
    dummy_embedding = np.broadcast_to(np.arange(4)[..., None], (4, 3)).astype(
      np.float32
    )
    embed_module = nn.Embed(
      num_embeddings=4,
      features=3,
      embedding_init=lambda rng, shape, dtype: dummy_embedding,
    )
    y, initial_params = embed_module.init_with_output(rng, x)
    np.testing.assert_allclose(y, dummy_embedding[None])
    z = embed_module.apply(
      initial_params, jnp.ones((3,)), method=embed_module.attend
    )
    np.testing.assert_allclose(z, 3.0 * jnp.arange(4))

  def test_embed_hash(self):
    self.assertEqual(hash(nn.Embed(2, 3)), hash(nn.Embed(2, 3)))
    self.assertNotEqual(hash(nn.Embed(3, 4)), hash(nn.Embed(2, 3)))

  def test_non_final_axis(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        return nn.DenseGeneral(features=6, axis=1, name='dense')(x)

    x = jnp.ones((2, 4, 8))
    y, variables = Foo().init_with_output(random.key(0), x)
    self.assertEqual(
      jax.tree_util.tree_map(jnp.shape, variables['params']),
      {'dense': {'kernel': (4, 6), 'bias': (6,)}},
    )
    self.assertEqual(y.shape, (2, 8, 6))

  def test_non_final_axes(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        return nn.DenseGeneral(features=6, axis=(0, 1), name='dense')(x)

    x = jnp.ones((2, 4, 8))
    y, variables = Foo().init_with_output(random.key(0), x)
    self.assertEqual(
      jax.tree_util.tree_map(jnp.shape, variables['params']),
      {'dense': {'kernel': (2, 4, 6), 'bias': (6,)}},
    )
    self.assertEqual(y.shape, (8, 6))

  def test_canonicalize_padding(self):
    def test_pad(pad, rank, expected=None):
      if expected is None:
        with self.assertRaises(ValueError):
          nn.linear.canonicalize_padding(pad, rank)
      else:
        self.assertEqual(nn.linear.canonicalize_padding(pad, rank), expected)

    test_pad('SAME', 2, 'SAME')
    test_pad(2, 3, [(2, 2), (2, 2), (2, 2)])
    test_pad((2, 2), 3)
    test_pad((2, 2), 1)
    test_pad([1, (2, 3)], 2, [(1, 1), (2, 3)])
    test_pad([None, (1, 2)], 2)


if __name__ == '__main__':
  absltest.main()
