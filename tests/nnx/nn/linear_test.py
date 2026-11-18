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

from functools import partial
import typing as tp

import jax
import jax.numpy as jnp
from absl.testing import absltest
from absl.testing import parameterized
from jax.lax import Precision
import numpy as np

from flax import linen
from flax import nnx
from flax.typing import Dtype, PrecisionLike, Shape


class TestLinearGeneral(parameterized.TestCase):
  @parameterized.product(
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
    precision=[Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST],
    preferred_element_type=[None, jnp.float32],
  )
  def test_basic(
    self,
    dtype,
    param_dtype,
    precision,
    preferred_element_type,
  ):
    module = nnx.LinearGeneral(
      2,
      3,
      rngs=nnx.Rngs(0),
      dtype=dtype,
      param_dtype=param_dtype,
      precision=precision,
      preferred_element_type=preferred_element_type,
    )
    y = module(jnp.ones((1, 2)))

    assert y.shape == (1, 3)
    if preferred_element_type is not None:
      assert y.dtype == preferred_element_type
    assert module.kernel.shape == (2, 3)
    assert module.kernel.dtype == param_dtype
    assert module.bias is not None
    assert module.bias.shape == (3,)

  def test_basic_multi_features(self):
    module = nnx.LinearGeneral(2, (3, 4), rngs=nnx.Rngs(0))
    y = module(jnp.ones((1, 2)))

    assert y.shape == (1, 3, 4)
    assert module.kernel.shape == (2, 3, 4)
    assert module.bias is not None
    assert module.bias.shape == (3, 4)


class TestLinenConsistency(parameterized.TestCase):
  @parameterized.product(
    use_bias=[True, False],
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
    precision=[Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST],
    preferred_element_type=[None, jnp.float32],
  )
  def test_nnx_linear_equivalence(
    self,
    use_bias: bool,
    dtype: tp.Optional[Dtype],
    param_dtype: Dtype,
    precision: PrecisionLike,
    preferred_element_type: tp.Optional[Dtype],
  ):
    key = jax.random.key(42)
    rngs = nnx.Rngs(42)
    IN_FEATURES = 32
    OUT_FEATURES = 64

    x = jax.numpy.ones((1, IN_FEATURES))
    model_nnx = nnx.eval_shape(
      lambda rngs: nnx.Linear(
        IN_FEATURES,
        OUT_FEATURES,
        use_bias=use_bias,
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
        preferred_element_type=preferred_element_type,
        rngs=rngs,
      ),
      rngs,
    )
    if preferred_element_type is not None:
      dot_general = partial(
        jax.lax.dot_general,
        preferred_element_type=preferred_element_type,
      )
    else:
      dot_general = None
    model = linen.Dense(
      OUT_FEATURES,
      use_bias=use_bias,
      dtype=dtype,
      param_dtype=param_dtype,
      precision=precision,
      dot_general=dot_general,
    )
    variables = model.init(key, x)
    model_nnx.kernel.set_value(variables['params']['kernel'])
    if use_bias:
      model_nnx.bias.set_value(variables['params']['bias'])

    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    assert isinstance(out, jax.Array)
    np.testing.assert_array_equal(out, out_nnx)

  @parameterized.product(
    einsum_str=['defab,bcef->adefc', 'd...ab,bc...->ad...c'],
    bias_shape=[None, (6, 7, 5)],
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
    precision=[Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST],
    preferred_element_type=[None, jnp.float32],
  )
  def test_nnx_einsum_equivalence(
    self,
    einsum_str,
    bias_shape: tp.Optional[Shape],
    dtype: tp.Optional[Dtype],
    param_dtype: Dtype,
    precision: PrecisionLike,
    preferred_element_type: tp.Optional[Dtype],
  ):
    key = jax.random.key(42)
    rngs = nnx.Rngs(42)
    INPUT_SHAPE = (8, 6, 7, 3, 4)
    KERNEL_SHAPE = (4, 5, 6, 7)

    x = jax.random.normal(key, INPUT_SHAPE)
    model_nnx = nnx.Einsum(
      einsum_str,
      KERNEL_SHAPE,
      bias_shape,
      dtype=dtype,
      param_dtype=param_dtype,
      precision=precision,
      preferred_element_type=preferred_element_type,
      rngs=rngs,
    )
    model = linen.Einsum(
      KERNEL_SHAPE,
      einsum_str,
      use_bias=True if bias_shape is not None else False,
      dtype=dtype,
      param_dtype=param_dtype,
      precision=precision,
      preferred_element_type=preferred_element_type,
    )

    variables = model.init(key, x)
    variables['params']['kernel'] = model_nnx.kernel[...]
    if bias_shape is not None:
      assert model_nnx.bias is not None
      variables['params']['bias'] = model_nnx.bias[...]
    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    assert isinstance(out, jax.Array)
    np.testing.assert_array_equal(out, out_nnx)

    variables = model.init(key, x)
    model_nnx.kernel.set_value(variables['params']['kernel'])
    if bias_shape is not None:
      assert model_nnx.bias is not None
      model_nnx.bias.set_value(variables['params']['bias'])
    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    assert isinstance(out, jax.Array)
    np.testing.assert_array_equal(out, out_nnx)

  def test_einsum_op(self):
    def custom_einsum(*args, **kwargs):
      out = jnp.einsum(*args, **kwargs)
      return out.reshape((1, *out.shape))
    model = nnx.Einsum('ab,bc->ac', (3, 4), einsum_op=custom_einsum,
                       rngs=nnx.Rngs(42))
    y = model(jnp.ones((2, 3)))
    assert y.shape == (1, 2, 4)


class TestPReLUConsistency(parameterized.TestCase):
  @parameterized.product(
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
  )
  def test_equivalence(self, dtype, param_dtype):
    key = jax.random.key(42)
    x = jnp.linspace(-10, 10, 20, dtype=dtype)
    negative_slope_init = 0.02
    nnx_prelu = nnx.PReLU(negative_slope_init=negative_slope_init, param_dtype=param_dtype)
    linen_prelu = linen.PReLU(negative_slope_init=negative_slope_init, param_dtype=param_dtype)

    variables = linen_prelu.init(key, x)
    expected = linen_prelu.apply(variables, x)
    output = nnx_prelu(x)
    np.testing.assert_array_equal(output, expected)

    # Check gradients
    @jax.jit
    def nnx_loss_function(model):
      y = model(x)
      return y.mean()

    @jax.jit
    def linen_loss_function(variables):
      y = linen_prelu.apply(variables, x)
      return y.mean()

    expected_loss, expected_grads = jax.value_and_grad(linen_loss_function)(variables)
    loss, grads = jax.value_and_grad(nnx_loss_function)(nnx_prelu)

    np.testing.assert_array_equal(loss, expected_loss)
    np.testing.assert_array_equal(
      expected_grads['params']['negative_slope'], grads.negative_slope[...]
    )


class TestEinsum(parameterized.TestCase):

  @parameterized.product(
    einsum_str_input_kernel_bias_output=[
      # matrix multiply: (batch, in) x (in, out) -> (batch, out)
      ('bi,io->bo', (4, 3), (3, 5), (5,), (4, 5)),
      # batched matmul with extra dims
      ('nta,hab->nthb', (16, 11, 2), (8, 2, 4), (8, 4), (16, 11, 8, 4)),
      # multi-head attention-like: (batch, seq, heads, dim)
      ('bshd,hdo->bsho', (2, 8, 4, 16), (4, 16, 32), (4, 32),
       (2, 8, 4, 32)),
      # contraction over multiple dims
      ('defab,bcef->adefc', (8, 6, 7, 3, 4), (4, 5, 6, 7), (6, 7, 5),
       (3, 8, 6, 7, 5)),
      # simple 2D without bias
      ('ij,jk->ik', (2, 3), (3, 4), None, (2, 4)),
    ],
  )
  def test_output_shape(self, einsum_str_input_kernel_bias_output):
    einsum_str, input_shape, kernel_shape, bias_shape, expected_shape = (
      einsum_str_input_kernel_bias_output
    )
    model = nnx.Einsum(
      einsum_str, kernel_shape, bias_shape, rngs=nnx.Rngs(0)
    )
    x = jnp.ones(input_shape)
    y = model(x)
    self.assertEqual(y.shape, expected_shape)
    self.assertEqual(model.kernel.shape, kernel_shape)
    if bias_shape is not None:
      self.assertIsNotNone(model.bias)
      self.assertEqual(model.bias.shape, bias_shape)
    else:
      self.assertIsNone(model.bias)

  @parameterized.product(
    dtype=[None, jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
    preferred_element_type=[None, jnp.float32],
  )
  def test_dtypes(self, dtype, param_dtype, preferred_element_type):
    model = nnx.Einsum(
      'bi,io->bo',
      (3, 5),
      (5,),
      dtype=dtype,
      param_dtype=param_dtype,
      preferred_element_type=preferred_element_type,
      rngs=nnx.Rngs(0),
    )
    self.assertEqual(model.kernel.dtype, param_dtype)
    self.assertEqual(model.bias.dtype, param_dtype)

    x = jnp.ones((2, 3))
    y = model(x)
    if preferred_element_type is not None:
      self.assertEqual(y.dtype, preferred_element_type)
    elif dtype is not None:
      self.assertEqual(y.dtype, dtype)
    else:
      # dtype=None: output dtype inferred from input (float32) and params
      expected_dtype = jnp.result_type(jnp.float32, param_dtype)
      self.assertEqual(y.dtype, expected_dtype)

  @parameterized.product(
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
  )
  def test_no_bias(self, dtype, param_dtype):
    model = nnx.Einsum(
      'bi,io->bo', (3, 5), bias_shape=None,
      dtype=dtype, param_dtype=param_dtype,
      kernel_init=nnx.initializers.ones, rngs=nnx.Rngs(0),
    )
    x = jnp.ones((2, 3), dtype=dtype)
    y = model(x)

    # kernel=ones: y[b,o] = sum_i(x[b,i] * 1) = 3.0 for all b,o
    rtol = (
      1e-3
      if dtype == jnp.float16 or param_dtype == jnp.float16
      else 1e-6
    )
    np.testing.assert_allclose(
      y, jnp.full((2, 5), 3.0, dtype=y.dtype), rtol=rtol,
    )

  def test_bias_addition_multidim(self):
    model = nnx.Einsum(
      'defab,bcef->adefc',
      (4, 5, 6, 7),
      (6, 7, 5),
      kernel_init=nnx.initializers.zeros,
      bias_init=nnx.initializers.ones,
      rngs=nnx.Rngs(0),
    )
    x = jnp.zeros((8, 6, 7, 3, 4))
    # Verify _infer_broadcasted_bias_shape computes correct shape
    # output='adefc': bias dims (e,f,c) from rhs -> [1, 1, 6, 7, 5]
    broadcasted_shape = model._infer_broadcasted_bias_shape(
      'defab,bcef->adefc', x, model.kernel[...]
    )
    self.assertEqual(broadcasted_shape, [1, 1, 6, 7, 5])
    # kernel=0 means einsum result is 0, output should be broadcast bias
    y = model(x)
    self.assertEqual(y.shape, (3, 8, 6, 7, 5))
    np.testing.assert_allclose(y, jnp.ones_like(y), rtol=1e-6)

  def test_bias_addition_simple(self):
    model = nnx.Einsum(
      'bi,io->bo', (3, 5), (5,), rngs=nnx.Rngs(0)
    )
    x = jnp.ones((4, 3))
    y_with_bias = model(x)

    no_bias_result = jnp.einsum('bi,io->bo', x, model.kernel[...])
    diff = y_with_bias - no_bias_result
    expected_diff = jnp.broadcast_to(
      model.bias[...], y_with_bias.shape
    )
    np.testing.assert_allclose(diff, expected_diff, rtol=1e-6)

  def test_bias_broadcast_with_ellipsis(self):
    model = nnx.Einsum(
      'd...ab,bc...->ad...c',
      (4, 5, 6, 7),
      (5, 6, 7),
      kernel_init=nnx.initializers.zeros,
      bias_init=nnx.initializers.ones,
      rngs=nnx.Rngs(0),
    )
    x = jnp.zeros((8, 6, 7, 3, 4))
    # Verify broadcasted bias shape for ellipsis equation
    broadcasted_shape = model._infer_broadcasted_bias_shape(
      'd...ab,bc...->ad...c', x, model.kernel[...]
    )
    self.assertEqual(broadcasted_shape, [1, 1, 6, 7, 5])
    y = model(x)
    self.assertEqual(y.shape, (3, 8, 6, 7, 5))
    np.testing.assert_allclose(y, jnp.ones_like(y), rtol=1e-6)

  def test_spaces_in_einsum_str(self):
    model_spaces = nnx.Einsum(
      'b i, i o -> b o', (3, 5), bias_shape=None,
      kernel_init=nnx.initializers.ones, rngs=nnx.Rngs(0),
    )
    model_clean = nnx.Einsum(
      'bi,io->bo', (3, 5), bias_shape=None,
      kernel_init=nnx.initializers.ones, rngs=nnx.Rngs(0),
    )
    x = jnp.ones((2, 3))
    np.testing.assert_array_equal(
      model_spaces(x), model_clean(x)
    )

  def test_einsum_str_missing_arrow_raises(self):
    with self.assertRaisesRegex(
      ValueError, 'must be explicit and include'
    ):
      nnx.Einsum('bi,io', (3, 5), rngs=nnx.Rngs(0))

  def test_einsum_str_wrong_operand_count_raises(self):
    with self.assertRaisesRegex(
      ValueError, 'exactly two operands'
    ):
      nnx.Einsum('a,b,c->d', (3,), rngs=nnx.Rngs(0))

  @parameterized.product(
    precision=[Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST],
  )
  def test_precision(self, precision):
    received = []

    def capturing_einsum(
      *args, precision=None, out_sharding=None, **kwargs,
    ):
      received.append(precision)
      return jnp.einsum(*args, precision=precision, **kwargs)

    model = nnx.Einsum(
      'bi,io->bo', (3, 5), bias_shape=None,
      precision=precision,
      einsum_op=capturing_einsum,
      rngs=nnx.Rngs(0),
    )
    model(jnp.ones((2, 3)))
    self.assertLen(received, 1)
    self.assertEqual(received[0], precision)

  @parameterized.product(
    bias_shape=[(5,), None],
  )
  def test_gradient_flow(self, bias_shape):
    model = nnx.Einsum(
      'bi,io->bo', (3, 5), bias_shape,
      kernel_init=nnx.initializers.ones,
      bias_init=nnx.initializers.zeros,
      rngs=nnx.Rngs(0),
    )
    x = jnp.ones((2, 3))

    grads = jax.grad(lambda m: m(x).sum())(model)
    # d(sum(x @ kernel + bias))/d(kernel[j,k]) = sum_i(x[i,j]) = 2.0
    np.testing.assert_allclose(
      grads.kernel[...], jnp.full((3, 5), 2.0), rtol=1e-6
    )
    if bias_shape is not None:
      # d(sum(bias))/d(bias[k]) = batch_size = 2.0
      np.testing.assert_allclose(
        grads.bias[...], jnp.full(bias_shape, 2.0), rtol=1e-6
      )
    else:
      self.assertIsNone(grads.bias)

  def test_einsum_str_call_override(self):
    received = []

    def capturing_einsum(
      expr, *args, out_sharding=None, **kwargs,
    ):
      received.append(expr)
      return jnp.einsum(expr, *args, **kwargs)

    # Constructor uses 'ab,bc->ac', call overrides with 'bi,io->bo'
    model = nnx.Einsum(
      'ab,bc->ac', (3, 5), bias_shape=None,
      einsum_op=capturing_einsum, rngs=nnx.Rngs(0),
    )
    x = jnp.ones((2, 3))

    # Without override: constructor string is used
    model(x)
    self.assertLen(received, 1)
    self.assertEqual(received[0], 'ab,bc->ac')

    # With override: call-time string must win
    model(x, einsum_str='bi,io->bo')
    self.assertLen(received, 2)
    self.assertEqual(received[1], 'bi,io->bo')

  def test_custom_initializers(self):
    model = nnx.Einsum(
      'bi,io->bo',
      (3, 5),
      (5,),
      kernel_init=nnx.initializers.ones,
      bias_init=nnx.initializers.zeros,
      param_dtype=jnp.float32,
      rngs=nnx.Rngs(0),
    )
    np.testing.assert_array_equal(
      model.kernel[...], jnp.ones((3, 5))
    )
    np.testing.assert_array_equal(
      model.bias[...], jnp.zeros((5,))
    )
    # Verify initializers affect forward pass
    x = jnp.ones((2, 3))
    y = model(x)
    # kernel=ones, bias=zeros: y = x @ ones + 0 = [[3,3,3,3,3],...]
    expected = jnp.full((2, 5), 3.0)
    np.testing.assert_allclose(y, expected, rtol=1e-6)

  def test_ellipsis_einsum_str(self):
    model = nnx.Einsum(
      'd...ab,bc...->ad...c',
      (4, 5, 6, 7),
      bias_shape=None,
      rngs=nnx.Rngs(0),
    )
    x = jnp.ones((8, 6, 7, 3, 4))
    y = model(x)
    self.assertEqual(y.shape, (3, 8, 6, 7, 5))


class TestLayersSameGraph(parameterized.TestCase):

  @parameterized.product(
      module_args_kwargs_initargs=[
          (nnx.LinearGeneral, (2, (3, 4)), ("kernel_init", "bias_init")),
          (nnx.Linear, (2, 4), ("kernel_init", "bias_init")),
          (nnx.Einsum, ("ik,kj->ij", (5, 4), 5), ("kernel_init", "bias_init")),
          (nnx.Conv, (2, 4, 3), ("kernel_init", "bias_init")),
          (nnx.ConvTranspose, (2, 4, 3), ("kernel_init", "bias_init")),
          (nnx.Embed, (2, 4), ("embedding_init",)),
          (
              nnx.MultiHeadAttention,
              (8, 5, 16),
              ("kernel_init", "out_kernel_init", "bias_init", "out_bias_init"),
          ),
          (nnx.BatchNorm, (3,), ("scale_init", "bias_init")),
          (nnx.LayerNorm, (3,), ("scale_init", "bias_init")),
          (nnx.RMSNorm, (3,), ("scale_init",)),
          (nnx.GroupNorm, (6, 3), ("scale_init", "bias_init")),
          (nnx.InstanceNorm, (6,), ("scale_init", "bias_init")),
          (
              nnx.LSTMCell,
              (4, 5),
              ("kernel_init", "recurrent_kernel_init", "bias_init"),
          ),
          (
              nnx.OptimizedLSTMCell,
              (4, 5),
              ("kernel_init", "recurrent_kernel_init", "bias_init"),
          ),
          (
              nnx.SimpleCell,
              (4, 5),
              ("kernel_init", "recurrent_kernel_init", "bias_init"),
          ),
          (
              nnx.GRUCell,
              (4, 5),
              ("kernel_init", "recurrent_kernel_init", "bias_init"),
          ),
      ],
  )
  def test(self, module_args_kwargs_initargs):
    module_cls, args, init_argnames = module_args_kwargs_initargs
    kwargs = {"rngs": nnx.Rngs(0)}
    init_zeros = nnx.initializers.zeros
    init_ones = nnx.initializers.ones
    init1_kwargs = {k: init_zeros for k in init_argnames}
    init2_kwargs = {k: init_ones for k in init_argnames}
    mod1 = module_cls(*args, **init1_kwargs, **kwargs)
    mod2 = module_cls(*args, **init2_kwargs, **kwargs)
    g1, g2 = nnx.graphdef(mod1), nnx.graphdef(mod2)
    assert g1 == g2


class TestLayersParamsMetadata(parameterized.TestCase):

  @parameterized.product(
      module_args_kwargs_initargs=[
          (nnx.LinearGeneral, (2, (3, 4)), (("kernel", 2, ()), ("bias", 1, ()))),
          (nnx.Linear, (2, 4), (("kernel", 2, ()), ("bias", 1, ()))),
          (nnx.Einsum, ("ik,kj->ij", (5, 4), 5), (("kernel", 2, ()), ("bias", 1, ()))),
          (nnx.Conv, (2, 4, 3), (("kernel", 2, ()), ("bias", 1, ()))),
          (nnx.ConvTranspose, (2, 4, 3), (("kernel", 2, ()), ("bias", 1, ()))),
          (nnx.Embed, (2, 4), (("embedding", 2, ()), )),
          (
              partial(nnx.MultiHeadAttention, normalize_qk=True),
              (8, 5, 16),
              (
                ("kernel", 2, (("query", "kernel"), ("key", "kernel"), ("value", "kernel"))),
                ("out_kernel", 2, (("out", "kernel"), )),
                ("bias", 1, (("query", "bias"), ("key", "bias"), ("value", "bias"))),
                ("out_bias", 1, (("out", "bias"), )),
                ("query_ln_scale", 1, (("query_ln", "scale"), )),
                ("key_ln_scale", 1, (("key_ln", "scale"), )),
              ),
          ),
          (nnx.BatchNorm, (3,), (("scale", 1, ()), ("bias", 1, ()))),
          (nnx.LayerNorm, (3,), (("scale", 1, ()), ("bias", 1, ()))),
          (nnx.RMSNorm, (3,), (("scale", 1, ()), )),
          (nnx.GroupNorm, (6, 3), (("scale", 1, ()), ("bias", 1, ()))),
          (nnx.InstanceNorm, (6,), (("scale", 1, ()), ("bias", 1, ()))),
          (
            nnx.LoRA,
            (3, 2, 4),
            (
              ("a", 2, ((None, "lora_a"), )),
              ("b", 2, ((None, "lora_b"), )),
            )
          ),
          (
            partial(nnx.LoRALinear, lora_rank=4),
            (3, 2),
            (
              ("a", 2, (("lora", "lora_a"), )),
              ("b", 2, (("lora", "lora_b"), )),
            )
          ),
          (
              nnx.LSTMCell,
              (4, 5),
              (
                (
                  "kernel",
                  2,
                  ((name, "kernel") for name in ["ii", "if_", "ig", "io"])
                ),
                (
                  "recurrent_kernel",
                  2,
                  ((name, "kernel") for name in ["hi", "hf", "hg", "ho"])
                ),
                (
                  "bias",
                  1,
                  ((name, "bias") for name in ["hi", "hf", "hg", "ho"])
                ),
              ),
          ),
          (
              nnx.OptimizedLSTMCell,
              (4, 5),
              (
                ("kernel", 2, (("dense_i", "kernel"), )),
                ("recurrent_kernel", 2, (("dense_h", "kernel"), )),
                ("bias", 1, (("dense_h", "bias"), )),
              )
          ),
          (
              nnx.SimpleCell,
              (4, 5),
              (
                ("kernel", 2, (("dense_i", "kernel"), )),
                ("bias", 1, (("dense_i", "bias"), )),
                ("recurrent_kernel", 2, (("dense_h", "kernel"), )),
              )
          ),
          (
              nnx.GRUCell,
              (4, 5),
              (
                ("kernel", 2, (("dense_i", "kernel"), )),
                ("bias", 1, (("dense_i", "bias"), )),
                ("recurrent_kernel", 2, (("dense_h", "kernel"), )),
              )
          ),
      ],
  )
  def test(self, module_args_kwargs_initargs):
    module_cls, args, metadata_argnames = module_args_kwargs_initargs
    kwargs = {"rngs": nnx.Rngs(0)}
    out_sharding = ("din", "dout")
    metadata_kwargs = {
      f"{key}_metadata": {"out_sharding": out_sharding[:le]}
      for key, le, _ in metadata_argnames
    }

    mesh = jax.make_mesh(
      (1, 1),
      out_sharding,
      axis_types=(jax.sharding.AxisType.Auto,) * len(out_sharding),
    )
    with jax.set_mesh(mesh):
      module = module_cls(*args, **metadata_kwargs, **kwargs)

    for key, le, attrs in metadata_argnames:
      attrs = attrs if attrs else ((None, key), )
      for attr_name, param_name in attrs:
        attr = getattr(module, attr_name) if attr_name is not None else module
        param = getattr(attr, param_name)
        self.assertIsNotNone(param.out_sharding)
        self.assertEqual(param.out_sharding, out_sharding[:le])


if __name__ == '__main__':
  absltest.main()
