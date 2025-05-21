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

import jax, jax.numpy as jnp
from jax.lax import Precision
import pytest

from flax import linen
from flax import nnx, config
from flax.typing import Dtype, PrecisionLike

import numpy as np

import typing as tp
from absl.testing import parameterized
from absl.testing import absltest


class TestMultiHeadAttention(parameterized.TestCase):
  def test_basic(self):
    module = nnx.MultiHeadAttention(
      num_heads=2,
      in_features=3,
      qkv_features=6,
      out_features=6,
      rngs=nnx.Rngs(0),
    )
    y = module(jnp.ones((1, 7, 3)), decode=False)
    assert y.shape == (1, 7, 6)

  @pytest.mark.skipif(
    config.flax_mutable_array,
    reason='sow is not supported with flax_mutable_array',
  )
  def test_multihead_sow_attention_weights(self):
    class Model(nnx.Module):
      attention_kwargs: dict

      def __init__(self, attention_kwargs, rng):
        self.attention_layers = [
          nnx.MultiHeadAttention(**attention_kwargs, rngs=rng) for i in range(3)
        ]

      def __call__(self, x, sow_weights=False):
        x = self.attention_layers[0](x, sow_weights=sow_weights)
        x = self.attention_layers[1](x)
        x = self.attention_layers[2](x, sow_weights=sow_weights)
        return x

    rng = nnx.Rngs(0)
    x = jnp.ones((4, 6, 8))

    module = Model(
      dict(
        in_features=8,
        num_heads=8,
        kernel_init=nnx.initializers.ones_init(),
        bias_init=nnx.initializers.zeros_init(),
        deterministic=False,
      ),
      rng,
    )
    module.set_attributes(decode=False)

    _ = module(x, True)
    intermediates = nnx.pop(module, nnx.Intermediate)
    assert intermediates['attention_layers'][0]['attention_weights'].value[
      0
    ].shape == (4, 8, 6, 6)
    assert 1 not in intermediates['attention_layers']
    assert intermediates['attention_layers'][2]['attention_weights'].value[
      0
    ].shape == (4, 8, 6, 6)

    _ = module(x)
    intermediates = nnx.pop(module, nnx.Intermediate)
    assert not intermediates  # empty

  def test_autoregressive_decode_with_x64(self):
    with jax.experimental.enable_x64():
      x = jnp.ones((1, 4, 4))
      module = nnx.MultiHeadAttention(
        in_features=4,
        num_heads=2,
        qkv_features=4,
        decode=True,
        rngs=nnx.Rngs(0),
      )
      module.init_cache(x.shape, dtype=x.dtype)
      assert module.cached_key.value.shape == (1, 4, 2, 2)
      assert module.cached_value.value.shape == (1, 4, 2, 2)

      y1 = module(x[:, :1, :])
      y2 = module(x[:, 1:2, :])

      assert y1.shape == (1, 1, 4)
      assert y2.shape == (1, 1, 4)

  @parameterized.product(keep_rngs=[True, False])
  def test_keep_rngs(self, keep_rngs):
    rngs = nnx.Rngs(42)
    module = nnx.MultiHeadAttention(
      in_features=4,
      num_heads=2,
      qkv_features=4,
      decode=True,
      rngs=rngs,
      dropout_rate=0.5,
      keep_rngs=keep_rngs
    )
    if keep_rngs:
      assert module.rngs == rngs
    else:
      assert module.rngs is None
    if keep_rngs:
      _, _, nondiff = nnx.split(module, nnx.Param, ...)
      assert nondiff["rngs"]["default"]["count"].type is nnx.RngCount
      assert nondiff["rngs"]["default"]["key"].type is nnx.RngKey
    else:
      nnx.split(module, nnx.Param)


# TODO: add all possible constructor argument values to parameterized.product
class TestLinenConsistency(parameterized.TestCase):
  @parameterized.product(
    use_bias=[True, False],
    dtype=[jnp.float32, jnp.float16],
    param_dtype=[jnp.float32, jnp.float16],
    precision=[Precision.DEFAULT, Precision.HIGH, Precision.HIGHEST],
    decode=[True, False],
    normalize_qk=[True, False],
  )
  def test_nnx_attention_equivalence(
    self,
    use_bias: bool,
    dtype: tp.Optional[Dtype],
    param_dtype: Dtype,
    precision: PrecisionLike,
    decode: bool,
    normalize_qk: bool,
  ):
    key = jax.random.key(42)
    rngs = nnx.Rngs(42)

    num_heads = 2
    in_features = 3
    qkv_features = 6
    out_features = 6

    x = jax.numpy.ones((1, in_features))
    model_nnx = nnx.MultiHeadAttention(
      num_heads=num_heads,
      in_features=in_features,
      qkv_features=qkv_features,
      out_features=out_features,
      use_bias=use_bias,
      dtype=dtype,
      param_dtype=param_dtype,
      precision=precision,
      decode=decode,
      normalize_qk=normalize_qk,
      rngs=rngs,
    )
    model = linen.MultiHeadDotProductAttention(
      num_heads=num_heads,
      qkv_features=qkv_features,
      out_features=out_features,
      use_bias=use_bias,
      dtype=dtype,
      param_dtype=param_dtype,
      precision=precision,
      decode=decode,
      normalize_qk=normalize_qk,
    )
    variables = model.init(key, x)

    for qkvo in ('query', 'key', 'value', 'out'):
      getattr(model_nnx, qkvo).kernel.value = variables['params'][qkvo][
        'kernel'
      ]
      if use_bias:
        getattr(model_nnx, qkvo).bias.value = variables['params'][qkvo]['bias']
    if decode:
      model_nnx.init_cache(x.shape, dtype=dtype)

    out_nnx = model_nnx(x)
    out, cache = model.apply(variables, x, mutable=['cache'])
    np.testing.assert_array_equal(out, out_nnx)


class TestKVFeatures(parameterized.TestCase):

  def test_varying_num_features(self):
    key = jax.random.key(42)
    rngs = nnx.Rngs(42)

    num_heads = 2
    in_features = 3
    in_kv_features = 4
    qkv_features = 6
    out_features = 6

    x = jax.numpy.ones((1, in_features))
    y = jax.random.normal(key, (1, in_kv_features))
    layer = nnx.MultiHeadAttention(
      num_heads=num_heads,
      in_features=in_features,
      qkv_features=qkv_features,
      out_features=out_features,
      in_kv_features=in_kv_features,
      rngs=rngs,
      decode=False
    )

    self.assertIsNotNone(layer(x, y))


if __name__ == '__main__':
  absltest.main()
