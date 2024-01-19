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

import jax
import jax.numpy as jnp

from flax.experimental import nnx


class TestMultiHeadAttention:
  def test_basic(self):
    module = nnx.MultiHeadAttention(
      num_heads=2,
      in_features=3,
      qkv_features=6,
      out_features=6,
      ctx=nnx.Ctx(0),
    )
    y = module(jnp.ones((1, 7, 3)), decode=False)
    assert y.shape == (1, 7, 6)

  def test_multihead_sow_attention_weights(self):
    class Model(nnx.Module):
      attention_kwargs: dict

      def __init__(self, attention_kwargs, *, ctx):
        self.attention_layers = [
          nnx.MultiHeadAttention(**attention_kwargs, ctx=ctx) for i in range(3)
        ]

      def __call__(self, x, *, sow_weights=False, ctx):
        x = self.attention_layers[0](x, sow_weights=sow_weights, ctx=ctx)
        x = self.attention_layers[1](x, ctx=ctx)
        x = self.attention_layers[2](x, sow_weights=sow_weights, ctx=ctx)
        return x

    ctx = nnx.Ctx(0)
    x = jnp.ones((4, 6, 8))

    module = Model(
      dict(
        in_features=8,
        num_heads=8,
        kernel_init=nnx.initializers.ones_init(),
        bias_init=nnx.initializers.zeros_init(),
        deterministic=False,
      ),
      ctx=ctx,
    )

    flags = dict(decode=False)
    _ = module(x, sow_weights=True, ctx=nnx.Ctx(flags=flags))
    intermediates = module.pop(nnx.Intermediate)
    assert intermediates['attention_layers/0/attention_weights'][0].shape == (
      4,
      8,
      6,
      6,
    )
    assert 'attention_layers/1/attention_weights' not in intermediates
    assert intermediates['attention_layers/2/attention_weights'][0].shape == (
      4,
      8,
      6,
      6,
    )

    flags = dict(decode=False)
    _ = module(x, ctx=nnx.Ctx(flags=flags))
    intermediates = module.pop(nnx.Intermediate)
    assert not intermediates  # empty

  def test_autoregressive_decode_with_x64(self):
    with jax.experimental.enable_x64():
      x = jnp.ones((1, 4, 4))
      module = nnx.MultiHeadAttention(
        in_features=4,
        num_heads=2,
        qkv_features=4,
        decode=True,
        ctx=nnx.Ctx(0),
      )
      module.init_cache(x.shape, dtype=x.dtype)
      assert module.cached_key.shape == (1, 4, 2, 2)
      assert module.cached_value.shape == (1, 4, 2, 2)

      y1 = module(x[:, :1, :])
      y2 = module(x[:, 1:2, :])

      assert y1.shape == (1, 1, 4)
      assert y2.shape == (1, 1, 4)
