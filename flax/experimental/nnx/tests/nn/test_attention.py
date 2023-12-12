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

import jax.numpy as jnp

from flax.experimental import nnx


class TestMultiHeadAttention:
  def test_basic(self):
    module = nnx.MultiHeadAttention(2, 3, 6, rngs=nnx.Rngs(0))
    y = module(jnp.ones((1, 7, 3)))
    assert y.shape == (1, 7, 6)

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
        features_in=8,
        num_heads=8,
        kernel_init=nnx.initializers.ones(),
        bias_init=nnx.initializers.zeros(),
        deterministic=False,
      ),
      rng,
    )

    _ = module(x, True)
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

    _ = module(x)
    intermediates = module.pop(nnx.Intermediate)
    assert not intermediates  # empty
