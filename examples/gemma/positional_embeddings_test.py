# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for the positional embeddings utilities."""

from absl.testing import absltest
from absl.testing import parameterized
import positional_embeddings
import jax
import jax.numpy as jnp
import numpy as np

# positional_embeddings.py uses implicit rank broadcast and needs this config to
# be 'allow', while the rest of Flax can use jax_numpy_rank_promotion=raise.
jax.config.update('jax_numpy_rank_promotion', 'allow')


class PositionalEmbeddingsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          input_embedding_shape=(2, 1, 1, 5),
          position=3,
          max_wavelength=100,
          expected=[[[[1.1411201, 1.0299965, 0.0100075, 1.99955, 1.0]]],
                    [[[1.1411201, 1.0299965, 0.0100075, 1.99955, 1.0]]]]
      )
  )
  def test_adds_positional_embeddings(
      self, input_embedding_shape, position, max_wavelength, expected
  ):
    outputs = positional_embeddings.add_positional_embedding(
        jnp.ones(input_embedding_shape), position, max_wavelength
    )
    np.testing.assert_array_almost_equal(outputs, jnp.array(expected))

  @parameterized.parameters(
      dict(
          input_embedding_shape=(2, 1, 2, 4),
          position=3,
          head_dim=4,
          max_wavelength=100,
          expected=[
              [[
                  [-1.1311126, 0.6598157, -0.8488725, 1.2508571],
                  [-1.1311126, 0.6598157, -0.8488725, 1.2508571],
              ]],
              [[
                  [-1.1311126, 0.6598157, -0.8488725, 1.2508571],
                  [-1.1311126, 0.6598157, -0.8488725, 1.2508571],
              ]],
          ],
      )
  )
  def test_rope_positional_embeddings(
      self, input_embedding_shape, position, head_dim, max_wavelength, expected
  ):
    outputs = positional_embeddings.apply_rope(
        jnp.ones(input_embedding_shape),
        jnp.array([[position]]),
        head_dim,
        max_wavelength,
    )
    np.testing.assert_array_almost_equal(outputs, jnp.array(expected))


if __name__ == "__main__":
  absltest.main()
