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
import numpy as np
from absl.testing import absltest
from temperature_sampler import temperature_sample

jax.config.update('jax_disable_most_optimizations', True)


class TestTemperatureSampler(absltest.TestCase):
  def test_temperature_sampler(self):
    tokens = jnp.array([[5, 0, 0, 0]], dtype=jnp.int32)
    cache = None
    key = jax.random.PRNGKey(0)

    def tokens_to_logits(tokens, cache):
      jax.debug.print('tokens: {}', tokens)
      logits = jax.nn.one_hot(tokens[..., -1:] + 1, 10)
      logits = jnp.where(logits < 0.5, float('-inf'), logits)
      logits = logits.squeeze(axis=1)
      return logits, cache

    new_tokens = temperature_sample(
      tokens, cache, tokens_to_logits, key, topk=5
    )

    np.testing.assert_array_equal(new_tokens, [[5, 6, 7, 8]])


if __name__ == '__main__':
  absltest.main()
