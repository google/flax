# Copyright 2021 The Flax Authors.
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

"""Tests for sst2.train."""
from absl.testing import absltest
from absl.testing import parameterized
from configs import default as default_config
import jax
import jax.test_util
import numpy as np
import train

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class TrainTest(parameterized.TestCase):

  def test_train_step_returns_correct_output_shape(self):
    """Tests if the train step function returns the correct shape."""
    # Create model and a state that contains the parameters.
    config = default_config.get_config()
    config.vocab_size = 13
    rng = jax.random.PRNGKey(config.seed)
    model = train.model_from_config(config)
    state = train.create_train_state(rng, config, model)

    token_ids = np.array([[2, 4, 3], [2, 6, 3]], dtype=np.int32)
    lengths = np.array([2, 3], dtype=np.int32)
    labels = np.zeros_like(lengths)
    batch = {'token_ids': token_ids, 'length': lengths, 'label': labels}
    rngs = {'dropout': rng}
    train_step_fn = jax.jit(train.train_step)
    new_state, metrics = train_step_fn(state, batch, rngs)
    self.assertIsInstance(new_state, train.TrainState)
    self.assertIsInstance(metrics, dict)

if __name__ == '__main__':
  absltest.main()
