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

"""Tests for flax.training.train_state."""

from absl.testing import absltest, parameterized
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax


class TrainStateTest(parameterized.TestCase):
  @parameterized.parameters(False, True)
  def test_apply_gradients_array_params(self, jit):
    state = train_state.TrainState.create(
      apply_fn=lambda params, x: params * x,
      params=jnp.full((2,), 2.0),
      tx=optax.sgd(0.1),
    )
    step = lambda state: state.apply_gradients(grads=2 * state.params)
    state = (jax.jit(step) if jit else step)(state)
    np.testing.assert_allclose(state.params, jnp.full((2,), 1.6), rtol=1e-6)
    self.assertEqual(state.step, 1)


if __name__ == '__main__':
  absltest.main()
