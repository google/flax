# Lint as: python3

# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for flax.struct."""

from typing import Any

from absl.testing import absltest

from flax import nn
from flax import optim
from flax import serialization
from flax import struct

import jax
from jax import random
import jax.numpy as jnp

import numpy as onp


@struct.dataclass
class Point:
  x: float
  y: float
  meta: Any = struct.field(pytree_node=False)


class SerializationTest(absltest.TestCase):

  def test_dataclass_serialization(self):
    p = Point(x=1, y=2, meta={'dummy': True})
    state_dict = serialization.to_state_dict(p)
    self.assertEqual(state_dict, {
        'x': 1,
        'y': 2,
    })
    restored_p = serialization.from_state_dict(p, {'x': 3, 'y': 4})
    expected_p = Point(x=3, y=4, meta={'dummy': True})
    self.assertEqual(restored_p, expected_p)

    with self.assertRaises(ValueError):  # invalid field
      serialization.from_state_dict(p, {'z': 3})
    with self.assertRaises(ValueError):  # missing field
      serialization.from_state_dict(p, {'x': 3})

  def test_model_serialization(self):
    rng = random.PRNGKey(0)
    model_def = nn.Dense.partial(features=1, kernel_init=nn.initializers.ones)
    _, model = model_def.create_by_shape(rng, [((1, 1), jnp.float32)])
    state = serialization.to_state_dict(model)
    self.assertEqual(state, {
        'params': {
            'kernel': onp.ones((1, 1)),
            'bias': onp.zeros((1,)),
        }
    })
    state = {
        'params': {
            'kernel': onp.zeros((1, 1)),
            'bias': onp.zeros((1,)),
        }
    }
    restored_model = serialization.from_state_dict(model, state)
    self.assertEqual(restored_model.params, state['params'])

  def test_optimizer_serialization(self):
    rng = random.PRNGKey(0)
    model_def = nn.Dense.partial(features=1, kernel_init=nn.initializers.ones)
    _, model = model_def.create_by_shape(rng, [((1, 1), jnp.float32)])
    optim_def = optim.Momentum(learning_rate=1.)
    optimizer = optim_def.create(model)
    state = serialization.to_state_dict(optimizer)
    expected_state = {
        'target': {
            'params': {
                'kernel': onp.ones((1, 1)),
                'bias': onp.zeros((1,)),
            }
        },
        'state': {
            'step': 0,
            'param_states': {
                'params': {
                    'kernel': {'momentum': onp.zeros((1, 1))},
                    'bias': {'momentum': onp.zeros((1,))},
                }
            }
        },
    }
    self.assertEqual(state, expected_state)
    state = jax.tree_map(lambda x: x + 1, expected_state)
    restored_optimizer = serialization.from_state_dict(optimizer, state)
    optimizer_plus1 = jax.tree_map(lambda x: x + 1, optimizer)
    self.assertEqual(restored_optimizer, optimizer_plus1)


if __name__ == '__main__':
  absltest.main()
