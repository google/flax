# Copyright 2020 The Flax Authors.
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

"""Class and functions to define and initialize the actor-critic model."""

import numpy as np
import flax
from flax import nn
import jax.numpy as jnp

class ActorCritic(flax.nn.Module):
  """Class defining the actor-critic model."""

  def apply(self, x, num_outputs):
    """Define the convolutional network architecture.

    Architecture originates from "Human-level control through deep reinforcement
    learning.", Nature 518, no. 7540 (2015): 529-533.
    Note that this is different than the one from  "Playing atari with deep
    reinforcement learning." arxiv.org/abs/1312.5602 (2013)
    """
    dtype = jnp.float32
    x = x.astype(dtype) / 255.
    x = nn.Conv(x, features=32, kernel_size=(8, 8),
                strides=(4, 4), name='conv1',
                dtype=dtype)
    x = nn.relu(x)
    x = nn.Conv(x, features=64, kernel_size=(4, 4),
                strides=(2, 2), name='conv2',
                dtype=dtype)
    x = nn.relu(x)
    x = nn.Conv(x, features=64, kernel_size=(3, 3),
                strides=(1, 1), name='conv3',
                dtype=dtype)
    x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(x, features=512, name='hidden', dtype=dtype)
    x = nn.relu(x)
    # Network used to both estimate policy (logits) and expected state value.
    # See github.com/openai/baselines/blob/master/baselines/ppo1/cnn_policy.py
    logits = nn.Dense(x, features=num_outputs, name='logits', dtype=dtype)
    policy_log_probabilities = nn.log_softmax(logits)
    value = nn.Dense(x, features=1, name='value', dtype=dtype)
    return policy_log_probabilities, value

def create_model(key: np.ndarray, num_outputs: int):
  input_dims = (1, 84, 84, 4)  # (minibatch, height, width, stacked frames)
  module = ActorCritic.partial(num_outputs=num_outputs)
  _, initial_par = module.init_by_shape(key, [(input_dims, jnp.float32)])
  model = flax.nn.Model(module, initial_par)
  return model

def create_optimizer(model: nn.base.Model, learning_rate: float):
  optimizer_def = flax.optim.Adam(learning_rate)
  optimizer = optimizer_def.create(model)
  return optimizer
