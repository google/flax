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

"""Library for training an MNIST model."""

import jax
from jax import numpy as jnp

import tensorflow_datasets as tfds

from flax import nn
from flax import optim
from flax.metrics import tensorboard


class CNN(nn.Module):
  """A simple CNN model."""

  # pylint: disable=arguments-differ
  def apply(self, x):
    x = nn.Conv(x, features=32, kernel_size=(3, 3))
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(x, features=64, kernel_size=(3, 3))
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(x, features=256)
    x = nn.relu(x)
    x = nn.Dense(x, features=10)
    x = nn.log_softmax(x)
    return x


def create_model(key):
  _, initial_params = CNN.init_by_shape(key, [((1, 28, 28, 1), jnp.float32)])
  model = nn.Model(CNN, initial_params)
  return model


def create_optimizer(model, learning_rate, beta):
  optimizer_def = optim.Momentum(learning_rate=learning_rate, beta=beta)
  optimizer = optimizer_def.create(model)
  return optimizer


def get_datasets():
  """Load MNIST train and test datasets into memory."""
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.
  return train_ds, test_ds


def train_and_evaluate(model_dir: str, batch_size: int, num_epochs: int,
                       learning_rate: float, momentum: float):
  """Runs a training and an evaluation loop.

  Args:
    model_dir:
    batch_size:
    num_epochs:
    learning_rate:
    momentum:
  """
  train_ds, test_ds = get_datasets()
  rng = jax.random.PRNGKey(0)

  model = create_model(rng)
  optimizer = create_optimizer(model=model, learning_rate=learning_rate, beta=momentum)



  summary_writer = tensorboard.SummaryWriter(model_dir)
