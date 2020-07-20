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

# Lint as: python3
"""Flax implementation of ResNet V1.
"""


from flax import linen as nn

import jax.numpy as jnp

from functools import partial
from typing import Any


class ResidualBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  strides: (int, int) = (1, 1)
  train: bool = True
  dtype: Any = jnp.float32

  def __call__(self, x):
    needs_projection = x.shape[-1] != self.filters * 4 or self.strides != (1, 1)
    batch_norm = partial(nn.BatchNorm, self, use_running_average=not self.train,
                                       momentum=0.9, epsilon=1e-5,
                                       dtype=self.dtype)
    conv = partial(nn.Conv, self, use_bias=False, dtype=self.dtype)

    residual = x
    if needs_projection:
      residual = conv(self.filters * 4, (1, 1), self.strides, name='proj_conv')(residual)
      residual = batch_norm(name='proj_bn')(residual)

    y = conv(self.filters, (1, 1), name='conv1')(x)
    y = batch_norm(name='bn1')(y)
    y = nn.relu(y)
    y = conv(self.filters, (3, 3), self.strides, name='conv2')(y)
    y = batch_norm(name='bn2')(y)
    y = nn.relu(y)
    y = conv(self.filters * 4, (1, 1), name='conv3')(y)

    y = batch_norm(name='bn3', scale_init=nn.initializers.zeros)(y)
    y = nn.relu(residual + y)
    return y


class ResNet(nn.Module):
  """ResNetV1."""
  num_classes: int
  num_filters: int = 64
  num_layers: int = 50
  train: bool = True
  dtype: Any = jnp.float32

  def __call__(self, x):
    if self.num_layers not in _block_size_options:
      raise ValueError('Please provide a valid number of layers')
    block_sizes = _block_size_options[self.num_layers]
    x = nn.Conv(self, self.num_filters, (7, 7), (2, 2),
                padding=[(3, 3), (3, 3)],
                use_bias=False,
                dtype=self.dtype,
                name='init_conv')(x)
    x = nn.BatchNorm(self, use_running_average=not self.train,
                     momentum=0.9, epsilon=1e-5,
                     dtype=self.dtype,
                     name='init_bn')(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(block_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = ResidualBlock(self, self.num_filters * 2 ** i,
                          strides=strides,
                          train=self.train,
                          dtype=self.dtype)(x)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self, self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, jnp.float32)
    x = nn.log_softmax(x)
    return x


# a dictionary mapping the number of layers in a resnet to the number of blocks
# in each stage of the model.
_block_size_options = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3]
}
