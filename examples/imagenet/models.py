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


from flax import nn

import jax.numpy as jnp


class ResidualBlock(nn.Module):
  """Bottleneck ResNet block."""

  def apply(self, x, filters, strides=(1, 1), train=True, dtype=jnp.float32):
    needs_projection = x.shape[-1] != filters * 4 or strides != (1, 1)
    batch_norm = nn.BatchNorm.partial(use_running_average=not train,
                                      momentum=0.9, epsilon=1e-5,
                                      dtype=dtype)
    conv = nn.Conv.partial(bias=False, dtype=dtype)

    residual = x
    if needs_projection:
      residual = conv(residual, filters * 4, (1, 1), strides, name='proj_conv')
      residual = batch_norm(residual, name='proj_bn')

    y = conv(x, filters, (1, 1), name='conv1')
    y = batch_norm(y, name='bn1')
    y = nn.relu(y)
    y = conv(y, filters, (3, 3), strides, name='conv2')
    y = batch_norm(y, name='bn2')
    y = nn.relu(y)
    y = conv(y, filters * 4, (1, 1), name='conv3')

    y = batch_norm(y, name='bn3', scale_init=nn.initializers.zeros)
    y = nn.relu(residual + y)
    return y


class ResNet(nn.Module):
  """ResNetV1."""

  def apply(self, x, num_classes, num_filters=64, num_layers=50,
            train=True, dtype=jnp.float32):
    if num_layers not in _block_size_options:
      raise ValueError('Please provide a valid number of layers')
    block_sizes = _block_size_options[num_layers]
    x = nn.Conv(x, num_filters, (7, 7), (2, 2),
                padding=[(3, 3), (3, 3)],
                bias=False,
                dtype=dtype,
                name='init_conv')
    x = nn.BatchNorm(x,
                     use_running_average=not train,
                     momentum=0.9, epsilon=1e-5,
                     dtype=dtype,
                     name='init_bn')
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(block_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = ResidualBlock(x, num_filters * 2 ** i,
                          strides=strides,
                          train=train,
                          dtype=dtype)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(x, num_classes, dtype=dtype)
    x = jnp.asarray(x, dtype)
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
