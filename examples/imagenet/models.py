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

"""Flax implementation of ResNet V1.
"""


from flax import nn

import jax.numpy as jnp


class ResNetBlock(nn.Module):
  """ResNet block."""

  def apply(self, x, filters, *,
            conv, norm, act,
            strides=(1, 1)):
    residual = x
    y = conv(x, filters, (3, 3), strides)
    y = norm(y)
    y = act(y)
    y = conv(y, filters, (3, 3))
    y = norm(y, scale_init=nn.initializers.zeros)

    if residual.shape != y.shape:
      residual = conv(residual, filters, (1, 1), strides, name='conv_proj')
      residual = norm(residual, name='norm_proj')

    return act(residual + y)


class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""

  def apply(self, x, filters, *,
            conv, norm, act,
            strides=(1, 1)):
    residual = x
    y = conv(x, filters, (1, 1))
    y = norm(y)
    y = act(y)
    y = conv(y, filters, (3, 3), strides)
    y = norm(y)
    y = act(y)
    y = conv(y, filters * 4, (1, 1))
    y = norm(y, scale_init=nn.initializers.zeros)

    if residual.shape != y.shape:
      residual = conv(residual, filters * 4, (1, 1), strides, name='conv_proj')
      residual = norm(residual, name='norm_proj')

    return act(residual + y)


class ResNet(nn.Module):
  """ResNetV1."""

  def apply(self, x, num_classes, *,
            stage_sizes,
            block_cls,
            num_filters=64,
            dtype=jnp.float32,
            act=nn.relu,
            train=True):
    conv = nn.Conv.partial(bias=False, dtype=dtype)
    norm = nn.BatchNorm.partial(
        use_running_average=not train,
        momentum=0.9, epsilon=1e-5,
        dtype=dtype)

    x = conv(x, num_filters, (7, 7), (2, 2),
             padding=[(3, 3), (3, 3)],
             name='conv_init')
    x = norm(x, name='bn_init')
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = block_cls(x, num_filters * 2 ** i,
                      strides=strides,
                      conv=conv,
                      norm=norm,
                      act=act)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(x, num_classes, dtype=dtype)
    x = jnp.asarray(x, dtype)
    x = nn.log_softmax(x)
    return x


ResNet18 = ResNet.partial(stage_sizes=[2, 2, 2, 2],
                          block_cls=ResNetBlock)
ResNet34 = ResNet.partial(stage_sizes=[3, 4, 6, 3],
                          block_cls=ResNetBlock)
ResNet50 = ResNet.partial(stage_sizes=[3, 4, 6, 3],
                          block_cls=BottleneckResNetBlock)
ResNet101 = ResNet.partial(stage_sizes=[3, 4, 23, 3],
                           block_cls=BottleneckResNetBlock)
ResNet152 = ResNet.partial(stage_sizes=[3, 8, 36, 3],
                           block_cls=BottleneckResNetBlock)
ResNet200 = ResNet.partial(stage_sizes=[3, 24, 36, 3],
                           block_cls=BottleneckResNetBlock)
