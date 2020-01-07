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

"""Common utils used in Flax examples."""

import jax
from jax import lax
import jax.numpy as jnp
import numpy as onp


def shard(xs):
  local_device_count = jax.local_device_count()
  return jax.tree_map(
      lambda x: x.reshape((local_device_count, -1) + x.shape[1:]), xs)


def onehot(labels, num_classes):
  x = (labels[..., None] == jnp.arange(num_classes)[None])
  return x.astype(jnp.float32)


def pmean(tree, axis_name='batch'):
  num_devices = lax.psum(1., axis_name)
  return jax.tree_map(lambda x: lax.psum(x, axis_name) / num_devices, tree)


def stack_forest(forest):
  stack_args = lambda *args: onp.stack(args)
  return jax.tree_multimap(stack_args, *forest)


def get_metrics(device_metrics):
  device_metrics = jax.tree_map(lambda x: x[0], device_metrics)
  metrics_np = jax.device_get(device_metrics)
  return stack_forest(metrics_np)
