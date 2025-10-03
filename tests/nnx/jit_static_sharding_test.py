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

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from flax import nnx


def _mesh_data():
  devices = mesh_utils.create_device_mesh((jax.device_count(),))
  return Mesh(devices, axis_names=("data",))


def _inputs_with_negatives():
  # Use inputs with negative values so ReLU has an effect
  x = jnp.linspace(-1.0, 1.0, 16, dtype=jnp.float32).reshape(4, 4)
  scale = 0.5
  return x, scale


def _fn(x, scale: float, use_relu: bool):
  y = x * scale
  if use_relu:
    y = jnp.maximum(y, 0)
  return y.sum()


def test_jit_static_no_static_in_shardings():
  mesh = _mesh_data()
  x_sharding = NamedSharding(mesh, P("data"))

  f = nnx.jit(_fn, in_shardings=(x_sharding, None), static_argnums=(2,))

  x, scale = _inputs_with_negatives()
  y_relu = f(x, scale, True)
  y_no_relu = f(x, scale, False)

  # verify static arg toggles result
  assert y_relu != y_no_relu


def test_jit_static_with_extra_static_entry():
  mesh = _mesh_data()
  x_sharding = NamedSharding(mesh, P("data"))

  g = nnx.jit(
    _fn, in_shardings=(x_sharding, None, None), static_argnums=(2,)
  )

  x, scale = _inputs_with_negatives()
  y_relu = g(x, scale, True)
  y_no_relu = g(x, scale, False)

  # verify static arg toggles result even if provided an extra sharding entry
  assert y_relu != y_no_relu

