# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Utils for positional embeddings (including RoPE)."""

import jax
import jax.numpy as jnp

_MAX_WAVELENGTH = 10_000


def add_positional_embedding(
    input_embedding: jax.Array,
    position: int,
    max_wavelength: int = _MAX_WAVELENGTH,
) -> jax.Array:
  """Adds positional embeddings to input embeddings."""
  embed_dim = input_embedding.shape[-1]
  num_timescales = embed_dim // 2
  log_timescale_increment = jnp.log(float(max_wavelength)) / jnp.maximum(
      jnp.asarray(num_timescales, dtype=jnp.float32) - 1, 1
  )
  inv_timescales = jnp.exp(
      jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment
  )
  scaled_time = position * inv_timescales
  signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)])
  signal = jnp.pad(signal, [[0, jnp.mod(embed_dim, 2)]])
  position_embedding = signal.astype(jnp.float32)

  return input_embedding + position_embedding


def apply_rope(
    inputs: jax.Array,  # [B, L]
    positions: jax.Array,  # [B, L]
    head_dim: int,
    max_wavelength: int = _MAX_WAVELENGTH,
    scale_factor: float = 1.0,
) -> jax.Array:
  """Applies RoPE."""
  fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
  timescale = max_wavelength**fraction

  sinusoid_inp = (
      positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
  )
  sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
  if scale_factor < 1.0:
    raise ValueError(f'scale_factor must be >= 1.0, got {scale_factor}')
  sinusoid_inp /= scale_factor

  sin = jnp.sin(sinusoid_inp)
  cos = jnp.cos(sinusoid_inp)

  first_half, second_half = jnp.split(inputs, 2, axis=-1)
  first_part = first_half * cos - second_half * sin
  second_part = second_half * cos + first_half * sin
  out = jnp.concatenate([first_part, second_part], axis=-1)
  return out.astype(inputs.dtype)
