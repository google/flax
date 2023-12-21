# Copyright 2023 The Flax Authors.
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

# Copied over from MaxText (https://github.com/google/maxtext/blob/main/MaxText/max_utils.py).

import logging
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from models import TransformerConfig, TransformerLM

from flax.experimental import nnx
from flax.training import train_state


class TrainState(train_state.TrainState):
  graphdef: nnx.GraphDef[TransformerLM]


# Mesh utils.
# -----------------------------------------------------------------------------


def create_device_mesh(config):
  """Creates a device mesh with each slice in its own data parallel group. If there is only one slice, uses two replicas."""
  devices = jax.devices()
  num_devices = len(devices)
  try:
    num_slices = 1 + max([d.slice_index for d in devices])
  except:
    num_slices = 1
  num_devices_per_slice = num_devices // num_slices
  logging.info(f'Devices: {devices}')
  logging.info(f'Number of devices: {num_devices}')

  multi_slice_env = hasattr(jax.devices()[0], 'slice_index')

  dcn_parallelism = [
    config.dcn_data_parallelism,
    config.dcn_fsdp_parallelism,
    config.dcn_tensor_parallelism,
  ]
  ici_parallelism = [
    config.ici_data_parallelism,
    config.ici_fsdp_parallelism,
    config.ici_tensor_parallelism,
  ]

  # Find possible unspecified parallelisms
  dcn_parallelism = fill_unspecified_mesh_axes(
    dcn_parallelism, num_slices, 'DCN'
  )
  ici_parallelism = fill_unspecified_mesh_axes(
    ici_parallelism, num_devices_per_slice, 'ICI'
  )

  if multi_slice_env:
    mesh = mesh_utils.create_hybrid_device_mesh(
      ici_parallelism, dcn_parallelism
    )
  else:
    mesh = mesh_utils.create_device_mesh(ici_parallelism)

  logging.info(f'Decided on mesh: {mesh}')
  logging.info(f'Mesh shape: {mesh.shape}')

  return mesh


def fill_unspecified_mesh_axes(
  parallelism_vals, target_product, parallelism_type
):
  """Evaluates unspecified DCN/ICI parallelism values"""
  if -1 in parallelism_vals:
    assert parallelism_vals.count(-1) == 1, (
      f'Found unspecified values (-1) for more than one {parallelism_type}   '
      '   parallelism axis. At most one axis can be unspecified.'
    )

    determined_val = target_product / np.prod(parallelism_vals) * -1

    assert determined_val >= 1 and determined_val.is_integer, (
      'Unspecified value unable to be determined with the given     '
      f' {parallelism_type} parallelism values'
    )

    parallelism_vals[parallelism_vals.index(-1)] = int(determined_val)

  target_type = 'slices' if parallelism_type == 'DCN' else 'devices per slice'

  assert np.prod(parallelism_vals) == target_product, (
    f'Number of {target_type} {target_product} does not match    the product'
    f' of the {parallelism_type} parallelism {np.prod(parallelism_vals)}'
  )

  return parallelism_vals


# State initialization utils.
# -----------------------------------------------------------------------------


def _to_array(x):
  if not isinstance(x, jax.Array):
    x = jnp.asarray(x)
  return x


def setup_initial_state(
  constructor: Callable[[TransformerConfig, jax.Array], TransformerLM],
  tx,
  config: TransformerConfig,
  rng: jax.Array,
  mesh: jax.sharding.Mesh,
) -> tuple[TrainState, TrainState]:
  """We initialize the model and optimizer state, and optionally load from a
  checkpoint as necessary.

  Args:
    constructor: the model constructor
    tx: the optax.GradientTransformation
    config: config object
    rng: jax.prng key
    mesh: jax.devices() mesh

  Returns:
    state: the initialized train state
    state_mesh_annotations: the mesh annotations for the train state
  """

  # Initialization

  with mesh:
    model = constructor(config, rng)
    params, static = model.split(nnx.Param)
    state = TrainState.create(
      apply_fn=static.apply, params=params, tx=tx, graphdef=static
    )
    state = jax.tree_map(_to_array, state)
    state_spec = nnx.get_partition_spec(state)
    state = jax.lax.with_sharding_constraint(state, state_spec)

  state_sharding = nnx.get_named_sharding(state, mesh)
  return state, state_sharding
