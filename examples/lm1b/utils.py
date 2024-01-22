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

# Copied over from MaxText (https://github.com/google/maxtext/blob/main/MaxText/max_utils.py).

import functools
import logging

import numpy as np
import flax.linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils


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
  logging.info(f"Devices: {devices}")
  logging.info(f"Number of devices: {num_devices}")

  multi_slice_env = hasattr(jax.devices()[0], "slice_index")

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
      dcn_parallelism, num_slices, "DCN"
  )
  ici_parallelism = fill_unspecified_mesh_axes(
      ici_parallelism, num_devices_per_slice, "ICI"
  )

  if multi_slice_env:
    mesh = mesh_utils.create_hybrid_device_mesh(
        ici_parallelism, dcn_parallelism
    )
  else:
    mesh = mesh_utils.create_device_mesh(ici_parallelism)

  logging.info(f"Decided on mesh: {mesh}")
  logging.info(f"Mesh shape: {mesh.shape}")

  return mesh


def fill_unspecified_mesh_axes(
    parallelism_vals, target_product, parallelism_type
):
  """Evaluates unspecified DCN/ICI parallelism values"""
  if -1 in parallelism_vals:
    assert parallelism_vals.count(-1) == 1, (
        f"Found unspecified values (-1) for more than one {parallelism_type}   "
        "   parallelism axis. At most one axis can be unspecified."
    )

    determined_val = target_product / np.prod(parallelism_vals) * -1

    assert determined_val >= 1 and determined_val.is_integer, (
        "Unspecified value unable to be determined with the given     "
        f" {parallelism_type} parallelism values"
    )

    parallelism_vals[parallelism_vals.index(-1)] = int(determined_val)

  target_type = "slices" if parallelism_type == "DCN" else "devices per slice"

  assert np.prod(parallelism_vals) == target_product, (
      f"Number of {target_type} {target_product} does not match    the product"
      f" of the {parallelism_type} parallelism {np.prod(parallelism_vals)}"
  )

  return parallelism_vals


# State initialization utils.
# -----------------------------------------------------------------------------


def unbox_logicallypartioned_trainstate(
    boxed_train_state: train_state.TrainState,
):
  """Unboxes the flax.LogicallyPartitioned pieces in a train state.

  Args:
    boxed_train_state: a train state that includes LogicallyPartitioned
      leaves.
  Returns:
    a TrainState where all all LogicallyPartitioned leaves have been unboxed.
  """
  return jax.tree_util.tree_map(
      lambda x: x.unbox() if isinstance(x, nn.spmd.LogicallyPartitioned) else x,
      boxed_train_state,
      is_leaf=lambda k: isinstance(k, nn.spmd.LogicallyPartitioned),
  )


def init_train_state(model, tx, config, key):
  """
  We pass in "static" objects like model, tx, config as JAX compares them by
  object hash, and instantiating them inside causes pjit top-level annotations
  to fail to match as pytree prefixes if we re-instantiate.

  Args: model, tx, config, key
  """
  input_shape = (config.per_device_batch_size, config.max_target_length)
  initial_variables = jax.jit(model.init)(
      key, jnp.ones(input_shape, jnp.float32)
  )

  state = train_state.TrainState.create(
      apply_fn=model.apply, params=initial_variables["params"], tx=tx
  )
  return state


def setup_initial_state(model, tx, config, rng, mesh):
  """We initialize the model and optimizer state, and optionally load from a
  checkpoint as necessary.

  Args:
    model: the flax model to initialize
    tx: the optax.GradientTransformation
    config: config object
    rng: jax.prng key
    mesh: jax.devices() mesh

  Returns:
    state: the initialized train state
    state_mesh_annotations: the mesh annotations for the train state
  """
  init_train_state_partial = functools.partial(
      init_train_state, model, tx, config
  )
  abstract_state = jax.eval_shape(init_train_state_partial, rng)
  state_logical_annotations = nn.get_partition_spec(abstract_state)

  # Initialization
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh_sharding(
        state_logical_annotations, mesh, config.logical_axis_rules
    )
    state = jax.jit(
        init_train_state_partial,
        in_shardings=None,  # type: ignore
        out_shardings=state_mesh_annotations,
    )(rng)

  state = unbox_logicallypartioned_trainstate(state)
  return state, state_mesh_annotations
