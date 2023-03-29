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

"""Utils for Orbax Checkpointing, available even after Flax Checkpointing is deprecated."""

from typing import Any, Optional
import warnings

import jax
from jax.sharding import Mesh
from orbax import checkpoint as orbax


PyTree = Any


def is_multiprocess_array(value: Any) -> bool:
  """Use GlobalAsyncCheckpointManager to save the array if it's only partially available on this host."""
  if isinstance(value, jax.Array):
    return not value.is_fully_addressable
  return False


def save_args_from_target(target: Any) -> Any:
  return jax.tree_util.tree_map(
      lambda x: orbax.SaveArgs(aggregate=not is_multiprocess_array(x)), target
  )


def restore_args_from_target(target: Any, mesh: Optional[Mesh] = None) -> Any:
  """Creates Orbax `restore_args` given a target Pytree.

  Args:
    target: The Pytree that has the same structure as the checkpoint. The arrays
      restored from checkpoint will have the same `sharding` as the target
      Pytree's corresponding arrays.
    mesh: DEPRECATED ARG. Please simply use your mesh to create the arrays
      in your `target`, no need to pass it here.

  Returns:
    A Pytree of Orbax `RestoreArgs` or `ArrayRestoreArgs`
  """
  def find_sharding(x):
    if is_multiprocess_array(x):
      return x.sharding
    return None

  # Simpler case: no multihost arrays
  if not any(
      jax.tree_util.tree_flatten(jax.tree_map(is_multiprocess_array, target))[0]
  ):
    return jax.tree_util.tree_map(lambda x: orbax.RestoreArgs(), target)

  # Multihost arrays: find sharding from the given target
  sharding_tree = jax.tree_util.tree_map(find_sharding, target)
  if mesh is not None:
    warnings.warn(
        (
            'restore_args_from_target(): `mesh` arg is deprecated. Simply'
            ' calling the function with target pytree should suffice.'
        ),
        DeprecationWarning,
    )
    axes_tree = jax.tree_util.tree_map(lambda s: s.spec, sharding_tree)
    return orbax.checkpoint_utils.restore_args_from_target(
        mesh, target, axes_tree
    )
  return orbax.checkpoint_utils.construct_restore_args(target, sharding_tree)
