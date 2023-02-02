# Copyright 2022 The Flax Authors.
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

import jax
from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental.maps import Mesh
from orbax import checkpoint as orbax


PyTree = type(jax.tree_util.tree_structure(None))


def is_multiprocess_array(value: Any) -> bool:
  """Use GlobalAsyncCheckpointManager to save the array if it's only partially available on this host."""
  if isinstance(value, GlobalDeviceArray):
    return True
  if jax.config.jax_array and isinstance(value, jax.Array):
    return not value.is_fully_addressable
  return False


def save_args_from_target(target: Any) -> Any:
  return jax.tree_util.tree_map(
      lambda x: orbax.SaveArgs(aggregate=not is_multiprocess_array(x)), target
  )


def restore_args_from_target(target: Any, mesh: Optional[Mesh]) -> Any:
  def find_axes(x):
    if is_multiprocess_array(x):
      if isinstance(x, GlobalDeviceArray):
        return x.mesh_axes
      return x.sharding.spec
    return None
  if not any(
      jax.tree_util.tree_flatten(jax.tree_map(is_multiprocess_array, target))[0]
  ):
    return jax.tree_util.tree_map(lambda x: orbax.RestoreArgs(), target)
  assert (
      mesh is not None
  ), 'Argument `mesh` required because `target` contains multiprocess array.'
  axes_tree = jax.tree_util.tree_map(find_axes, target)
  return orbax.checkpoint_utils.restore_args_from_target(
      mesh, target, axes_tree
  )
