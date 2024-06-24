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
"""Helper functions."""

from __future__ import annotations

from typing import Callable, Optional, Tuple, TypeVar, Union
import flax
from flax import nnx
from flax.typing import VariableDict  # pylint: disable=g-importing-member,g-multiple-import

M = TypeVar('M', bound='nnx.Module')


def _flatten_path(path: Tuple[Union[str, int], ...]) -> str:
  def f(item) -> str:
    if isinstance(item, str):
      return f'{item}'
    elif isinstance(item, int):
      return f'[{item}]'
    else:
      raise ValueError(f'Unexpected type {type(item)}')

  return '.'.join([f(item) for item in path]).replace('.[', '[')


def module_from_linen_variables(
    module_factory: Callable[[], M],
    variables: VariableDict,
    map_key_fn: Optional[
        Callable[[Tuple[str, ...]], Tuple[Union[str, int], ...]]
    ] = None,
) -> M:
  """Returns an `nnx.Module` initialized with the `variables` of a linen module.

  Args:
    module_factory: A no-args callable that returns an `nnx.Module`.
    variables: A dictionary of variables.
    map_key_fn: An optional function for mapping keys in the `variables`
      dictionary to keys in the `nnx.Module`'s state. If not provided it is
      assumed that after removing the collection name the keys in the
      `variables` dictionary are the same as the keys in the `nnx.Module`'s
      state.
  """
  if map_key_fn is None:

    def map_key_fn(path: Tuple[str, ...]) -> Tuple[Union[str, int], ...]:
      return path[1:] if 'params' in variables else path

  mdl: M = nnx.eval_shape(module_factory)
  graph_def, state = nnx.split(mdl)
  state = state.flat_state()
  for path, val in flax.traverse_util.flatten_dict(variables).items():
    mapped_path = map_key_fn(path)
    if mapped_path not in state:
      raise ValueError(
          f"'{mdl.__class__.__name__}.{_flatten_path(mapped_path)}' doesn't "
          f' exist (original path={path}).'
      )
    state[mapped_path].value = val
  state = nnx.State.from_flat_path(state)

  return nnx.merge(graph_def, state)
