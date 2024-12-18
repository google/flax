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
# pytype: skip-file


import io
import typing as tp
from itertools import groupby
from types import MappingProxyType

import jax
import rich.console
import rich.table
import rich.text
import yaml
import jax.numpy as jnp

from flax.nnx import graph, rnglib, variablelib


def tabulate(
  obj,
  depth: int = 2,
  table_kwargs: tp.Mapping[str, tp.Any] = MappingProxyType({}),
  column_kwargs: tp.Mapping[str, tp.Any] = MappingProxyType({}),
  console_kwargs: tp.Mapping[str, tp.Any] = MappingProxyType({}),
) -> str:
  """Create a table summary of the states.

  Args:
    obj: The object to summarize.
    depth: The depth of the table.

  Returns:
    A string representing the states in a table.
  """
  state = graph.state(obj)
  graph_map = dict(graph.iter_graph(obj))
  flat_state = sorted(state.flat_state())

  def key_fn(
    path_state: tuple[graph.PathParts, variablelib.VariableState[tp.Any]],
  ):
    path, _ = path_state
    if len(path) <= depth:
      return path[:-1]
    return path[:depth]

  rows = groupby(flat_state, key_fn)
  table = sorted((path, list(flat_states)) for path, flat_states in rows)

  state_types = {variable_state.type for _, variable_state in flat_state}
  # replace RngKey and RngCount with RngState
  if rnglib.RngKey in state_types:
    state_types.remove(rnglib.RngKey)
    state_types.add(rnglib.RngState)
  if rnglib.RngCount in state_types:
    state_types.remove(rnglib.RngCount)
    state_types.add(rnglib.RngState)
  # sort based on MRO
  state_types = _sort_variable_types(state_types)

  rich_table = rich.table.Table(
    show_header=True,
    show_lines=True,
    show_footer=True,
    title=f'{type(obj).__name__} Summary',
    **table_kwargs,
  )

  rich_table.add_column('path', **column_kwargs)
  rich_table.add_column('type', **column_kwargs)

  for state_type in state_types:
    rich_table.add_column(state_type.__name__, **column_kwargs)

  for key_path, row_states in table:
    row: list[str] = []
    node = graph_map[key_path]
    type_state_groups = variablelib.split_flat_state(row_states, state_types)
    path_str = '/'.join(map(str, key_path))
    node_type = type(node).__name__
    row.extend([path_str, node_type])

    for state_type, type_path_and_states in zip(state_types, type_state_groups):
      attributes = {}
      for state_path, variable_state in type_path_and_states:
        if len(state_path) == len(key_path) + 1:
          name = str(state_path[-1])
          value = variable_state.value
          value_repr = _render_array(value) if _has_shape_dtype(value) else ''
          metadata = variable_state.get_metadata()

          if metadata:
            attributes[name] = {
              'value': value_repr,
              **metadata,
            }
          elif value_repr:
            attributes[name] = value_repr

      if attributes:
        col_repr = _as_yaml_str(attributes) + '\n\n'
      else:
        col_repr = ''

      type_states = [state for _, state in type_path_and_states]
      size_, bytes_ = _size_and_bytes(type_states)
      col_repr += f'[bold]{_size_and_bytes_repr(size_, bytes_)}[/bold]'
      row.append(col_repr)

    rich_table.add_row(*row)

  rich_table.columns[1].footer = rich.text.Text.from_markup(
    'Total', justify='right'
  )
  flat_states = variablelib.split_flat_state(flat_state, state_types)

  for i, (state_type, type_path_and_states) in enumerate(
    zip(state_types, flat_states)
  ):
    type_states = [state for _, state in type_path_and_states]
    size_, bytes_ = _size_and_bytes(type_states)
    size_repr = _size_and_bytes_repr(size_, bytes_)
    rich_table.columns[i + 2].footer = size_repr

  rich_table.caption_style = 'bold'
  rich_table.caption = (
    f'\nTotal Parameters: {_size_and_bytes_repr(*_size_and_bytes(state))}'
  )

  return '\n' + _get_rich_repr(rich_table, console_kwargs) + '\n'


def _get_rich_repr(obj, console_kwargs):
  f = io.StringIO()
  console = rich.console.Console(file=f, **console_kwargs)
  console.print(obj)
  return f.getvalue()


def _size_and_bytes(pytree: tp.Any) -> tuple[int, int]:
  leaves = jax.tree.leaves(pytree)
  size = sum(x.size for x in leaves if hasattr(x, 'size'))
  num_bytes = sum(
    x.size * x.dtype.itemsize for x in leaves if hasattr(x, 'size')
  )
  return size, num_bytes


def _size_and_bytes_repr(size: int, num_bytes: int) -> str:
  if not size:
    return ''
  bytes_repr = _bytes_repr(num_bytes)
  return f'{size:,} [dim]({bytes_repr})[/dim]'


def _bytes_repr(num_bytes):
  count, units = (
    (f'{num_bytes / 1e9 :,.1f}', 'GB')
    if num_bytes > 1e9
    else (f'{num_bytes / 1e6 :,.1f}', 'MB')
    if num_bytes > 1e6
    else (f'{num_bytes / 1e3 :,.1f}', 'KB')
    if num_bytes > 1e3
    else (f'{num_bytes:,}', 'B')
  )

  return f'{count} {units}'


def _has_shape_dtype(value):
  return hasattr(value, 'shape') and hasattr(value, 'dtype')


def _as_yaml_str(value) -> str:
  if (hasattr(value, '__len__') and len(value) == 0) or value is None:
    return ''

  file = io.StringIO()
  yaml.safe_dump(
    value,
    file,
    default_flow_style=False,
    indent=2,
    sort_keys=False,
    explicit_end=False,
  )
  return file.getvalue().replace('\n...', '').replace("'", '').strip()


def _render_array(x):
  shape, dtype = jnp.shape(x), jnp.result_type(x)
  shape_repr = ','.join(str(x) for x in shape)
  return f'[dim]{dtype}[/dim][{shape_repr}]'


def _sort_variable_types(types: tp.Iterable[type]):
  def _variable_parents_count(t: type):
    return sum(1 for p in t.mro() if issubclass(p, variablelib.Variable))

  type_sort_key = {t: (-_variable_parents_count(t), t.__name__) for t in types}
  return sorted(types, key=lambda t: type_sort_key[t])
