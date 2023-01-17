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

"""Flax Module summary library."""
from abc import ABC, abstractmethod
import dataclasses
import io
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Type, Union

import flax.linen.module as module_lib
from flax.core import meta
from flax.core.scope import CollectionFilter, FrozenVariableDict, MutableVariableDict
import jax
import jax.numpy as jnp
import rich.console
import rich.table
import rich.text
import yaml
import numpy as np

PRNGKey = Any  # pylint: disable=invalid-name
RNGSequences = Dict[str, PRNGKey]
Array = Any    # pylint: disable=invalid-name

class _ValueRepresentation(ABC):
  """A class that represents a value in the summary table."""

  @abstractmethod
  def render(self) -> str:
    ...

@dataclasses.dataclass
class _ArrayRepresentation(_ValueRepresentation):
  shape: Tuple[int, ...]
  dtype: Any

  @classmethod
  def from_array(cls, x: Array) -> '_ArrayRepresentation':
    return cls(jnp.shape(x), jnp.result_type(x))

  @classmethod
  def render_array(cls, x) -> str:
    return cls.from_array(x).render()

  def render(self):
    shape_repr = ','.join(str(x) for x in self.shape)
    return f'[dim]{self.dtype}[/dim][{shape_repr}]'

@dataclasses.dataclass
class _PartitionedArrayRepresentation(_ValueRepresentation):
  array_representation: _ArrayRepresentation
  names: meta.LogicalNames

  @classmethod
  def from_partitioned(cls, partitioned: meta.Partitioned) -> '_PartitionedArrayRepresentation':
    return cls(_ArrayRepresentation.from_array(partitioned.value), partitioned.names)

  def render(self):
    return self.array_representation.render() + f' [dim]P[/dim]{self.names}'

@dataclasses.dataclass
class _ObjectRepresentation(_ValueRepresentation):
  obj: Any

  def render(self):
    return repr(self.obj)

@dataclasses.dataclass
class Row:
  """Contains the information about a single row in the summary table.

  Attributes:
    path: A tuple of strings that represents the path to the module.
    outputs: Output of the Module as reported by `capture_intermediates`.
    module_variables: Dictionary of variables in the module (no submodules
      included).
    counted_variables: Dictionary of variables that should be counted for this
      row, if no summarization is done (e.g. `depth=None` in `module_summary`)
      then this field is the same as `module_variables`, however if a
      summarization is done then this dictionary potentially contains parameters
      from submodules depending on the depth of the Module in question.
  """
  path: Tuple[str, ...]
  module_type: Type[module_lib.Module]
  method: str
  inputs: Any
  outputs: Any
  module_variables: Dict[str, Dict[str, Any]]
  counted_variables: Dict[str, Dict[str, Any]]

  def __post_init__(self):
    self.inputs = self.inputs
    self.outputs = self.outputs
    self.module_variables = self.module_variables
    self.counted_variables = self.counted_variables

  def size_and_bytes(self, collections: Iterable[str]) -> Dict[str, Tuple[int, int]]:
    return {
        col: _size_and_bytes(self.counted_variables[col])
        if col in self.counted_variables else (0, 0) for col in collections
    }


class Table(List[Row]):
  """A list of Row objects.

  Table inherits from `List[Row]` so it has all the methods of a list, however
  it also contains some additional fields:

  * `module`: the module that this table is summarizing
  * `collections`: a list containing the parameter collections (e.g. 'params', 'batch_stats', etc)
  """

  def __init__(self, module: module_lib.Module, collections: Sequence[str],
               rows: Iterable[Row]):
    super().__init__(rows)
    self.module = module
    self.collections = collections


def tabulate(
  module: module_lib.Module,
  rngs: Union[PRNGKey, RNGSequences],
  depth: Optional[int] = None,
  show_repeated: bool = False,
  mutable: CollectionFilter = True,
  console_kwargs: Optional[Mapping[str, Any]] = None,
  **kwargs,
) -> Callable[..., str]:
  """Returns a function that creates a summary of the Module represented as a table.

  This function accepts most of the same arguments and internally calls `Module.init`,
  except that it returns a function of the form `(*args, **kwargs) -> str` where `*args`
  and `**kwargs` are passed to `method` (e.g. `__call__`) during the forward pass.

  `tabulate` uses `jax.eval_shape` under the hood to run the forward computation without
  consuming any FLOPs or allocating memory.

  Additional arguments can be passed into the `console_kwargs` argument, for example,
  `{'width': 120}`. For a full list of `console_kwargs` arguments, see:
  https://rich.readthedocs.io/en/stable/reference/console.html#rich.console.Console

  Example::

    import jax
    import jax.numpy as jnp
    import flax.linen as nn

    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        h = nn.Dense(4)(x)
        return nn.Dense(2)(h)

    x = jnp.ones((16, 9))
    tabulate_fn = nn.tabulate(Foo(), jax.random.PRNGKey(0))

    print(tabulate_fn(x))


  This gives the following output::

                                    Foo Summary
    ┏━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ path    ┃ module ┃ inputs        ┃ outputs       ┃ params               ┃
    ┡━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
    │         │ Foo    │ float32[16,9] │ float32[16,2] │                      │
    ├─────────┼────────┼───────────────┼───────────────┼──────────────────────┤
    │ Dense_0 │ Dense  │ float32[16,9] │ float32[16,4] │ bias: float32[4]     │
    │         │        │               │               │ kernel: float32[9,4] │
    │         │        │               │               │                      │
    │         │        │               │               │ 40 (160 B)           │
    ├─────────┼────────┼───────────────┼───────────────┼──────────────────────┤
    │ Dense_1 │ Dense  │ float32[16,4] │ float32[16,2] │ bias: float32[2]     │
    │         │        │               │               │ kernel: float32[4,2] │
    │         │        │               │               │                      │
    │         │        │               │               │ 10 (40 B)            │
    ├─────────┼────────┼───────────────┼───────────────┼──────────────────────┤
    │         │        │               │         Total │ 50 (200 B)           │
    └─────────┴────────┴───────────────┴───────────────┴──────────────────────┘

                          Total Parameters: 50 (200 B)


  **Note**: rows order in the table does not represent execution order,
  instead it aligns with the order of keys in `variables` which are sorted
  alphabetically.

  Args:
    module: The module to tabulate.
    rngs: The rngs for the variable collections as passed to `Module.init`.
    depth: controls how many submodule deep the summary can go. By default its
      `None` which means no limit. If a submodule is not shown because of the
      depth limit, its parameter count and bytes will be added to the row of its
      first shown ancestor such that the sum of all rows always adds up to the
      total number of parameters of the Module.
    mutable: Can be bool, str, or list. Specifies which collections should be
      treated as mutable: ``bool``: all/no collections are mutable. ``str``: The
      name of a single mutable collection. ``list``: A list of names of mutable
      collections. By default all collections except 'intermediates' are
      mutable.
    show_repeated: If `True`, repeated calls to the same module will be shown
      in the table, otherwise only the first call will be shown. Default is
      `False`.
    console_kwargs: An optional dictionary with additional keyword arguments that
      are passed to `rich.console.Console` when rendering the table. Default arguments
      are `{'force_terminal': True, 'force_jupyter': False}`.
    **kwargs: Additional arguments passed to `Module.init`.

  Returns:
    A function that accepts the same `*args` and `**kwargs` of the forward pass
    (`method`) and returns a string with a tabular representation of the
    Modules.
  """

  def _tabulate_fn(*fn_args, **fn_kwargs):
    table_fn = _get_module_table(module, depth=depth, show_repeated=show_repeated)
    table = table_fn(rngs, *fn_args, mutable=mutable, **fn_kwargs, **kwargs)
    return _render_table(table, console_kwargs)

  return _tabulate_fn

def _get_module_table(
    module: module_lib.Module,
    depth: Optional[int],
    show_repeated: bool,
) -> Callable[..., Table]:
  """A function that takes a Module and returns function with the same signature as `init`
  but returns the Table representation of the Module."""

  def _get_table_fn(*args, **kwargs):

    with module_lib._tabulate_context():

      def _get_variables():
        return module.init(*args, **kwargs)

      variables = jax.eval_shape(_get_variables)
      calls = module_lib._context.call_info_stack[-1].calls
      calls.sort(key=lambda c: c.index)

    collections: Set[str] = set(variables.keys())
    rows = []
    all_paths: Set[Tuple[str, ...]] = set(call.path for call in calls)
    visited_paths: Set[Tuple[str, ...]] = set()

    for c in calls:
      call_depth = len(c.path)
      inputs = _process_inputs(c.args, c.kwargs)

      if c.path in visited_paths:
        if not show_repeated:
          continue
        module_vars = {}
        counted_vars = {}
      elif depth is not None:
        if call_depth > depth:
          continue
        module_vars, _ = _get_module_variables(c.path, variables, all_paths)
        if call_depth == depth:
          counted_vars = _get_path_variables(c.path, variables)
        else:
          counted_vars = module_vars
      else:
        module_vars, _ = _get_module_variables(c.path, variables, all_paths)
        counted_vars = module_vars

      visited_paths.add(c.path)
      rows.append(
        Row(c.path, c.module_type, c.method, inputs, c.outputs, module_vars, counted_vars))

    return Table(module, tuple(collections), rows)

  return _get_table_fn

def _get_module_variables(
  path: Tuple[str, ...], variables: FrozenVariableDict, all_paths: Set[Tuple[str, ...]]
) -> Tuple[MutableVariableDict, Any]:
  """A function that takes a path and variables structure and returns a
  (module_variables, submodule_variables) tuple for that path. _get_module_variables
  uses the `all_paths` set to determine if a variable belongs to a submodule or not."""
  module_variables = _get_path_variables(path, variables)
  submodule_variables: Any = {collection: {} for collection in module_variables}
  all_keys = set(key for collection in module_variables.values() for key in collection)

  for key in all_keys:
    submodule_path = path + (key,)
    if submodule_path in all_paths:

      for collection in module_variables:
        if key in module_variables[collection]:
          submodule_variables[collection][key] = module_variables[collection].pop(key)

  return module_variables, submodule_variables

def _get_path_variables(path: Tuple[str, ...], variables: FrozenVariableDict) -> MutableVariableDict:
  """A function that takes a path and a variables structure and returns the variable structure at
  that path."""
  path_variables = {}

  for collection in variables:
    collection_variables = variables[collection]
    for name in path:
      if name not in collection_variables:
        collection_variables = None
        break
      collection_variables = collection_variables[name]

    if collection_variables is not None:
      path_variables[collection] = collection_variables.unfreeze()

  return path_variables

def _process_inputs(args, kwargs) -> Any:
  """A function that normalizes the representation of the ``args`` and ``kwargs``
  for the ``inputs`` column."""
  if args and kwargs:
    input_values = (*args, kwargs)
  elif args and not kwargs:
    input_values = args[0] if len(args) == 1 else args
  elif kwargs and not args:
    input_values = kwargs
  else:
    input_values = ()

  return input_values

def _render_table(table: Table, console_extras: Optional[Mapping[str, Any]]) -> str:
  """A function that renders a Table to a string representation using rich."""
  console_kwargs = {'force_terminal': True, 'force_jupyter': False}
  if console_extras is not None:
    console_kwargs.update(console_extras)

  non_params_cols = 4
  rich_table = rich.table.Table(
      show_header=True,
      show_lines=True,
      show_footer=True,
      title=f'{table.module.__class__.__name__} Summary',
  )

  rich_table.add_column('path')
  rich_table.add_column('module')
  rich_table.add_column('inputs')
  rich_table.add_column('outputs')

  for col in table.collections:
    rich_table.add_column(col)

  for row in table:
    collections_size_repr = []

    for collection, size_bytes in row.size_and_bytes(table.collections).items():
      col_repr = ''

      if collection in row.module_variables:
        module_variables = _represent_tree(row.module_variables[collection])
        module_variables = _normalize_structure(module_variables)
        col_repr += _as_yaml_str(
          _summary_tree_map(_maybe_render, module_variables))
        if col_repr:
          col_repr += '\n\n'

      col_repr += f'[bold]{_size_and_bytes_repr(*size_bytes)}[/bold]'
      collections_size_repr.append(col_repr)

    no_show_methods = {'__call__', '<lambda>'}
    path_repr = '/'.join(row.path)
    method_repr = f' [dim]({row.method})[/dim]' if row.method not in no_show_methods else ''
    rich_table.add_row(
        path_repr,
        row.module_type.__name__ + method_repr,
        _as_yaml_str(_summary_tree_map(_maybe_render, _normalize_structure(row.inputs))),
        _as_yaml_str(_summary_tree_map(_maybe_render, _normalize_structure(row.outputs))),
        *collections_size_repr)

  # add footer with totals
  rich_table.columns[non_params_cols - 1].footer = rich.text.Text.from_markup(
      'Total', justify='right')

  # get collection totals
  collection_total = {col: (0, 0) for col in table.collections}
  for row in table:
    for col, size_bytes in row.size_and_bytes(table.collections).items():
      collection_total[col] = (
          collection_total[col][0] + size_bytes[0],
          collection_total[col][1] + size_bytes[1],
      )

  # add totals to footer
  for i, col in enumerate(table.collections):
    rich_table.columns[non_params_cols + i].footer = \
      _size_and_bytes_repr(*collection_total[col])

  # add final totals to caption
  caption_totals = (0, 0)
  for (size, num_bytes) in collection_total.values():
    caption_totals = (
        caption_totals[0] + size,
        caption_totals[1] + num_bytes,
    )

  rich_table.caption_style = 'bold'
  rich_table.caption = f'\nTotal Parameters: {_size_and_bytes_repr(*caption_totals)}'

  return '\n' + _get_rich_repr(rich_table, console_kwargs) + '\n'

def _summary_tree_map(f, tree, *rest):
  return jax.tree_util.tree_map(f, tree, *rest, is_leaf=lambda x: x is None)

def _size_and_bytes_repr(size: int, num_bytes: int) -> str:
  if not size:
    return ''
  bytes_repr = _bytes_repr(num_bytes)
  return f'{size:,} [dim]({bytes_repr})[/dim]'


def _size_and_bytes(pytree: Any) -> Tuple[int, int]:
  leaves = jax.tree_util.tree_leaves(pytree)
  size = sum(x.size for x in leaves if hasattr(x, 'size'))
  num_bytes = sum(x.size * x.dtype.itemsize for x in leaves if hasattr(x, 'size'))
  return size, num_bytes


def _get_rich_repr(obj, console_kwargs):
  f = io.StringIO()
  console = rich.console.Console(file=f, **console_kwargs)
  console.print(obj)
  return f.getvalue()


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
  return file.getvalue().replace('\n...', '').replace('\'', '').strip()


def _normalize_structure(obj):
  if isinstance(obj, _ValueRepresentation):
    return obj
  if isinstance(obj, (tuple, list)):
    return tuple(map(_normalize_structure, obj))
  elif isinstance(obj, Mapping):
    return {k: _normalize_structure(v) for k, v in obj.items()}
  elif dataclasses.is_dataclass(obj):
    return {f.name: _normalize_structure(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
  else:
    return obj

def _bytes_repr(num_bytes):
  count, units = ((f'{num_bytes / 1e9 :,.1f}', 'GB') if num_bytes > 1e9 else
                  (f'{num_bytes / 1e6 :,.1f}', 'MB') if num_bytes > 1e6 else
                  (f'{num_bytes / 1e3 :,.1f}', 'KB') if num_bytes > 1e3 else
                  (f'{num_bytes:,}', 'B'))

  return f'{count} {units}'


def _get_value_representation(x: Any) -> _ValueRepresentation:
  if isinstance(x, (int, float, bool, type(None))) or (
    isinstance(x, np.ndarray) and np.isscalar(x)):
    return _ObjectRepresentation(x)
  elif isinstance(x, meta.Partitioned):
    return _PartitionedArrayRepresentation.from_partitioned(x)
  try:
    return _ArrayRepresentation.from_array(x)
  except:
    return _ObjectRepresentation(x)

def _represent_tree(x):
  """Returns a tree with the same structure as `x` but with each leaf replaced
  by a `_ValueRepresentation` object."""
  return jax.tree_util.tree_map(
    _get_value_representation, x,
    is_leaf=lambda x: x is None or isinstance(x, meta.Partitioned))

def _maybe_render(x):
  return x.render() if hasattr(x, 'render') else repr(x)