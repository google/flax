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
import dataclasses
import io
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import flax
from flax.core.scope import CollectionFilter, DenyList
import jax
import rich.console
import rich.table
import rich.text
import yaml

PRNGKey = Any  # pylint: disable=invalid-name
RNGSequences = Dict[str, PRNGKey]
Array = Any    # pylint: disable=invalid-name


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
  outputs: Any
  module_variables: Dict[str, Dict[str, Array]]
  counted_variables: Dict[str, Dict[str, Array]]

  def size_and_bytes(self,
                     collections: Iterable[str]) -> Dict[str, Tuple[int, int]]:
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

  def __init__(self, module: 'flax.linen.Module', collections: List[str],
               rows: Iterable[Row]):
    super().__init__(rows)
    self.module = module
    self.collections = collections


def tabulate(
    module: 'flax.linen.Module',
    rngs: Union[PRNGKey, RNGSequences],
    method: Optional[Callable[..., Any]] = None,
    mutable: CollectionFilter = True,
    depth: Optional[int] = None,
    exclude_methods: Sequence[str] = (),
) -> Callable[..., str]:
  """Returns a function that creates a summary of the Module represented as a table.

  This function accepts most of the same arguments as `Module.init`, except that
  it returns a function of the form `(*args, **kwargs) -> str` where `*args` and
  `**kwargs`
  are passed to `method` (e.g. `__call__`) during the forward pass.

  `tabulate` uses `jax.eval_shape` under the hood to run the forward computation
  without
  consuming any FLOPs or allocating memory.

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
      ┏━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
      ┃ path    ┃ outputs       ┃ params               ┃
      ┡━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
      │ Inputs  │ float32[16,9] │                      │
      ├─────────┼───────────────┼──────────────────────┤
      │ Dense_0 │ float32[16,4] │ bias: float32[4]     │
      │         │               │ kernel: float32[9,4] │
      │         │               │                      │
      │         │               │ 40 (160 B)           │
      ├─────────┼───────────────┼──────────────────────┤
      │ Dense_1 │ float32[16,2] │ bias: float32[2]     │
      │         │               │ kernel: float32[4,2] │
      │         │               │                      │
      │         │               │ 10 (40 B)            │
      ├─────────┼───────────────┼──────────────────────┤
      │ Foo     │ float32[16,2] │                      │
      ├─────────┼───────────────┼──────────────────────┤
      │         │         Total │ 50 (200 B)           │
      └─────────┴───────────────┴──────────────────────┘

                  Total Parameters: 50 (200 B)

  **Note**: rows order in the table does not represent execution order,
  instead it aligns with the order of keys in `variables` which are sorted
  alphabetically.

  Args:
    module: The module to tabulate.
    method: An optional method. If provided, applies this method. If not
      provided, applies the ``__call__`` method.
    mutable: Can be bool, str, or list. Specifies which collections should be
      treated as mutable: ``bool``: all/no collections are mutable. ``str``: The
      name of a single mutable collection. ``list``: A list of names of mutable
      collections. By default all collections except 'intermediates' are
      mutable.
    depth: controls how many submodule deep the summary can go. By default its
      `None` which means no limit. If a submodule is not shown because of the
      depth limit, its parameter count and bytes will be added to the row of its
      first shown ancestor such that the sum of all rows always adds up to the
      total number of parameters of the Module.
    exclude_methods: A sequence of strings that specifies which methods should
      be ignored. In case a module calls a helper method from its main method,
      use this argument to exclude the helper method from the summary to avoid
      ambiguity.

  Returns:
    A function that accepts the same `*args` and `**kwargs` of the forward pass
    (`method`) and returns a string with a tabular representation of the
    Modules.
  """

  def _tabulate_fn(*args, **kwargs):
    table_fn = _get_module_table(module, rngs, method=method, 
                                 mutable=mutable, depth=depth, 
                                 exclude_methods=set(exclude_methods))
    table = table_fn(*args, **kwargs)
    return _render_table(table)

  return _tabulate_fn


def _get_module_table(
    module: 'flax.linen.Module',
    rngs: Union[PRNGKey, RNGSequences],
    method: Optional[Callable[..., Any]],
    mutable: CollectionFilter,
    depth: Optional[int],
    exclude_methods: Set[str],
) -> Callable[..., Table]:

  exclude_methods.add("setup")

  def _get_table_fn(*args, **kwargs):
    output_methods: Set[str] = set()

    def capture_intermediates(_module, method_name: str):
      if method_name in exclude_methods:
        return False
      else:
        output_methods.add(method_name)
        return True

    shape_variables = jax.eval_shape(lambda: module.init(
        rngs,
        *args,
        method=method,
        mutable=mutable,
        capture_intermediates=capture_intermediates,
        **kwargs,
    ))

    collections: List[str] = [
        col for col in shape_variables.keys() if col != 'intermediates'
    ]
    shape_variables = shape_variables.unfreeze()
    rows = list(
        _flatten_to_rows(
            path=(),
            variables=shape_variables,
            depth=depth,
            output_methods=output_methods))

    if args and kwargs:
      input_values = (*args, kwargs)
    elif args and not kwargs:
      input_values = args[0] if len(args) == 1 else args
    elif kwargs and not args:
      input_values = kwargs
    else:
      input_values = ''

    inputs_row = Row(('Inputs',), input_values, {}, {})
    rows.insert(0, inputs_row)

    return Table(module, collections, rows)

  return _get_table_fn


def _flatten_to_rows(
    path: Tuple[str, ...],
    variables: Dict[str, Any],
    depth: Optional[int],
    output_methods: Set[str],
) -> Iterable[Row]:

  # get variables only for this Module
  module_variables = _get_module_variables(variables)
  module_outputs = {
      key: value
      for key, value in variables['intermediates'].items()
      if key in output_methods
  }

  if len(module_outputs) == 0:
    output = None
  elif len(module_outputs) > 1:
    raise ValueError(
        f"Cannot infer output, module '{'/'.join(path)}' has multiple "
        f"intermediates: {list(module_outputs.keys())}. Use the `exclude_methods` "
        f"argument to make sure each module only reports one output.")
  else:
    output = list(module_outputs.values())[0][0]

  if depth is not None and depth == 0:
    # don't recurse, yield current level
    # count_variables contains all variables that are not intermediates
    variables = variables.copy()
    del variables['intermediates']
    module_variables.pop('intermediates')
    yield Row(
        path=path,
        outputs=output,
        module_variables=module_variables,
        counted_variables=variables,
    )
  else:
    # recurse into lower levels
    keys = list(key for key in variables['intermediates'].keys()
                if key not in module_variables['intermediates'])

    # add keys from other collections
    # dont use set here because we want to preserve order
    for collection in variables:
      if collection != 'intermediates':
        for key in variables[collection]:
          if key not in keys and key not in module_variables.get(
              collection, {}):
            keys.append(key)

    for key in keys:
      next_path = path + (key,)
      next_variables = _step_into(variables, key)
      yield from _flatten_to_rows(
          path=next_path,
          variables=next_variables,
          depth=depth - 1 if depth is not None else None,
          output_methods=output_methods,
      )

    # current row
    yield Row(
        path=path,
        outputs=output,
        module_variables=module_variables,
        counted_variables=module_variables,
    )


def _step_into(variables: Dict[str, Any], key: str):
  return {
      col: params[key] for col, params in variables.items() if key in params
  }


def _get_module_variables(variables: Dict[str, Any]) -> Dict[str, Any]:

  module_variables: Dict[str, Dict[str, Any]] = {
      collection: {
          name: value
          for name, value in params.items()
          if not isinstance(value, Mapping)  # is this robust?
      } for collection, params in variables.items()
  }
  # filter empty collectionswhen
  module_variables = {
      collection: params
      for collection, params in module_variables.items()
      if len(params) > 0
  }

  return module_variables


def _render_table(table: Table) -> str:
  rich_table = rich.table.Table(
      show_header=True,
      show_lines=True,
      show_footer=True,
      title=f'{table.module.__class__.__name__} Summary',
  )

  rich_table.add_column('path')
  rich_table.add_column('outputs')

  for col in table.collections:
    rich_table.add_column(col)

  for row in table:
    collections_size_repr = []

    for collection, size_bytes in row.size_and_bytes(table.collections).items():
      col_repr = ''

      if collection in row.module_variables:
        col_repr += _as_yaml_str(
            jax.tree_map(_format_value, row.module_variables[collection]))
        col_repr += '\n\n'

      col_repr += f'[bold]{_size_and_bytes_repr(*size_bytes)}[/bold]'
      collections_size_repr.append(col_repr)

    rich_table.add_row(
        '/'.join(row.path) if row.path else table.module.__class__.__name__,
        _as_yaml_str(jax.tree_map(_format_value, row.outputs)),
        *collections_size_repr)

  # add footer with totals
  rich_table.columns[1].footer = rich.text.Text.from_markup(
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
    rich_table.columns[2 +
                       i].footer = _size_and_bytes_repr(*collection_total[col])

  # add final totals to caption
  caption_totals = (0, 0)
  for (size, num_bytes) in collection_total.values():
    caption_totals = (
        caption_totals[0] + size,
        caption_totals[1] + num_bytes,
    )

  rich_table.caption_style = 'bold'
  rich_table.caption = f'\nTotal Parameters: {_size_and_bytes_repr(*caption_totals)}'

  return '\n' + _get_rich_repr(rich_table) + '\n'


def _size_and_bytes_repr(size: int, num_bytes: int) -> str:
  if not size:
    return ''
  bytes_repr = _bytes_repr(num_bytes)
  return f'{size:,} [dim]({bytes_repr})[/dim]'


def _size_and_bytes(pytree: Any) -> Tuple[int, int]:
  leaves = jax.tree_leaves(pytree)
  size = sum(x.size for x in leaves)
  num_bytes = sum(x.size * x.dtype.itemsize for x in leaves)
  return size, num_bytes


def _get_rich_repr(obj):
  f = io.StringIO()
  console = rich.console.Console(file=f, force_terminal=True)
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


def _format_value(value):
  if hasattr(value, 'shape') and hasattr(value, 'dtype'):
    shape_repr = ','.join(map(str, value.shape))
    return f'[dim]{value.dtype}[/dim][{shape_repr}]'
  else:
    return str(value)


def _bytes_repr(num_bytes):
  count, units = ((f'{num_bytes / 1e9 :,.1f}', 'GB') if num_bytes > 1e9 else
                  (f'{num_bytes / 1e6 :,.1f}', 'MB') if num_bytes > 1e6 else
                  (f'{num_bytes / 1e3 :,.1f}', 'KB') if num_bytes > 1e3 else
                  (f'{num_bytes:,}', 'B'))

  return f'{count} {units}'
