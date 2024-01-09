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

from __future__ import annotations

import dataclasses
import typing as tp
from abc import ABCMeta
from copy import deepcopy
from functools import partial

import jax
import jax.tree_util as jtu
import numpy as np
import typing_extensions as tpe

from flax.experimental.nnx.nnx import (
  errors,
  filterlib,
  graph_utils,
  ids,
  reprlib,
  tracers,
)
from flax.experimental.nnx.nnx import variables as variableslib
from flax.experimental.nnx.nnx.graph_utils import GraphDef
from flax.experimental.nnx.nnx.proxy_caller import (
  ApplyCaller,
  CallableProxy,
  DelayedAccessor,
)
from flax.experimental.nnx.nnx.rnglib import Rngs
from flax.experimental.nnx.nnx.state import State
from flax.experimental.nnx.nnx.variables import Variable

A = tp.TypeVar('A')
B = tp.TypeVar('B')
M = tp.TypeVar('M', bound='Module')
S = tp.TypeVar('S', bound=tp.Union[State, tuple[State, ...]])
V = tp.TypeVar('V', bound=variableslib.Variable[tp.Any])

Path = str
PathParts = tuple[str, ...]

StateMapping = tp.Mapping[Path, tp.Any]


@tp.runtime_checkable
class _HasSetup(tp.Protocol):
  def setup(self) -> None:
    ...


SEEN_MODULES_REPR: tp.Optional[tp.Set[ids.UUID]] = None


class ModuleVariablesMapping(
  tp.MutableMapping[str, Variable[tp.Any]], reprlib.Representable
):
  __slots__ = ('_module',)

  def __init__(self, module: Module):
    if tp.TYPE_CHECKING:
      self._module = module
    else:
      object.__setattr__(self, '_module', module)

  def __getitem__(self, key: str | int) -> Variable[tp.Any]:
    if isinstance(key, int):
      key = str(key)

    module_vars = vars(self._module)
    if key not in module_vars:
      raise KeyError(f'Variable {key} not found')
    value = module_vars[key]

    if not isinstance(value, Variable):
      raise KeyError(f"Variable '{key}' is not found.")

    return value

  def __setitem__(self, name: str, value: Variable[tp.Any]) -> None:
    vars(self._module)[name] = value

  def __getattr__(self, name: str) -> Variable[tp.Any]:
    module_vars = vars(self._module)
    if name not in module_vars:
      raise AttributeError(f'Variable {name!r} not found')
    value = module_vars[name]
    if not isinstance(value, Variable):
      raise AttributeError(f"Variable '{name}' is not found.")
    return value

  def __setattr__(self, name: str, value: Variable[tp.Any]) -> None:
    vars(self._module)[name] = value

  def __delitem__(self, name: str) -> None:
    delattr(self._module, name)

  def __iter__(self) -> tp.Iterator[str]:
    for name, value in vars(self._module).items():
      if isinstance(value, Variable):
        yield name

  def __len__(self) -> int:
    return sum(1 for _ in self)

  def __nnx_repr__(self):
    yield reprlib.Object(type(self), start='{', end='}', value_sep=': ')
    for name, value in vars(self._module).items():
      if isinstance(value, Variable):
        yield reprlib.Attr(repr(name), value)


class ModuleState(reprlib.Representable):
  __slots__ = ('_trace_state', '_id')

  def __init__(self):
    self._trace_state = tracers.TraceState()
    self._id = ids.uuid()

  @property
  def trace_state(self) -> tracers.TraceState:
    return self._trace_state

  @property
  def id(self) -> ids.UUID:
    return self._id

  def __nnx_repr__(self):
    yield reprlib.Object(type(self))
    yield reprlib.Attr('trace_state', self._trace_state)


class ModuleMeta(ABCMeta):
  if not tp.TYPE_CHECKING:

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
      return self._meta_call(*args, **kwargs)

  def _meta_call(cls: tp.Type[M], *args, **kwargs) -> M:
    module = cls.__new__(cls, *args, **kwargs)
    vars(module)['_module__state'] = ModuleState()
    module.__init__(*args, **kwargs)

    if dataclasses.is_dataclass(module):
      if isinstance(module, _HasSetup):
        module.setup()

      assert isinstance(module, Module)

      for field in dataclasses.fields(module):
        if not field.init:
          continue
        value = vars(module)[field.name]
        # set Rngs instances to None
        if isinstance(value, Rngs):
          vars(module)[field.name] = None
          continue

        if 'nnx_variable_constructor' not in field.metadata:
          continue

        variable_constructor = field.metadata['nnx_variable_constructor']
        vars(module)[field.name] = variable_constructor(value)

    return module


tuple_reduce = lambda xs, x: xs + (x,)
tuple_init = lambda: ()

Updates = tp.Union[
  M,
  GraphDef[M],
  tuple[State, GraphDef[M]],
  tuple[tuple[State, ...], GraphDef[M]],
  State,
  tuple[State, ...],
]


class Module(reprlib.Representable, metaclass=ModuleMeta):
  if tp.TYPE_CHECKING:
    _module__state: ModuleState

  if not tp.TYPE_CHECKING:

    def __getattribute__(self, name: str) -> Any:
      value = object.__getattribute__(self, name)
      if isinstance(value, Variable):
        return value.get_value()
      return value

    def __setattr__(self, name: str, value: Any) -> None:
      self._setattr(name, value)

  def _setattr(self, name: str, value: tp.Any) -> None:
    if not self._module__state.trace_state.is_valid():
      raise errors.TraceContextError(
        'Cannot mutate Module from different trace level'
      )

    vars_dict = vars(self)
    if name in vars_dict:
      if isinstance(variable := vars_dict[name], Variable):
        if isinstance(value, Variable):
          if type(value) != type(variable):
            raise ValueError(
              f"Trying to assign a Variable of type '{type(value).__name__}' "
              f"to the Module attribute '{name}' of a different type "
              f"'{type(variable).__name__}'."
            )
          variable.copy_from(value)
        else:
          variable.set_value(value)
      else:
        vars_dict[name] = value
    else:
      if isinstance(value, (jax.Array, np.ndarray, State)):
        raise ValueError(
          f"Trying to assign a '{type(value).__name__}' to the Module"
          f" attribute '{name}'. This is not supported. Non-hashable "
          'objects are not valid static state in JAX. Please wrap '
          'the value in a Variable type instead.'
        )
      vars_dict[name] = value

  @property
  def variables(self) -> ModuleVariablesMapping:
    return ModuleVariablesMapping(self)

  def __deepcopy__(self: M, memo=None) -> M:
    state, graphdef = self.split()
    graphdef = deepcopy(graphdef)
    state = deepcopy(state)
    return graphdef.merge(state)

  def __hash__(self) -> int:
    return hash(self._module__state.id)

  def __nnx_repr__(self):
    global SEEN_MODULES_REPR

    if SEEN_MODULES_REPR is None:
      SEEN_MODULES_REPR = set()
      clear_seen = True
    else:
      clear_seen = False

    if self._module__state.id in SEEN_MODULES_REPR:
      yield reprlib.Object(type=type(self), empty_repr='...')
      return

    yield reprlib.Object(type=type(self))
    SEEN_MODULES_REPR.add(self._module__state.id)

    try:
      for name, value in vars(self).items():
        if isinstance(value, Module) or (
          not isinstance(value, Variable) and not name.startswith('_')
        ):
          yield reprlib.Attr(name, value)
    finally:
      if clear_seen:
        SEEN_MODULES_REPR = None

  @classmethod
  def init(cls: type[M], *args, **kwargs) -> tuple[State, GraphDef[M]]:
    return cls(*args, **kwargs).split()

  @classmethod
  @property
  def create_abstract(cls: type[M]) -> type[M]:
    def lift_rngs(kwargs: dict[str, tp.Any]):
      if 'rngs' in kwargs and isinstance(kwargs['rngs'], Rngs):
        kwargs['rngs'] = kwargs['rngs'].copy()
      return kwargs

    def _create_abstract(accessor: DelayedAccessor, *args, **kwargs):
      constructor = accessor(cls)
      state, graphdef = jax.eval_shape(
        lambda: constructor(*args, **lift_rngs(kwargs)).split()
      )
      return graphdef.merge(state)

    return CallableProxy(_create_abstract)  # type: ignore

  def clone(self: M) -> M:
    return merge(self.split())

  @tp.overload
  def split(self: M) -> tuple[State, GraphDef[M]]:
    ...

  @tp.overload
  def split(self: M, first: filterlib.Filter, /) -> tuple[State, GraphDef[M]]:
    ...

  @tp.overload
  def split(
    self: M,
    first: filterlib.Filter,
    second: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
  ) -> tuple[State, tpe.Unpack[tuple[State, ...]], GraphDef[M]]:
    ...

  def split(
    self: M, *filters: filterlib.Filter
  ) -> tuple[State, tpe.Unpack[tuple[State, ...]], GraphDef[M]]:
    state, graphdef = graph_utils.graph_flatten(self)

    if len(filters) == 0:
      states = (state,)
    elif len(filters) == 1:
      states = (state.split(filters[0]),)
    else:
      states = state.split(filters[0], filters[1], *filters[2:])

    return *states, graphdef

  def get_state(self) -> State:
    state, _ = self.split()
    return state

  def get_graphdef(self: M) -> GraphDef[M]:
    _, graphdef = self.split()
    return graphdef

  @tp.overload
  def extract(self, first: filterlib.Filter, /) -> State:
    ...

  @tp.overload
  def extract(
    self,
    first: filterlib.Filter,
    second: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
  ) -> tuple[State, ...]:
    ...

  def extract(
    self,
    first: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
  ) -> tp.Union[State, tuple[State, ...]]:
    state = self.get_state()

    if len(filters) == 0:
      states = state.extract(first)
    else:
      states = state.extract(first, filters[0], *filters[1:])

    return states

  @tp.overload
  def pop(
    self,
    filter: filterlib.Filter,
    /,
  ) -> State:
    ...

  @tp.overload
  def pop(
    self,
    filter: filterlib.Filter,
    filter2: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
  ) -> tuple[State, ...]:
    ...

  def pop(
    self, *filters: filterlib.Filter
  ) -> tp.Union[State, tuple[State, ...]]:
    if len(filters) == 0:
      raise ValueError('Expected at least one filter')

    states = graph_utils.graph_pop(self, filters)

    if len(states) == 1:
      return states[0]
    else:
      return states

  @property
  def apply(self: M) -> ApplyCaller[M]:
    def _apply(accessor: DelayedAccessor, *args, **kwargs) -> tuple[tp.Any, M]:
      module = self.clone()
      fn = accessor(module)
      out = fn(*args, **kwargs)
      return out, module

    return CallableProxy(_apply)  # type: ignore

  def update(self: M, update: Updates[M], *updates: Updates[M]) -> None:
    updates = (update, *updates)

    def _states_and_moduledef(
      updates,
    ) -> tuple[list[State], tp.Optional[Module]]:
      leaves = jax.tree_util.tree_leaves(
        updates, is_leaf=lambda x: isinstance(x, (GraphDef, State))
      )
      states: list[State] = []
      module: tp.Optional[Module] = None

      for leaf in leaves:
        if isinstance(leaf, (Module, GraphDef)):
          if module is not None:
            raise ValueError(
              'Expected only one GraphDef or Module in the updates'
            )

          if isinstance(leaf, Module):
            module = leaf
            states.append(leaf.get_state())
          else:
            module = leaf.make_empty()
            module
        elif isinstance(leaf, State):
          states.append(leaf)
        else:
          raise ValueError(
            'Expected a GraphDef, Module or State, got'
            f' {type(leaf).__name__}'
          )

      return states, module

    states, module_update = _states_and_moduledef(updates)

    if module_update is not None:
      graph_utils.graph_update_static(self, module_update)

    if states:
      graph_utils.graph_update_dynamic(self, states)

  def sow(
    self,
    variable_type: tp.Type[variableslib.Variable[tp.Any]],
    name: str,
    value: A,
    reduce_fn: tp.Callable[[B, A], B] = tuple_reduce,
    init_fn: tp.Callable[[], B] = tuple_init,  # type: ignore
  ) -> None:
    if hasattr(self, name):
      variable = vars(self)[name]
      if not isinstance(variable, variableslib.Variable):
        raise ValueError(
          f"Expected '{name}' to be a Variable, got {type(variable).__name__}"
        )
      elif type(variable) != variable_type:
        raise ValueError(
          f"Expected '{name}' to be of type '{variable_type.__name__}', "
          f"got '{type(variable).__name__}'"
        )
      reduced_value = reduce_fn(variable.value, value)
      setattr(self, name, reduced_value)
    else:
      reduced_value = reduce_fn(init_fn(), value)
      setattr(self, name, variable_type(reduced_value))

  def modules(self) -> tp.Iterator[tuple[Path, Module]]:
    for path, value in graph_utils.iter_nodes(self):
      if isinstance(value, Module):
        yield path, value

  def __init_subclass__(cls, experimental_pytree: bool = False) -> None:
    super().__init_subclass__()

    graph_utils.register_node_type(
      cls,
      _module_graph_flatten,
      _module_graph_get_key,
      _module_graph_set_key,
      _module_graph_has_key,
      _module_graph_all_keys,
      create_empty=_module_graph_create_empty,
      init=_module_graph_init,
    )

    if experimental_pytree:
      jtu.register_pytree_with_keys(
        cls,
        partial(_module_flatten, with_keys=True),
        _module_unflatten,
        flatten_func=partial(_module_flatten, with_keys=False),
      )


# -------------------------
# Pytree Definition
# -------------------------
def _module_flatten(module: Module, *, with_keys: bool):
  state, graphdef = module.split()
  variables = state.raw_mapping
  paths = tuple(variables.keys())

  if with_keys:
    children = tuple(
      (jtu.DictKey(path), variable) for path, variable in variables.items()
    )
  else:
    children = tuple(variables.values())

  return children, (paths, graphdef)


def _module_unflatten(
  paths_moduledef: tuple[tuple[Path, ...], GraphDef[M]],
  variables: tuple[Variable[tp.Any], ...],
) -> M:
  paths, graphdef = paths_moduledef
  return graphdef.merge(State(zip(paths, variables)))


# -------------------------
# Graph Definition
# -------------------------
def _module_graph_flatten(module: Module):
  nodes = tuple(
    (name, value)
    for name, value in vars(module).items()
    if name != '_module__state'
  )
  return nodes, type(module)


def _module_graph_get_key(module: Module, name: str) -> tp.Any:
  return vars(module)[name]


def _module_graph_set_key(module: M, name: str, value: tp.Any) -> M:
  setattr(module, name, value)
  return module


def _module_graph_has_key(module: Module, name: str) -> bool:
  return hasattr(module, name)


def _module_graph_all_keys(module: Module) -> tuple[str, ...]:
  return tuple(name for name in vars(module).keys() if name != '_module__state')


def _module_graph_create_empty(cls: tp.Type[M]) -> M:
  module = object.__new__(cls)
  vars(module).update(_module__state=ModuleState())
  return module


def _module_graph_init(node: Module, items: tuple[tuple[str, tp.Any], ...]):
  vars(node).update(items)


def first_from(*args: tp.Optional[A], error_msg: str) -> A:
  """Return the first non-None argument.

  If all arguments are None, raise a ValueError with the given error message.

  Args:
    *args: the arguments to check
    error_msg: the error message to raise if all arguments are None
  Returns:
    The first non-None argument.
  """
  for arg in args:
    if arg is not None:
      return arg
  raise ValueError(error_msg)


def merge(
  state_and_def: tuple[tpe.Unpack[tuple[State, ...]], GraphDef[M]],
) -> M:
  *states, graphdef = state_and_def
  return graphdef.merge(*states)
