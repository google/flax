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
import enum
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
  ids,
  reprlib,
  tracers,
)
from flax.experimental.nnx.nnx import variables as variableslib
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
StateDict = tp.Dict[Path, tp.Any]
StateMapping = tp.Mapping[Path, tp.Any]


class _ProxyContext(tp.Protocol):
  def __call__(self, accessor: 'DelayedAccessor', /, *args, **kwargs) -> tp.Any:
    ...


@tp.runtime_checkable
class _HasSetup(tp.Protocol):
  def setup(self) -> None:
    ...


@dataclasses.dataclass
class CallableProxy:
  _proxy_context: _ProxyContext
  _proxy_callable: tp.Callable[..., tp.Any]

  def __call__(self, *args, **kwargs):
    return self._proxy_context(self._proxy_callable, *args, **kwargs)

  def __getattr__(self, name) -> 'CallableProxy':
    return CallableProxy(
      self._proxy_context, getattr(self._proxy_callable, name)
    )

  def __getitem__(self, key) -> 'CallableProxy':
    return CallableProxy(self._proxy_context, self._proxy_callable[key])


def _identity(x):
  return x


@dataclasses.dataclass
class DelayedAccessor:
  accessor: tp.Callable[[tp.Any], tp.Any] = _identity

  def __call__(self, x):
    return self.accessor(x)

  def __getattr__(self, name):
    return DelayedAccessor(lambda x: getattr(x, name))

  def __getitem__(self, key):
    return DelayedAccessor(lambda x: x[key])


class ApplyCaller(tp.Protocol, tp.Generic[A]):
  def __getattr__(self, __name) -> 'ApplyCaller[A]':
    ...

  def __getitem__(self, __name) -> 'ApplyCaller[A]':
    ...

  def __call__(self, *args, **kwargs) -> tuple[tp.Any, A]:
    ...


@dataclasses.dataclass(repr=False)
class _SubmodulesRepr(reprlib.Representable):
  submodules: tuple[tuple[str, tp.Union['ModuleDef[Module]', int]], ...]

  def __nnx_repr__(self):
    yield reprlib.Object(type='', value_sep=', ')

    for name, submodule in self.submodules:
      yield reprlib.Attr(repr(name), submodule, start='(', end=')')


class ModuleDef(tp.Generic[M], reprlib.Representable):
  __slots__ = (
    '_type',
    '_index',
    '_submodules',
    '_static_fields',
    '_variables',
    '_module_state',
  )

  def __init__(
    self,
    type: tp.Type[M],
    index: int,
    submodules: tuple[tuple[str, tp.Union['ModuleDef[Module]', int]], ...],
    static_fields: tuple[tuple[str, tp.Any], ...],
    variables: tuple[
      tuple[str, variableslib.Variable[variableslib.Empty]], ...
    ],
    module_state: 'ModuleStateTuple',
  ):
    self._type = type
    self._index = index
    self._submodules = submodules
    self._static_fields = static_fields
    self._variables = variables
    self._module_state = module_state

  def __nnx_repr__(self):
    yield reprlib.Object(type=type(self))

    yield reprlib.Attr('type', self._type.__name__)
    yield reprlib.Attr('index', self._index)
    yield reprlib.Attr('static_fields', self._static_fields)
    yield reprlib.Attr('variables', self._variables)
    yield reprlib.Attr('submodules', _SubmodulesRepr(self._submodules))

  def __hash__(self) -> int:
    return hash(
      (self._type, self._submodules, self._static_fields, self._variables)
    )

  def __eq__(self, other: tp.Any) -> bool:
    if not isinstance(other, ModuleDef):
      return False
    return (
      self._type == other._type
      and self._submodules == other._submodules
      and self._static_fields == other._static_fields
    )

  @property
  def type(self) -> tp.Type[M]:
    return self._type

  @property
  def index(self) -> int:
    return self._index

  @property
  def submodules(
    self,
  ) -> tuple[tuple[str, tp.Union['ModuleDef[Module]', int]], ...]:
    return self._submodules

  @property
  def static_fields(self) -> tuple[tuple[str, tp.Any], ...]:
    return self._static_fields

  @property
  def variables(
    self,
  ) -> tuple[tuple[str, variableslib.Variable[variableslib.Empty]], ...]:
    return self._variables

  @property
  def module_state(self) -> 'ModuleStateTuple':
    return self._module_state

  def make_module(self) -> M:
    return _build_module(self)

  def merge(self, state: State, *states: State) -> M:
    states = (state, *states)
    module = self.make_module()
    _update_module_dynamic_state(module, states)
    return module

  def apply(
    self, state: State, *states: State
  ) -> ApplyCaller[tuple[State, 'ModuleDef[M]']]:
    accessesor = DelayedAccessor()

    def _apply(
      accessesor, *args, **kwargs
    ) -> tuple[tp.Any, tuple[State, ModuleDef[M]]]:
      module = self.merge(state, *states)
      fn = accessesor(module)
      out = fn(*args, **kwargs)
      return out, module.split()

    return CallableProxy(_apply, accessesor)  # type: ignore


def _moddef_flatten(moduledef: ModuleDef[M]):
  return (), (
    moduledef._type,
    moduledef._index,
    moduledef._submodules,
    moduledef._static_fields,
    moduledef._variables,
    moduledef._module_state,
  )


def _moddef_unflatten(
  metadata: tuple[
    tp.Type[M],
    int,
    tuple[tuple[str, tp.Union['ModuleDef[Module]', int]], ...],
    tuple[tuple[str, tp.Any], ...],
    tuple[tuple[str, variableslib.Variable[variableslib.Empty]], ...],
    'ModuleStateTuple',
  ],
  _,
) -> ModuleDef[M]:
  return ModuleDef(*metadata)


jtu.register_pytree_node(ModuleDef, _moddef_flatten, _moddef_unflatten)


SEEN_MODULES_REPR: tp.Optional[tp.Set[ids.UUID]] = None

ModuleStateTuple = tuple[()]


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

  def to_tuple(self) -> ModuleStateTuple:
    return ()

  @classmethod
  def from_tuple(cls, tup: ModuleStateTuple) -> 'ModuleState':
    return cls(*tup)

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
  ModuleDef[M],
  tuple[State, ModuleDef[M]],
  tuple[tuple[State, ...], ModuleDef[M]],
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

  def _setattr(self, name: str, value: Any) -> None:
    if not self._module__state.trace_state.is_valid():
      raise errors.TraceContextError(
        'Cannot mutate Module from different trace level'
      )

    vars_dict = vars(self)
    if name in vars_dict and isinstance(vars_dict[name], Variable):
      vars_dict[name] = vars_dict[name].set_value(value)
    else:
      if isinstance(value, Variable):
        value = value.copy()
      elif isinstance(value, (jax.Array, np.ndarray, State)):
        raise ValueError(
          f"Trying to assing a '{type(value).__name__}' to the Module"
          f" attribute '{name}'. This is not supported. Non-hashable "
          'objects are not valid static state in JAX. Please wrap '
          'the value in a Variable type instead.'
        )
      vars_dict[name] = value

  def __deepcopy__(self: M, memo=None) -> M:
    state, moduledef = self.split()
    moduledef = deepcopy(moduledef)
    state = deepcopy(state)
    return moduledef.merge(state)

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
  def init(cls: type[M], *args, **kwargs) -> tuple[State, ModuleDef[M]]:
    return cls(*args, **kwargs).split()

  @classmethod
  @property
  def create_abstract(cls: type[M]) -> type[M]:
    accessesor = DelayedAccessor()

    def lift_rngs(kwargs: dict[str, tp.Any]):
      if 'rngs' in kwargs and isinstance(kwargs['rngs'], Rngs):
        kwargs['rngs'] = kwargs['rngs'].copy()
      return kwargs

    def _create_abstract(accessesor, *args, **kwargs):
      constructor = accessesor(cls)
      state, moduledef = jax.eval_shape(
        lambda: constructor(*args, **lift_rngs(kwargs)).split()
      )
      return moduledef.merge(state)

    return CallableProxy(_create_abstract, accessesor)  # type: ignore

  def clone(self: M) -> M:
    return merge(self.split())

  @tp.overload
  def split(self: M) -> tuple[State, ModuleDef[M]]:
    ...

  @tp.overload
  def split(self: M, first: filterlib.Filter, /) -> tuple[State, ModuleDef[M]]:
    ...

  @tp.overload
  def split(
    self: M,
    first: filterlib.Filter,
    second: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
  ) -> tuple[State, tpe.Unpack[tuple[State, ...]], ModuleDef[M]]:
    ...

  def split(
    self: M, *filters: filterlib.Filter
  ) -> tuple[State, tpe.Unpack[tuple[State, ...]], ModuleDef[M]]:
    moduledef = self.get_moduledef()
    state = self.get_state()

    if len(filters) == 0:
      states = (state,)
    elif len(filters) == 1:
      states = (state.split(filters[0]),)
    else:
      states = state.split(filters[0], filters[1], *filters[2:])

    return *states, moduledef

  def get_state(self) -> State:
    return State(_iter_state(self))

  def get_moduledef(self: M) -> ModuleDef[M]:
    module_index: tp.Dict[ids.UUID, int] = {}
    path: PathParts = ()
    moduledef = _make_moduledef_recursive(self, module_index, path)
    assert isinstance(moduledef, ModuleDef)
    return moduledef

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

    states = _pop(self, filters)

    if len(states) == 1:
      return states[0]
    else:
      return states

  @property
  def apply(self: M) -> ApplyCaller[M]:
    accessesor = DelayedAccessor()

    def _apply(accessesor, *args, **kwargs) -> tuple[tp.Any, M]:
      module = self.clone()
      fn = accessesor(module)
      out = fn(*args, **kwargs)
      return out, module

    return CallableProxy(_apply, accessesor)  # type: ignore

  def update(self: M, update: Updates[M], *updates: Updates[M]) -> None:
    updates = (update, *updates)

    def _states_and_moduledef(
      updates,
    ) -> tuple[list[State], tp.Optional[Module]]:
      leaves = jax.tree_util.tree_leaves(
        updates, is_leaf=lambda x: isinstance(x, (ModuleDef, State))
      )
      states: list[State] = []
      module: tp.Optional[Module] = None

      for leaf in leaves:
        if isinstance(leaf, (Module, ModuleDef)):
          if module is not None:
            raise ValueError(
              'Expected only one ModuleDef or Module in the updates'
            )

          if isinstance(leaf, Module):
            module = leaf
            states.append(leaf.get_state())
          else:
            module = leaf.make_module()
        elif isinstance(leaf, State):
          states.append(leaf)
        else:
          raise ValueError(
            'Expected a ModuleDef, Module or State, got'
            f' {type(leaf).__name__}'
          )

      return states, module

    states, module_update = _states_and_moduledef(updates)

    if module_update is not None:
      _update_module_static_state(self, module_update)

    if states:
      _update_module_dynamic_state(self, states)

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

  def for_each(
    self, module_type: tp.Type[M], fn: tp.Callable[[M], None]
  ) -> None:
    visited: tp.Set[ids.UUID] = set()
    self._on_all(module_type, fn, visited)

  def _on_all(
    self,
    module_type: tp.Type[M],
    fn: tp.Callable[[M], None],
    visited: tp.Set[ids.UUID],
  ) -> None:
    if self._module__state.id in visited:
      return

    visited.add(self._module__state.id)

    if isinstance(self, module_type):
      fn(self)

    for value in vars(self).values():
      if isinstance(value, Module):
        value._on_all(module_type, fn, visited)

  def __init_subclass__(cls, experimental_pytree: bool = False) -> None:
    super().__init_subclass__()

    if experimental_pytree:
      jtu.register_pytree_with_keys(
        cls,
        partial(_module_flatten, with_keys=True),
        _module_unflatten,
        flatten_func=partial(_module_flatten, with_keys=False),
      )


# Pytree Definition
def _module_flatten(module: Module, *, with_keys: bool):
  state, moduledef = module.split()
  variables = state.variables
  paths = tuple(variables.keys())

  if with_keys:
    children = tuple(
      (jtu.DictKey(path), variable) for path, variable in variables.items()
    )
  else:
    children = tuple(variables.values())

  return children, (paths, moduledef)


def _module_unflatten(
  paths_moduledef: tuple[tuple[Path, ...], ModuleDef[M]],
  variables: tuple[Variable[tp.Any], ...],
) -> M:
  paths, moduledef = paths_moduledef
  return moduledef.merge(State(zip(paths, variables)))


def _make_moduledef_recursive(
  module: M,
  module_index: tp.Dict[ids.UUID, int],
  path: PathParts,
) -> tp.Union[ModuleDef[M], int]:
  if module._module__state.id in module_index:
    return module_index[module._module__state.id]

  index = len(module_index)
  module_index[module._module__state.id] = index

  submodules = []
  static_fields = []
  variables = []

  for name, value in sorted(vars(module).items(), key=lambda x: x[0]):
    value_path = (*path, name)
    if isinstance(value, Module):
      submodule_def = _make_moduledef_recursive(value, module_index, value_path)
      submodules.append((name, submodule_def))
    elif isinstance(value, variableslib.Variable):
      variables.append((name, value.as_empty()))
    elif not name.startswith('_module__'):
      static_fields.append((name, value))

  module_def = ModuleDef(
    type=type(module),
    index=index,
    submodules=tuple(submodules),
    static_fields=tuple(static_fields),
    variables=tuple(variables),
    module_state=module._module__state.to_tuple(),
  )
  return module_def


def _iter_state(module: Module) -> tp.Iterator[tuple[Path, tp.Any]]:
  seen_modules: tp.Set[ids.UUID] = set()
  path_parts: PathParts = ()

  yield from _iter_state_recursive(module, seen_modules, path_parts)


def _iter_state_recursive(
  module: Module, seen_modules: tp.Set[ids.UUID], path_parts: PathParts
) -> tp.Iterator[tuple[Path, tp.Any]]:
  if module._module__state.id in seen_modules:
    return

  seen_modules.add(module._module__state.id)

  for name, value in sorted(vars(module).items(), key=lambda x: x[0]):
    new_path_parts = (*path_parts, name)
    if isinstance(value, Module):
      yield from _iter_state_recursive(value, seen_modules, new_path_parts)
    elif isinstance(value, variableslib.Variable):
      if value.is_empty:
        # skip empty Variables
        continue
      path = '/'.join(new_path_parts)
      yield path, value


def _set_value_at_path(
  module: tp.Any, path_parts: tp.Union[PathParts, tp.List[str]], value: tp.Any
):
  if len(path_parts) == 1:
    setattr(module, path_parts[0], value)
  else:
    _set_value_at_path(vars(module)[path_parts[0]], path_parts[1:], value)


def _get_value_path(module: tp.Any, path: tp.Sequence[str]) -> tp.Any:
  if len(path) == 0:
    return module
  else:
    return _get_value_path(vars(module)[path[0]], path[1:])


def _build_module(moduledef: ModuleDef[M]) -> M:
  index_module: tp.Dict[int, Module] = {}
  module = _build_module_recursive(moduledef, index_module)
  return module


def _build_module_recursive(
  moduledef: tp.Union[ModuleDef[M], int],
  index_module: tp.Dict[int, Module],
) -> M:
  if isinstance(moduledef, int):
    return index_module[moduledef]  # type: ignore

  assert moduledef.index not in index_module

  # add a dummy module to the index to avoid infinite recursion
  module = object.__new__(moduledef.type)
  index_module[moduledef.index] = module

  submodules = {
    name: _build_module_recursive(submodule, index_module)
    for name, submodule in moduledef.submodules
  }

  vars(module).update(moduledef.static_fields)
  vars(module).update(moduledef.variables)
  vars(module).update(submodules)
  vars(module)['_module__state'] = ModuleState.from_tuple(
    moduledef.module_state
  )

  return module


def _pop(
  module: Module,
  filters: tuple[filterlib.Filter, ...],
) -> tuple[State, ...]:
  module_index: tp.Dict[ids.UUID, int] = {}
  path_parts: PathParts = ()
  predicates = tuple(filterlib.to_predicate(filter) for filter in filters)
  states = tuple({} for _ in predicates)
  _pop_recursive(module, module_index, path_parts, states, predicates)

  return tuple(State(x) for x in states)


def _pop_recursive(
  module: Module,
  module_index: tp.Dict[ids.UUID, int],
  path_parts: PathParts,
  states: tuple[tp.Dict[Path, tp.Any]],
  predicates: tuple[filterlib.Predicate, ...],
) -> None:
  if module._module__state.id in module_index:
    return

  for name, value in list(vars(module).items()):
    if isinstance(value, Module):
      _pop_recursive(
        value, module_index, (*path_parts, name), states, predicates
      )
      continue
    elif not isinstance(value, Variable):
      continue
    elif value.is_empty:
      continue

    path = '/'.join((*path_parts, name))
    for state, predicate in zip(states, predicates):
      if predicate(path, value):
        state[path] = value
        # empty Variable attributes
        setattr(module, name, value.as_empty())
        break
    else:
      # NOTE: should we raise an error here?
      pass

  module_index[module._module__state.id] = len(module_index)


def _update_module_dynamic_state(
  module: Module,
  updates: tp.Union[State, tp.Sequence[State]],
) -> None:
  if isinstance(updates, State):
    new_states = (updates,)
  else:
    new_states = updates

  state: StateDict = {}
  for new_state in new_states:
    state.update(new_state.variables)

  for path, value in state.items():
    path_parts = path.split('/')
    _set_value_at_path(module, path_parts, value)


# _StaticSubmoduleState = tp.Literal["new", "updated"]
class _StaticModuleStatus(enum.Enum):
  NEW = enum.auto()
  UPDATED = enum.auto()


def _update_module_static_state(module: M, updates: M) -> None:
  cache: dict[Module, _StaticModuleStatus] = {}
  _update_module_static_state_recursive(
    module, updates, cache, _StaticModuleStatus.UPDATED, ()
  )


def _update_module_static_state_recursive(
  module: M,
  updates: M,
  cache: dict[Module, _StaticModuleStatus],
  status: _StaticModuleStatus,
  path: PathParts,
) -> None:
  if type(module) != type(updates):
    raise ValueError(
      f'Expected an instance of {type(module).__name__}, got'
      f' {type(updates).__name__}'
    )

  if updates in cache:
    if cache[updates] != status:
      str_path = '/'.join(path)
      if status is _StaticModuleStatus.NEW:
        raise ValueError(
          f'Trying to add a new submodule at path {str_path!r} but a'
          ' submodule with the same reference has been updated'
        )
      else:
        raise ValueError(
          f'Trying to update a submodule at path {str_path!r} but a new'
          ' submodule with the same reference has been added'
        )
    return

  cache[updates] = status

  module_vars = vars(module)
  for name, value in vars(updates).items():
    if isinstance(value, variableslib.Variable):
      continue
    elif isinstance(value, Module):
      if name in module_vars:
        _update_module_static_state_recursive(
          module_vars[name],
          value,
          cache,
          _StaticModuleStatus.UPDATED,
          (*path, name),
        )
      else:
        if value in cache:
          if cache[value] is not _StaticModuleStatus.NEW:
            raise ValueError(
              f'Trying to add a new submodule at path {name!r} but a'
              ' submodule with the same reference has been updated'
            )
        else:
          cache[value] = _StaticModuleStatus.NEW

      setattr(module, name, value)
    else:  # static field
      setattr(module, name, value)


def first_from(*args: tp.Optional[A]) -> A:
  """Return the first non-None argument."""
  for arg in args:
    if arg is not None:
      return arg
  raise ValueError('No non-None arguments found.')


def merge(
  state_and_def: tuple[tpe.Unpack[tuple[State, ...]], ModuleDef[M]]
) -> M:
  *states, moduledef = state_and_def
  return moduledef.merge(*states)
