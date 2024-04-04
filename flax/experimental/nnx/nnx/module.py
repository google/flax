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

from __future__ import annotations

import dataclasses
import typing as tp
from functools import partial

import jax
import jax.tree_util as jtu

from flax.experimental.nnx.nnx import (
  filterlib,
  graph_utils,
)
from flax.experimental.nnx.nnx import variables as variableslib
from flax.experimental.nnx.nnx.graph_utils import GraphDef, GraphNodeMeta
from flax.experimental.nnx.nnx.proxy_caller import (
  ApplyCaller,
  CallableProxy,
  DelayedAccessor,
)
from flax.experimental.nnx.nnx.rnglib import Rngs
from flax.experimental.nnx.nnx.state import State
from flax.experimental.nnx.nnx.variables import Variable
from flax.typing import Path, PathParts

A = tp.TypeVar('A')
B = tp.TypeVar('B')
M = tp.TypeVar('M', bound='Module')
S = tp.TypeVar('S', bound=tp.Union[State, tuple[State, ...]])
V = tp.TypeVar('V', bound=variableslib.Variable[tp.Any])

StateMapping = tp.Mapping[Path, tp.Any]
tuple_reduce = lambda xs, x: xs + (x,)
tuple_init = lambda: ()

@tp.runtime_checkable
class _HasSetup(tp.Protocol):
  def setup(self) -> None:
    ...


class ModuleMeta(GraphNodeMeta):
  if not tp.TYPE_CHECKING:

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
      return _module_meta_call(cls, *args, **kwargs)


def _module_meta_call(cls: tp.Type[M], *args, **kwargs) -> M:
  module: M = GraphNodeMeta.__call__(cls, *args, **kwargs)

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

  return module


class Module(graph_utils.GraphNode, metaclass=ModuleMeta):
  @classmethod
  def init(cls: type[M], *args, **kwargs) -> tuple[GraphDef[M], State]:
    return cls(*args, **kwargs).split()

  @classmethod
  @property
  def create_abstract(cls: type[M]) -> type[M]:
    def lift_rngs(kwargs: dict[str, tp.Any]):
      if 'rngs' in kwargs and isinstance(rngs := kwargs['rngs'], tp.Mapping):
        kwargs['rngs'] = Rngs(rngs)
      return kwargs

    def _create_abstract(accessor: DelayedAccessor, *args, **kwargs):
      constructor = accessor(cls)
      if 'rngs' in kwargs and isinstance(rngs := kwargs['rngs'], Rngs):
        kwargs['rngs'] = rngs.fork()
      graphdef, state = jax.eval_shape(
        lambda: constructor(*args, **lift_rngs(kwargs)).split()
      )
      return graphdef.merge(state)

    return CallableProxy(_create_abstract)  # type: ignore

  @classmethod
  def partial_init(cls: type[M], state: State, *states: State) -> type[M]:
    """Creates a constuctor that initializes the Module with the given state.

    ``partial_init`` takes one or more States and returns a constructor that uses
    ``jax.jit`` to initialize the Module and update its state with the given
    States. Its semantically equivalent to::

      module = MyModule(*args, **kwargs)
      module.update(state, *states)

    However, thanks to dead code elimination the resulting constructor will only
    initialize the subset of ``Variable``'s that were part of the given state(s).

    Example::

      >>> import jax.numpy as jnp
      >>> import jax
      >>> from flax.experimental import nnx
      ...
      >>> bias = jax.random.normal(jax.random.key(0), (4,))
      >>> state = nnx.State({'bias': bias}) # in reality load it from a checkpoint
      >>> linear = nnx.Linear.partial_init(state)(2, 4, rngs=nnx.Rngs(1))
      >>> y = linear(jnp.ones((1, 2)))
      ...
      >>> assert jnp.allclose(linear.bias, bias)
      >>> assert y.shape == (1, 4)

    Args:
      state: The State to initialize the Module with.
      *states: Additional States to initialize the Module with.

    Returns:
      A constructor that initializes the Module with the given States.
    """
    states = (state, *states)

    def lift_rngs(kwargs: dict[str, tp.Any]):
      if 'rngs' in kwargs and isinstance(rngs := kwargs['rngs'], tp.Mapping):
        kwargs['rngs'] = Rngs(rngs)
      return kwargs

    def _partial_init(accessor: DelayedAccessor, *args, **kwargs):
      constructor: tp.Callable[[], M] = accessor(cls)
      if 'rngs' in kwargs and isinstance(rngs := kwargs['rngs'], Rngs):
        kwargs['rngs'] = rngs.fork()

      def _partial_init_constructor():
        module = constructor(*args, **lift_rngs(kwargs))
        graph_utils.update(module, *states)
        return graph_utils.split(module)

      graphdef: GraphDef[M]
      state: State
      graphdef, state = jax.jit(_partial_init_constructor)()
      module = graph_utils.merge(graphdef, state)
      return module

    return CallableProxy(_partial_init)  # type: ignore

  def clone(self: M) -> M:
    return graph_utils.merge(*self.split())

  @tp.overload
  def split(self: M) -> tuple[GraphDef[M], State]:
    ...

  @tp.overload
  def split(self: M, first: filterlib.Filter, /) -> tuple[GraphDef[M], State]:
    ...

  @tp.overload
  def split(
    self: M,
    first: filterlib.Filter,
    second: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
  ) -> tuple[GraphDef[M], State, tpe.Unpack[tuple[State, ...]]]:
    ...

  def split(
    self: M, *filters: filterlib.Filter
  ) -> tuple[GraphDef[M], State, tpe.Unpack[tuple[State, ...]]]:
    return graph_utils.split(self, *filters)

  def get_state(self) -> State:
    _, state = graph_utils.split(self)
    return state

  def get_graphdef(self: M) -> GraphDef[M]:
    graphdef, _ = graph_utils.split(self)
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
      module = graph_utils.clone(self)
      fn = accessor(module)
      out = fn(*args, **kwargs)
      return out, module

    return CallableProxy(_apply)  # type: ignore

  def update(
    self: M, update: graph_utils.Updates[M], /, *updates: graph_utils.Updates[M]
  ) -> None:
    graph_utils.update(self, update, *updates)

  def sow(
    self,
    variable_type: tp.Type[variableslib.Variable[tp.Any]],
    name: str,
    value: A,
    reduce_fn: tp.Callable[[B, A], B] = tuple_reduce,
    init_fn: tp.Callable[[], B] = tuple_init,  # type: ignore
  ) -> None:
    if hasattr(self, name):
      variable = getattr(self, name)
      if not isinstance(variable, variableslib.Variable):
        raise ValueError(
          f"Expected '{name}' to be a Variable, got {type(variable).__name__}"
        )
      elif type(variable) != variable_type:
        raise ValueError(
          f"Expected '{name}' to be of type '{variable_type.__name__}', "
          f"got '{type(variable).__name__}'"
        )
      variable.raw_value = reduce_fn(variable.raw_value, value)
    else:
      reduced_value = reduce_fn(init_fn(), value)
      setattr(self, name, variable_type(reduced_value))

  def modules(self) -> tp.Iterator[tuple[PathParts, Module]]:
    for path, value in graph_utils.iter_nodes(self):
      if isinstance(value, Module):
        yield path, value

  def set_attributes(
    self,
    *filters: filterlib.Filter,
    raise_if_not_found: bool = True,
    **attributes: tp.Any,
  ) -> None:
    """Sets the attributes of nested Modules including the current Module.
    If the attribute is not found in the Module, it is ignored.

    Example::

      >>> from flax.experimental import nnx
      ...
      >>> class Block(nnx.Module):
      ...   def __init__(self, din, dout, *, rngs: nnx.Rngs):
      ...     self.linear = nnx.Linear(din, dout, rngs=rngs)
      ...     self.dropout = nnx.Dropout(0.5, deterministic=False)
      ...     self.batch_norm = nnx.BatchNorm(10, use_running_average=False, rngs=rngs)
      ...
      >>> block = Block(2, 5, rngs=nnx.Rngs(0))
      >>> block.dropout.deterministic, block.batch_norm.use_running_average
      (False, False)
      >>> block.set_attributes(deterministic=True, use_running_average=True)
      >>> block.dropout.deterministic, block.batch_norm.use_running_average
      (True, True)

    ``Filter``'s can be used to set the attributes of specific Modules::

      >>> block = Block(2, 5, rngs=nnx.Rngs(0))
      >>> block.set_attributes(nnx.Dropout, deterministic=True, use_running_average=True)
      >>> # Only the dropout will be modified
      >>> block.dropout.deterministic, block.batch_norm.use_running_average
      (True, False)

    Args:
      *filters: Filters to select the Modules to set the attributes of.
      raise_if_not_found: If True (default), raises a ValueError if at least one attribute
        instance is not found in one of the selected Modules.
      **attributes: The attributes to set.
    """
    remaining_attributes = set(attributes.keys())
    if not filters:
      filters = (True,)
    predicates = tuple(map(filterlib.to_predicate, filters))
    for path, module in self.modules():
      for predicate in predicates:
        if predicate(path, module):
          for name, value in attributes.items():
            if hasattr(module, name):
              if name in remaining_attributes:
                remaining_attributes.remove(name)
              setattr(module, name, value)
          break

    if remaining_attributes and raise_if_not_found:
      raise ValueError(
        f'Could not find at least one instance of the following attributes: {remaining_attributes}'
      )

  def __init_subclass__(cls, experimental_pytree: bool = False) -> None:
    super().__init_subclass__()

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
  graphdef, state = graph_utils.split(module)
  key_values = sorted(state.raw_mapping.items())
  keys = tuple(key for key, _ in key_values)

  if with_keys:
    children = tuple((jtu.DictKey(key), value) for key, value in key_values)
  else:
    children = tuple(value for _, value in key_values)

  return children, (keys, graphdef)


def _module_unflatten(
  paths_moduledef: tuple[tuple[Path, ...], GraphDef[M]],
  variables: tuple[Variable[tp.Any], ...],
) -> M:
  paths, graphdef = paths_moduledef
  return graphdef.merge(State(zip(paths, variables)))


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


