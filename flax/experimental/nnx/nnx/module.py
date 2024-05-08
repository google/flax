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

import jax.tree_util as jtu

from flax.experimental.nnx.nnx import (
  filterlib,
  graph,
)
from flax.experimental.nnx.nnx import variables as variableslib
from flax.experimental.nnx.nnx.graph import GraphDef, GraphNode, GraphNodeMeta
from flax.experimental.nnx.nnx.proxy_caller import (
  CallableProxy,
  DelayedAccessor,
)
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

  return module


class Module(graph.GraphNode, metaclass=ModuleMeta):
  """"""

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

  @property
  def init(self: M) -> M:
    """Calls a method in initialization mode.

    When a method is called using ``init``, the ``is_initializing`` method
    will return ``True``. This is useful to implement Modules that support
    lazy initialization.

    Example::

      >>> from flax.experimental import nnx
      >>> import jax
      >>> import jax.numpy as jnp
      ...
      >>> class Linear(nnx.Module):
      ...   def __init__(self, dout, rngs: nnx.Rngs):
      ...     self.dout = dout
      ...     self.rngs = rngs
      ...
      ...   def __call__(self, x):
      ...     if self.is_initializing():
      ...       din = x.shape[-1]
      ...       if not hasattr(self, 'w'):
      ...         key = self.rngs.params()
      ...         self.w = nnx.Param(jax.random.uniform(key, (din, self.dout)))
      ...       if not hasattr(self, 'b'):
      ...         self.b = nnx.Param(jnp.zeros((self.dout,)))
      ...
      ...     return x @ self.w + self.b
      ...
      >>> linear = Linear(3, nnx.Rngs(0))
      >>> x = jnp.ones((5, 2))
      >>> y = linear.init(x)
      >>> linear.w.value.shape
      (2, 3)
      >>> linear.b.value.shape
      (3,)
      >>> y.shape
      (5, 3)
    """

    def _init_context(accessor: DelayedAccessor, *args, **kwargs):
      for _, value in graph.iter_nodes(self):
        if isinstance(value, GraphNode):
          value._graph_node__state._initializing = True

      method = accessor(self)
      try:
        out = method(*args, **kwargs)
      finally:
        for _, value in graph.iter_nodes(self):
          if isinstance(value, GraphNode):
            value._graph_node__state._initializing = False

      return out

    return CallableProxy(_init_context)  # type: ignore

  def is_initializing(self) -> bool:
    """Returns whether the Module is initializing.

    ``is_initializing`` returns ``True`` if the Module is currently being run
    under ``init``.
    """

    return self._graph_node__state._initializing

  def iter_modules(self) -> tp.Iterator[tuple[PathParts, Module]]:
    """Iterates over all nested Modules of the current Module, including the current Module.

    ``iter_modules`` creates a generator that yields the path and the Module instance, where
    the path is a tuple of strings or integers representing the path to the Module from the
    root Module.

    Example::

      >>> from flax.experimental import nnx
      ...
      >>> class Block(nnx.Module):
      ...   def __init__(self, din, dout, *, rngs: nnx.Rngs):
      ...     self.linear = nnx.Linear(din, dout, rngs=rngs)
      ...     self.dropout = nnx.Dropout(0.5)
      ...     self.batch_norm = nnx.BatchNorm(10, rngs=rngs)
      ...
      ...
      >>> model = Block(2, 5, rngs=nnx.Rngs(0))
      >>> for path, module in model.iter_modules():
      ...   print(path, type(module).__name__)
      ...
      () Block
      ('batch_norm',) BatchNorm
      ('dropout',) Dropout
      ('linear',) Linear
    """
    for path, value in graph.iter_nodes(self):
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
      >>> block.set_attributes(nnx.Dropout, deterministic=True)
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
    for path, module in self.iter_modules():
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

  def train(self, **attributes):
    """Sets the Module to training mode.

    ``train`` uses ``set_attributes`` to recursively set attributes ``deterministic=False``
    and ``use_running_average=False`` of all nested Modules that have these attributes.
    Its primarily used to control the runtime behavior of the ``Dropout`` and ``BatchNorm``
    Modules.

    Example::

      >>> from flax.experimental import nnx
      ...
      >>> class Block(nnx.Module):
      ...   def __init__(self, din, dout, *, rngs: nnx.Rngs):
      ...     self.linear = nnx.Linear(din, dout, rngs=rngs)
      ...     # initialize Dropout and BatchNorm in eval mode
      ...     self.dropout = nnx.Dropout(0.5, deterministic=True)
      ...     self.batch_norm = nnx.BatchNorm(10, use_running_average=True, rngs=rngs)
      ...
      >>> block = Block(2, 5, rngs=nnx.Rngs(0))
      >>> block.dropout.deterministic, block.batch_norm.use_running_average
      (True, True)
      >>> block.train()
      >>> block.dropout.deterministic, block.batch_norm.use_running_average
      (False, False)

    Args:
      **attributes: additional attributes passed to ``set_attributes``.
    """
    return self.set_attributes(
      deterministic=False,
      use_running_average=False,
      **attributes,
      raise_if_not_found=False,
    )

  def eval(self, **attributes):
    """Sets the Module to evaluation mode.

    ``eval`` uses ``set_attributes`` to recursively set attributes ``deterministic=True``
    and ``use_running_average=True`` of all nested Modules that have these attributes.
    Its primarily used to control the runtime behavior of the ``Dropout`` and ``BatchNorm``
    Modules.

    Example::

      >>> from flax.experimental import nnx
      ...
      >>> class Block(nnx.Module):
      ...   def __init__(self, din, dout, *, rngs: nnx.Rngs):
      ...     self.linear = nnx.Linear(din, dout, rngs=rngs)
      ...     self.dropout = nnx.Dropout(0.5)
      ...     self.batch_norm = nnx.BatchNorm(10, rngs=rngs)
      ...
      >>> block = Block(2, 5, rngs=nnx.Rngs(0))
      >>> block.dropout.deterministic, block.batch_norm.use_running_average
      (False, False)
      >>> block.eval()
      >>> block.dropout.deterministic, block.batch_norm.use_running_average
      (True, True)

    Args:
      **attributes: additional attributes passed to ``set_attributes``.
    """
    return self.set_attributes(
      deterministic=True,
      use_running_average=True,
      **attributes,
      raise_if_not_found=False,
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
  graphdef, state = graph.split(module)
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
  return graph.merge(graphdef, State(zip(paths, variables)))


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


