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

from collections import defaultdict
import dataclasses
import functools
import threading
import typing as tp
import typing_extensions as tpe

from flax.nnx.nnx import graph, rnglib
import flax.nnx.nnx.module as nnx_module
from flax.nnx.nnx.proxy_caller import (
  CallableProxy,
  DelayedAccessor,
)
from flax.nnx.nnx.object import Object

M = tp.TypeVar('M', bound='Module')
F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])


@dataclasses.dataclass
class CompactContext:
  module: 'Module'
  type_counter: defaultdict[type, int] = dataclasses.field(
    default_factory=lambda: defaultdict(lambda: 0)
  )


@dataclasses.dataclass
class ModuleContext(threading.local):
  parent_stack: list[tp.Optional[CompactContext]] = dataclasses.field(
    default_factory=lambda: [None]
  )


MODULE_CONTEXT = ModuleContext()


@dataclasses.dataclass
class Scope(Object):
  rngs: rnglib.Rngs


@tp.runtime_checkable
class _HasSetup(tp.Protocol):
  def setup(self) -> None: ...


class ModuleMeta(nnx_module.ModuleMeta):
  if not tp.TYPE_CHECKING:

    def __call__(cls, *args, **kwargs):
      return _module_meta_call(cls, *args, **kwargs)


def _module_meta_call(cls: tp.Type[M], *args, **kwargs) -> M:
  # compact behavior
  parent_ctx = MODULE_CONTEXT.parent_stack[-1]
  parent = None
  module: M

  if parent_ctx is not None:
    if 'parent' in kwargs:
      parent = kwargs.pop('parent')
      if parent is not None:
        raise ValueError(
          f"'parent' can only be set to None, got {type(parent).__name__}"
        )
      name = None
    else:
      type_index = parent_ctx.type_counter[cls]
      parent_ctx.type_counter[cls] += 1

      # define the name
      if 'name' in kwargs:
        name = kwargs.pop('name')
        if not isinstance(name, str):
          raise ValueError(f"'name' must be a 'str', got {type(name).__name__}")
      else:
        name = f'{cls.__name__}_{type_index}'

      parent = parent_ctx.module

      if hasattr(parent, name):
        module = getattr(parent, name)
        return module
  else:
    name = None

  module = nnx_module.ModuleMeta.__call__(cls, *args, **kwargs)
  module.scope = None

  if parent is not None:
    assert name is not None
    setattr(parent, name, module)
    # adopt the parent scope
    module.scope = parent.scope

  if dataclasses.is_dataclass(module):
    if isinstance(module, _HasSetup):
      module.setup()

  return module


class ModuleBase:
  if tp.TYPE_CHECKING:
    scope: Scope | None


@tpe.dataclass_transform(field_specifiers=(dataclasses.field,))  # type: ignore[not-supported-yet]
class Module(nnx_module.Module, ModuleBase, metaclass=ModuleMeta):
  def _set_scope(self, scope: Scope | None):
    """Recursively sets the scope for the Module and its children."""
    for _, value in graph.iter_graph(self):
      if isinstance(value, Module):
        value.scope = scope

  @property
  def init(self: M) -> M:
    """Calls a method in initialization mode.

    When a method is called using ``init``, the ``is_initializing`` method
    will return ``True``. This is useful to implement Modules that support
    lazy initialization.

    Example::

      >>> from flax import nnx
      >>> from flax.nnx import compat as nnc
      >>> import jax
      >>> import jax.numpy as jnp
      ...
      >>> class Linear(nnc.Module):
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
      for _, value in graph.iter_graph(self):
        if isinstance(value, Object):
          value._object__state._initializing = True

      method = accessor(self)
      try:
        out = method(*args, **kwargs)
      finally:
        for _, value in graph.iter_graph(self):
          if isinstance(value, Object):
            value._object__state._initializing = False

      return out

    return CallableProxy(_init_context)  # type: ignore

  def is_initializing(self) -> bool:
    """Returns whether the Module is initializing.

    ``is_initializing`` returns ``True`` if the Module is currently being run
    under ``init``.
    """

    return self._object__state._initializing

  def __init_subclass__(cls, experimental_pytree: bool = False) -> None:
    super().__init_subclass__(experimental_pytree)

    cls = dataclasses.dataclass(repr=False)(cls)


def compact(f: F) -> F:
  @functools.wraps(f)
  def compact_wrapper(self, *args, **kwargs):
    if not isinstance(self, Module):
      raise ValueError(
        f"Expected 'self' to be a nnx.compat.Module, got {type(self).__name__}"
      )

    MODULE_CONTEXT.parent_stack.append(CompactContext(self))

    try:
      return f(self, *args, **kwargs)
    finally:
      MODULE_CONTEXT.parent_stack.pop()

  return compact_wrapper  # type: ignore


# register Module as a dataclass_transform
