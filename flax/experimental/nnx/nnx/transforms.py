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
import functools
import typing as tp
from abc import abstractmethod
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import jax.stages

from flax.experimental.nnx.nnx import (
  filterlib,
  rnglib,
  spmd,
  variables,
)
from flax.experimental.nnx.nnx.module import GraphDef, Module, ModuleMeta
from flax.experimental.nnx.nnx.proxy_caller import (
  CallableProxy,
  DelayedAccessor,
)
from flax.experimental.nnx.nnx.state import State
from flax.typing import Leaf

A = tp.TypeVar('A')
C = tp.TypeVar('C')
B = tp.TypeVar('B')
F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])
G = tp.TypeVar('G', bound=tp.Callable[..., tp.Any])
M = tp.TypeVar('M', bound=Module)
N = tp.TypeVar('N', bound=Module)

AxisName = tp.Hashable
Leaves = tp.List[Leaf]


def _check_args(args: tuple[tp.Any, ...]):
  """Check if Rngs is passed as a positional argument and raise an error."""
  for arg in args:
    if isinstance(arg, rnglib.Rngs):
      raise ValueError(
        "Rngs must be passed as a keyword argument named 'rngs', not a"
        ' positional argument'
      )


class LiftedModule(Module, tp.Generic[M]):
  @abstractmethod
  def _call(self, accessor: DelayedAccessor, *args, **kwargs) -> tp.Any:
    ...

  @property
  @abstractmethod
  def _submodule(self) -> M:
    ...

  def __call__(self, *args, **kwargs) -> tp.Any:
    return self.call(*args, **kwargs)  # type: ignore

  @property
  def call(self) -> tp.Any:
    module = self

    def check_and_call(accessor: DelayedAccessor, *args, **kwargs):
      _check_args(args)
      return self._call(accessor, *args, **kwargs)

    proxy = CallableProxy(check_and_call)

    while isinstance(module._submodule, LiftedModule):
      module = module._submodule
      proxy = proxy.call

    return proxy  # type: ignore


# -------------------------------
# jit
# -------------------------------

UNSPECIFIED = object()


@dataclasses.dataclass
class JITOptions:
  in_shardings: tp.Any
  out_shardings: tp.Any
  static_argnums: tp.Union[int, tp.Sequence[int], None]
  static_argnames: tp.Union[str, tp.Iterable[str], None]
  donate_argnums: tp.Union[int, tp.Sequence[int]]
  keep_unused: bool
  device: tp.Optional[jax.Device]
  backend: tp.Optional[str]
  inline: bool
  abstracted_axes: tp.Optional[tp.Any]

  def get_kwargs(self) -> dict[str, tp.Any]:
    kwargs = vars(self).copy()
    if kwargs['in_shardings'] is UNSPECIFIED:
      kwargs.pop('in_shardings')
    if kwargs['out_shardings'] is UNSPECIFIED:
      kwargs.pop('out_shardings')
    return kwargs


class JITMeta(ModuleMeta):
  def __call__(
    self,
    module_constructor: tp.Callable[..., M],
    *,
    in_shardings: tp.Any = UNSPECIFIED,
    out_shardings: tp.Any = UNSPECIFIED,
    static_argnums: tp.Union[int, tp.Sequence[int], None] = None,
    static_argnames: tp.Union[str, tp.Iterable[str], None] = None,
    donate_argnums: tp.Union[int, tp.Sequence[int]] = (),
    keep_unused: bool = False,
    device: tp.Optional[jax.Device] = None,
    backend: tp.Optional[str] = None,
    inline: bool = False,
    abstracted_axes: tp.Optional[tp.Any] = None,
  ) -> tp.Callable[..., 'JIT[M]']:
    super_call = super().__call__

    def _create_jit(*args, **kwargs) -> JIT[M]:
      _check_args(args)
      return super_call(
        module_constructor=module_constructor,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        keep_unused=keep_unused,
        device=device,
        backend=backend,
        inline=inline,
        abstracted_axes=abstracted_axes,
        # submodule args
        module_init_args=args,
        module_init_kwargs=kwargs,
      )

    return _create_jit


class JittedFn(tp.Protocol, tp.Generic[M]):
  def __call__(
    self, state_and_def: tuple[State | tuple[State, ...], GraphDef[M]]
  ) -> tuple[tuple[State | tuple[State, ...], GraphDef[M]], tp.Any]:
    ...


def get_jitted_fn(_module_type: type[M], f, options: JITOptions) -> JittedFn[M]:
  jit_kwargs = options.get_kwargs()

  @functools.partial(jax.jit, **jit_kwargs)
  def jitted_fn(
    state_and_def: tuple[State | tuple[State, ...], GraphDef[M]],
    *args,
    **kwargs,
  ):
    _check_args(args)
    states, graphdef = state_and_def

    if isinstance(states, State):
      states = (states,)

    if 'rngs' in kwargs:
      kwargs['rngs'] = rnglib.Rngs(kwargs['rngs'])
    module = graphdef.merge(*states)
    out = f(module, *args, **kwargs)

    updates = module.split()
    out = (updates, out)

    return out

  return jitted_fn


def jit_init(
  jitted_fn: JittedFn[M],
  module: M,
  args: tuple[tp.Any, ...],
  kwargs: dict[str, tp.Any],
) -> None:
  if not isinstance(module, Module):
    raise TypeError(f'Expected Module, got {type(module).__name__}')

  module = tp.cast(M, module)

  if 'rngs' in kwargs and isinstance(rngs := kwargs['rngs'], rnglib.Rngs):
    kwargs['rngs'] = rngs.fork()

  state_and_def = module.split()
  out = jitted_fn(state_and_def, *args, **kwargs)
  updates, _ = out
  module.update(updates)


def jit_apply(
  jitted_fn: JittedFn[M],
  module: M,
  args: tuple[tp.Any, ...],
  kwargs: dict[str, tp.Any],
) -> tp.Any:
  if not isinstance(module, Module):
    raise TypeError(f'Expected Module, got {type(module).__name__}')

  module = tp.cast(M, module)

  if 'rngs' in kwargs and isinstance(rngs := kwargs['rngs'], rnglib.Rngs):
    kwargs['rngs'] = rngs.fork()

  state_and_def = module.split()
  updates, out = jitted_fn(state_and_def, *args, **kwargs)
  module.update(updates)
  return out


class JIT(LiftedModule[M], metaclass=JITMeta):
  def __init__(
    self,
    module_constructor: tp.Callable[..., M],
    *,
    in_shardings: tp.Any = UNSPECIFIED,
    out_shardings: tp.Any = UNSPECIFIED,
    static_argnums: tp.Union[int, tp.Sequence[int], None] = None,
    static_argnames: tp.Union[str, tp.Iterable[str], None] = None,
    donate_argnums: tp.Union[int, tp.Sequence[int]] = (),
    keep_unused: bool = False,
    device: tp.Optional[jax.Device] = None,
    backend: tp.Optional[str] = None,
    inline: bool = False,
    abstracted_axes: tp.Optional[tp.Any] = None,
    # submodule args
    module_init_args: tuple[tp.Any, ...],
    module_init_kwargs: dict[str, tp.Any],
  ):
    self.options = JITOptions(
      in_shardings=in_shardings,
      out_shardings=out_shardings,
      static_argnums=static_argnums,
      static_argnames=static_argnames,
      donate_argnums=donate_argnums,
      keep_unused=keep_unused,
      device=device,
      backend=backend,
      inline=inline,
      abstracted_axes=abstracted_axes,
    )
    self.accessor: tp.Optional[DelayedAccessor] = None

    def jit_call_module(module, *args, **kwargs):
      assert self.accessor is not None
      f = self.accessor(module)
      return f(*args, **kwargs)

    self.jitted_fn: JittedFn[M] = get_jitted_fn(
      M, jit_call_module, self.options
    )
    self.module_constructor = module_constructor
    self.jit_module = self.module_constructor(
      *module_init_args, **module_init_kwargs
    )

  @property
  def _submodule(self) -> M:
    return self.jit_module

  def _call(self, accessor: DelayedAccessor, *args, **kwargs) -> Any:
    self.accessor = accessor
    try:
      out = jit_apply(self.jitted_fn, self.jit_module, args, kwargs)
    finally:
      self.accessor = None
    return out


def jit(
  f: F,
  *,
  in_shardings: tp.Any = UNSPECIFIED,
  out_shardings: tp.Any = UNSPECIFIED,
  static_argnums: tp.Union[int, tp.Sequence[int], None] = None,
  static_argnames: tp.Union[str, tp.Iterable[str], None] = None,
  donate_argnums: tp.Union[int, tp.Sequence[int]] = (),
  keep_unused: bool = False,
  device: tp.Optional[jax.Device] = None,
  backend: tp.Optional[str] = None,
  inline: bool = False,
  abstracted_axes: tp.Optional[tp.Any] = None,
  is_init: tp.Optional[bool] = None,
) -> F:
  if is_init is None:
    is_init = f.__name__ == '__init__'

  if static_argnames is None:
    static_argnames = []
  elif isinstance(static_argnames, str):
    static_argnames = [static_argnames]
  else:
    static_argnames = list(static_argnames)

  options = JITOptions(
    in_shardings=in_shardings,
    out_shardings=out_shardings,
    static_argnums=static_argnums,
    static_argnames=static_argnames,
    donate_argnums=donate_argnums,
    keep_unused=keep_unused,
    device=device,
    backend=backend,
    inline=inline,
    abstracted_axes=abstracted_axes,
  )
  jitted_fn = get_jitted_fn(Module, f, options)

  if is_init:

    @functools.wraps(f)
    def jit_init_wrapper(module: Module, *args, **kwargs):
      _check_args(args)
      jit_init(jitted_fn, module, args, kwargs)

    wrapper = jit_init_wrapper
    wrapper.inner = jitted_fn
  else:

    @functools.wraps(f)
    def jit_apply_wrapper(module: Module, *args, **kwargs):
      _check_args(args)
      return jit_apply(jitted_fn, module, args, kwargs)

    wrapper = jit_apply_wrapper
    wrapper.inner = jitted_fn

  return wrapper  # type: ignore


# -------------------------------
# grad
# -------------------------------


@dataclasses.dataclass
class GradOptions:
  wrt: filterlib.Filter
  has_aux: bool
  holomorphic: bool
  allow_int: bool
  reduce_axes: tp.Sequence[AxisName]
  return_value: bool


class GradMeta(ModuleMeta):
  def __call__(
    self,
    module_constructor: tp.Callable[..., M],
    *,
    wrt: filterlib.Filter = variables.Param,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: tp.Sequence[AxisName] = (),
    return_value: bool = False,
  ) -> tp.Callable[..., 'Grad[M]']:
    super_call = super().__call__

    def _create_grad(*args, **kwargs) -> Grad[M]:
      _check_args(args)
      return super_call(
        module_constructor=module_constructor,
        wrt=wrt,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
        reduce_axes=reduce_axes,
        return_value=return_value,
        # submodule args
        module_init_args=args,
        module_init_kwargs=kwargs,
      )

    return _create_grad


class Grad(LiftedModule[M], metaclass=GradMeta):
  def __init__(
    self,
    module_constructor: tp.Callable[..., M],
    *,
    wrt: filterlib.Filter = variables.Param,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: tp.Sequence[AxisName] = (),
    return_value: bool = False,
    # submodule args
    module_init_args: tuple[tp.Any, ...],
    module_init_kwargs: dict[str, tp.Any],
  ):
    self.options = GradOptions(
      wrt=wrt,
      has_aux=has_aux,
      holomorphic=holomorphic,
      allow_int=allow_int,
      reduce_axes=reduce_axes,
      return_value=return_value,
    )
    self.module_constructor = module_constructor
    self.grad_module = self.module_constructor(
      *module_init_args, **module_init_kwargs
    )

  @property
  def _submodule(self) -> M:
    return self.grad_module

  def _call(self, accessor: DelayedAccessor, *args, **kwargs) -> Any:
    def grad_call_apply(module, *args, **kwargs):
      return accessor(module)(*args, **kwargs)

    return grad_apply(
      self.options, grad_call_apply, self.grad_module, *args, **kwargs
    )


def grad_apply(options: GradOptions, f, module: Module, *args, **kwargs):
  if not isinstance(module, Module):
    raise TypeError(f'Expected a Module, got {type(module).__name__}')

  predicate = filterlib.to_predicate(options.wrt)

  diff, nondiff, graphdef = module.split(predicate, ...)
  transform = jax.value_and_grad if options.return_value else jax.grad

  @functools.partial(
    transform,
    argnums=0,  # we'll handle this ourselves
    has_aux=True,
    holomorphic=options.holomorphic,
    allow_int=options.allow_int,
    reduce_axes=options.reduce_axes,
  )
  def grad_fn(diff: State):
    nonlocal graphdef

    module = graphdef.merge(diff, nondiff)
    out = f(module, *args, **kwargs)

    updates, graphdef = module.split()
    if options.has_aux:
      loss, aux = out
      out = (loss, (updates, aux))
    else:
      out = (out, updates)

    return out

  out = grad_fn(diff)

  updates: State
  if options.return_value:
    if options.has_aux:
      (loss, (updates, aux)), grads = out
      out = (loss, aux), grads
    else:
      (loss, updates), grads = out
      out = loss, grads
  else:
    if options.has_aux:
      grads, (updates, aux) = out
      out = grads, aux
    else:
      out, updates = out

  module.update(updates, graphdef)
  return out


@tp.overload
def grad(
  f: tp.Callable[..., tp.Any],
  wrt: filterlib.Filter = variables.Param,
  *,
  holomorphic: bool = False,
  allow_int: bool = False,
  reduce_axes: tp.Sequence[AxisName] = (),
) -> tp.Callable[..., State]:
  ...


@tp.overload
def grad(
  f: tp.Callable[..., tp.Any],
  wrt: filterlib.Filter = variables.Param,
  *,
  has_aux: tp.Literal[True],
  holomorphic: bool = False,
  allow_int: bool = False,
  reduce_axes: tp.Sequence[AxisName] = (),
) -> tp.Callable[..., tuple[State, tp.Any]]:
  ...


def grad(
  f: tp.Callable[..., tp.Any],
  wrt: filterlib.Filter = variables.Param,
  *,
  has_aux: bool = False,
  holomorphic: bool = False,
  allow_int: bool = False,
  reduce_axes: tp.Sequence[AxisName] = (),
) -> tp.Callable[..., tp.Union[tuple[State, tp.Any], State]]:
  if f.__name__ == '__init__':
    raise ValueError('Cannot use `grad` with `__init__`')

  options = GradOptions(
    wrt=wrt,
    has_aux=has_aux,
    holomorphic=holomorphic,
    allow_int=allow_int,
    reduce_axes=reduce_axes,
    return_value=False,
  )

  @functools.wraps(f)
  def grad_wrapper(module: Module, *args, **kwargs):
    _check_args(args)
    return grad_apply(options, f, module, *args, **kwargs)

  return grad_wrapper  # type: ignore


@tp.overload
def value_and_grad(
  f: tp.Callable[..., tp.Any],
  wrt: filterlib.Filter = variables.Param,
  *,
  holomorphic: bool = False,
  allow_int: bool = False,
  reduce_axes: tp.Sequence[AxisName] = (),
) -> tp.Callable[..., tuple[jax.Array, State]]:
  ...


@tp.overload
def value_and_grad(
  f: tp.Callable[..., tp.Any],
  wrt: filterlib.Filter = variables.Param,
  *,
  has_aux: tp.Literal[True],
  holomorphic: bool = False,
  allow_int: bool = False,
  reduce_axes: tp.Sequence[AxisName] = (),
) -> tp.Callable[..., tuple[tuple[jax.Array, tp.Any], State]]:
  ...


def value_and_grad(
  f: tp.Callable[..., tp.Any],
  wrt: filterlib.Filter = variables.Param,
  *,
  has_aux: bool = False,
  holomorphic: bool = False,
  allow_int: bool = False,
  reduce_axes: tp.Sequence[AxisName] = (),
) -> tp.Callable[
  ...,
  tp.Union[tuple[tuple[jax.Array, tp.Any], State], tuple[jax.Array, State]],
]:
  if f.__name__ == '__init__':
    raise ValueError('Cannot use `value_and_grad` with `__init__`')

  options = GradOptions(
    wrt=wrt,
    has_aux=has_aux,
    holomorphic=holomorphic,
    allow_int=allow_int,
    reduce_axes=reduce_axes,
    return_value=True,
  )

  @functools.wraps(f)
  def value_and_grad_wrapper(module: Module, *args, **kwargs):
    _check_args(args)
    return grad_apply(options, f, module, *args, **kwargs)

  return value_and_grad_wrapper  # type: ignore


# -------------------------------
# scan
# -------------------------------


@dataclasses.dataclass
class ScanOptions:
  variable_axes: tp.Mapping[filterlib.Filter, int]
  broadcast_rngs: filterlib.Filter
  in_args_axes: tp.Any
  in_kwargs_axes: tp.Any
  out_axes: tp.Any
  length: tp.Optional[int]
  reverse: bool
  unroll: int
  scan_metadata: tp.Mapping[str, tp.Any]
  scan_output: bool


class ScanMeta(ModuleMeta):
  def __call__(
    self,
    module_constructor: tp.Callable[..., M],
    *,
    variable_axes: tp.Mapping[filterlib.Filter, int] = MappingProxyType({}),
    broadcast_rngs: filterlib.Filter = None,
    in_args_axes: tp.Any = 0,
    in_kwargs_axes: tp.Any = 0,
    out_axes: tp.Any = 0,
    length: tp.Optional[int] = None,
    reverse: bool = False,
    unroll: int = 1,
    scan_metadata: tp.Mapping[str, tp.Any] = MappingProxyType({}),
    scan_output: bool = True,
  ) -> tp.Callable[..., 'Scan[M]']:
    super_call = super().__call__

    def _create_scan(*args, **kwargs) -> Scan[M]:
      _check_args(args)
      return super_call(
        module_constructor=module_constructor,
        module_init_args=args,
        module_init_kwargs=kwargs,
        variable_axes=variable_axes,
        broadcast_rngs=broadcast_rngs,
        in_args_axes=in_args_axes,
        in_kwargs_axes=in_kwargs_axes,
        out_axes=out_axes,
        length=length,
        reverse=reverse,
        unroll=unroll,
        scan_metadata=scan_metadata,
        scan_output=scan_output,
      )

    return _create_scan


class Scan(LiftedModule[M], metaclass=ScanMeta):
  def __init__(
    self,
    module_constructor: tp.Callable[..., M],
    *,
    variable_axes: tp.Mapping[filterlib.Filter, int] = MappingProxyType({}),
    broadcast_rngs: filterlib.Filter = None,
    in_args_axes: tp.Any = 0,
    in_kwargs_axes: tp.Any = 0,
    out_axes: tp.Any = 0,
    length: tp.Optional[int] = None,
    reverse: bool = False,
    unroll: int = 1,
    scan_metadata: tp.Mapping[str, tp.Any] = MappingProxyType({}),
    scan_output: bool = True,
    # submodule args
    module_init_args: tuple[tp.Any, ...],
    module_init_kwargs: dict[str, tp.Any],
  ):
    self.module_constructor = module_constructor
    self.options = ScanOptions(
      variable_axes=variable_axes,
      broadcast_rngs=broadcast_rngs,
      in_args_axes=in_args_axes,
      in_kwargs_axes=in_kwargs_axes,
      out_axes=out_axes,
      length=length,
      reverse=reverse,
      unroll=unroll,
      scan_metadata=scan_metadata,
      scan_output=scan_output,
    )
    self.scan_module = scan_init(
      self.options, module_constructor, module_init_args, module_init_kwargs
    )

  @property
  def _submodule(self) -> M:
    return self.scan_module

  def _call(
    self, accessor: DelayedAccessor, *args, **kwargs
  ) -> tuple[tp.Any, tp.Any]:
    if len(args) < 1:
      raise TypeError(
        f'Expected at least 1 positional arguments, got {len(args)}'
      )
    _check_args(args)
    carry_arg, args = args[0], args[1:]

    def scan_call_apply(module, *args, **kwargs):
      return accessor(module)(*args, **kwargs)

    return scan_apply(
      self.options,
      scan_call_apply,
      self.scan_module,
      carry_arg,
      args,
      kwargs,
    )


class ScanCall(tp.Protocol, tp.Generic[C, B]):
  def __call__(
    self,
    module: Module,
    carry_arg: C,
    *args: tp.Any,
    **kwargs: tp.Any,
  ) -> tuple[C, B] | C:
    ...


def scan_init(
  options: ScanOptions,
  module_constructor: tp.Callable[..., M],
  module_init_args: tuple[tp.Any, ...],
  module_init_kwargs: dict[str, tp.Any],
) -> M:
  if options.variable_axes and options.length is None:
    raise ValueError('Cannot use variable_axes without specifying a length')

  _check_args(module_init_args)

  rngs = module_init_kwargs.pop('rngs', None)

  if rngs is not None and not isinstance(rngs, rnglib.Rngs):
    raise TypeError(f'Expected a Rngs, got {type(rngs).__name__}')

  split_keys = []

  if rngs is not None:
    if not isinstance(rngs, rnglib.Rngs):
      raise TypeError(f'Expected a Rngs, got {type(rngs).__name__}')

    forked_rngs = rngs.fork(
      {filterlib.Not(options.broadcast_rngs): options.length}
    )
    split_keys, broadcast_keys = forked_rngs.splits, forked_rngs.broadcasts

    if split_keys and options.length is None:
      raise ValueError('Cannot split RNGs without specifying a length')

  else:
    split_keys = None
    broadcast_keys = None

  graphdef: tp.Optional[GraphDef[M]] = None

  def _init_state(split_keys, broadcast_keys):
    nonlocal graphdef

    if split_keys is not None:
      assert broadcast_keys is not None
      module_init_kwargs['rngs'] = rnglib.Rngs(**split_keys, **broadcast_keys)

    module = module_constructor(*module_init_args, **module_init_kwargs)

    # lift module
    filters = (*options.variable_axes.keys(), ...)

    *states, graphdef = module.split(*filters)

    return tuple(states)

  if split_keys is not None or options.variable_axes:
    init_out_axes = (*options.variable_axes.values(), None)
    _init_state = jax.vmap(
      _init_state,
      in_axes=(0, None),
      out_axes=init_out_axes,
      axis_size=options.length,
    )

  *axes_states, carry_state = _init_state(split_keys, broadcast_keys)
  graphdef = tp.cast(GraphDef[M], graphdef)

  # add additional axis name to Variable.sharding
  if spmd.PARTITION_NAME in options.scan_metadata:
    axes_states = [
      spmd.add_axis(state, index, options.scan_metadata)
      for state, index in zip(axes_states, options.variable_axes.values())
    ]

  module = graphdef.merge(*axes_states, carry_state)

  return module


def scan_apply(
  options: ScanOptions,
  f: ScanCall[C, B],
  module: Module,
  carry_arg: C,
  args: tuple[tp.Any, ...],
  kwargs: dict[str, tp.Any],
) -> tuple[C, B] | C:
  rngs = kwargs.pop('rngs', None)

  # split module state
  filters = (*options.variable_axes.keys(), ...)
  *scan_states, carry_state, graphdef = module.split(*filters)

  # transpose axes state
  scan_states = tuple(
    jax.tree_util.tree_map(lambda x: jnp.moveaxis(x, axis, 0), axes_state)
    for axes_state, axis in zip(scan_states, options.variable_axes.values())
  )
  # transpose axes arg
  scan_args = jax.tree_util.tree_map(
    lambda axis, node: jax.tree_util.tree_map(
      lambda x: jnp.moveaxis(x, axis, 0), node
    )
    if axis is not None
    else None,
    options.in_args_axes,
    args,
    is_leaf=lambda x: x is None,
  )
  broadcast_args = jax.tree_map(
    lambda axis, node: node if axis is None else None,
    options.in_args_axes,
    args,
    is_leaf=lambda x: x is None,
  )
  scan_kwargs = jax.tree_util.tree_map(
    lambda axis, node: jax.tree_util.tree_map(
      lambda x: jnp.moveaxis(x, axis, 0), node
    )
    if axis is not None
    else None,
    options.in_kwargs_axes,
    kwargs,
    is_leaf=lambda x: x is None,
  )
  broadcast_kwargs = jax.tree_util.tree_map(
    lambda axis, node: None if axis is not None else node,
    options.in_kwargs_axes,
    kwargs,
    is_leaf=lambda x: x is None,
  )

  # infer length
  lengths: tp.Set[int] = set(
    x.shape[0]
    for x in jax.tree_util.tree_leaves((scan_states, scan_args, scan_kwargs))
  )

  if len(lengths) > 1:
    raise ValueError(
      'Inconsistent lengths between variable_axes states and '
      f'arguments: {lengths}'
    )
  elif len(lengths) == 0:
    if options.length is None:
      raise ValueError(
        'Cannot infer length from variable_axes states or axes_arg, '
        'please specify `length`'
      )
    length = options.length
  else:
    length = lengths.pop()
    if options.length is not None and options.length != length:
      raise ValueError(
        f'Specified length {options.length} is not the same as the inferred '
        f'length {length}'
      )

  # split rng state
  if rngs is not None:
    if not isinstance(rngs, rnglib.Rngs):
      raise TypeError(f'Expected a Rngs, got {type(rngs).__name__}')
    forked_rngs = rngs.fork({filterlib.Not(options.broadcast_rngs): length})
    split_keys, broadcast_keys = forked_rngs.splits, forked_rngs.broadcasts
  else:
    split_keys = None
    broadcast_keys = None

  moduledef_out: tp.Optional[GraphDef[Module]] = None

  def scan_fn(
    carry: tuple[State, tp.Any],
    scan: tuple[
      dict[str, rnglib.RngStream] | None,
      tuple[State, ...],
      tuple[tp.Any, ...],
      dict[str, tp.Any],
    ],
  ):
    nonlocal moduledef_out
    carry_state, carry_arg = carry
    split_keys, scan_states, scan_args, scan_kwargs = scan

    # merge args and kwargs
    args = jax.tree_util.tree_map(
      lambda axis, scan, broadcast: scan if axis is not None else broadcast,
      options.in_args_axes,
      scan_args,
      broadcast_args,
      is_leaf=lambda x: x is None,
    )
    kwargs = jax.tree_util.tree_map(
      lambda axis, scan, broadcast: scan if axis is not None else broadcast,
      options.in_kwargs_axes,
      scan_kwargs,
      broadcast_kwargs,
      is_leaf=lambda x: x is None,
    )

    # merge rng state
    if split_keys is not None:
      assert broadcast_keys is not None
      kwargs['rngs'] = rnglib.Rngs(**split_keys, **broadcast_keys)

    # remove metadata axis name from Variable.sharding
    if spmd.PARTITION_NAME in options.scan_metadata:
      scan_states = [
        spmd.remove_axis(state, index, options.scan_metadata)
        for state, index in zip(scan_states, options.variable_axes.values())
      ]

    # merge module state
    module = graphdef.merge(*scan_states, carry_state)

    output = f(module, carry_arg, *args, **kwargs)

    if options.scan_output:
      if not isinstance(output, tuple) or len(output) != 2:
        raise ValueError(
          'Expected a tuple of length 2 as the output of the scan function, '
          f'got {output}'
        )
      output = tp.cast(tuple[C, B], output)
      carry_out, scan_out = output
    else:
      output = tp.cast(C, output)
      carry_out = output
      scan_out = None

    # split module state
    *scan_states_out, carry_state_out, moduledef_out = module.split(*filters)
    carry_state_new = carry_state_out - carry_state

    # remove new carry state
    carry_state_out = carry_state_out - carry_state_new

    # add metadata axis name to Variable.sharding
    if spmd.PARTITION_NAME in options.scan_metadata:
      scan_states_out = [
        spmd.add_axis(state, index, options.scan_metadata)
        for state, index in zip(scan_states_out, options.variable_axes.values())
      ]

    full_carry_out = (carry_state_out, carry_out)
    full_scan_out = (scan_states_out, carry_state_new, scan_out)

    return full_carry_out, full_scan_out

  carry = (carry_state, carry_arg)
  scan = (split_keys, scan_states, scan_args, scan_kwargs)

  full_carry_out, full_scan_out = jax.lax.scan(
    scan_fn,
    carry,
    scan,
    length=length,
    reverse=options.reverse,
    unroll=options.unroll,
  )
  carry_state, carry_out = full_carry_out
  scan_states, carry_state_new, scan_out = full_scan_out
  assert moduledef_out is not None

  # transpose axes state
  scan_states = tuple(
    jax.tree_util.tree_map(lambda x: jnp.moveaxis(x, 0, axis), axes_state)
    for axes_state, axis in zip(scan_states, options.variable_axes.values())
  )
  # transpose axes arg
  scan_out = jax.tree_util.tree_map(
    lambda axis, node: jax.tree_util.tree_map(
      lambda x: jnp.moveaxis(x, 0, axis), node
    ),
    options.out_axes,
    scan_out,
  )
  # slice new carry state
  carry_state_new = jax.tree_util.tree_map(lambda x: x[0], carry_state_new)

  module.update(((*scan_states, carry_state, carry_state_new), moduledef_out))

  if options.scan_output:
    return carry_out, scan_out
  else:
    return carry_out


def scan(
  f: F,
  *,
  variable_axes: tp.Mapping[filterlib.Filter, int] = MappingProxyType({}),
  broadcast_rngs: filterlib.Filter = None,
  in_args_axes: tp.Any = 0,
  in_kwargs_axes: tp.Any = 0,
  out_axes: tp.Any = 0,
  length: tp.Optional[int] = None,
  reverse: bool = False,
  unroll: int = 1,
  is_init: tp.Optional[bool] = None,
  scan_metadata: tp.Mapping[str, tp.Any] = {},
  scan_output: bool = True,
) -> F:
  if is_init is None:
    is_init = f.__name__ == '__init__'

  options = ScanOptions(
    variable_axes=variable_axes,
    broadcast_rngs=broadcast_rngs,
    in_args_axes=in_args_axes,
    in_kwargs_axes=in_kwargs_axes,
    out_axes=out_axes,
    length=length,
    reverse=reverse,
    unroll=unroll,
    scan_metadata=scan_metadata,
    scan_output=scan_output,
  )

  if is_init:

    @functools.wraps(f)
    def scan_init_wrapper(module: Module, *args, **kwargs):
      def module_constructor(*args, **kwargs):
        _check_args(args)
        f(module, *args, **kwargs)
        return module

      lifted_module = scan_init(options, module_constructor, args, kwargs)
      module.update(lifted_module)

    wrapper = scan_init_wrapper

  else:

    @functools.wraps(f)
    def scan_apply_wrapper(
      module: Module,
      *args,
      **kwargs,
    ) -> tuple[C, tp.Any]:
      if len(args) < 2:
        raise TypeError(
          f'Expected at least 2 positional arguments, got {len(args)}'
        )
      _check_args(args)

      carry_arg, args = args[0], args[1:]
      return scan_apply(options, f, module, carry_arg, args, kwargs)

    wrapper = scan_apply_wrapper

  return wrapper  # type: ignore


# -------------------------------
# remat
# -------------------------------


class RematMeta(ModuleMeta):
  def __call__(
    self,
    module_constructor: tp.Callable[..., M],
    # variables: lift.CollectionFilter = True,
    # rngs: lift.PRNGSequenceFilter = True,
    prevent_cse: bool = True,
    static_argnums: tp.Union[int, tuple[int, ...]] = (),
    policy: tp.Optional[tp.Callable[..., bool]] = None,
  ) -> tp.Callable[..., 'Remat[M]']:
    super_call = super().__call__

    def create_remat(*args, **kwargs) -> Remat[M]:
      _check_args(args)
      return super_call(
        module_constructor=module_constructor,
        module_init_args=args,
        module_init_kwargs=kwargs,
        prevent_cse=prevent_cse,
        static_argnums=static_argnums,
        policy=policy,
      )

    return create_remat


@dataclasses.dataclass
class RematOptions:
  prevent_cse: bool
  static_argnums: tp.Union[int, tuple[int, ...]]
  policy: tp.Optional[tp.Callable[..., bool]]

  def __post_init__(self):
    if isinstance(self.static_argnums, int):
      self.static_argnums = (self.static_argnums,)

    # add 2 as an offset to account for state and keys
    self.static_argnums = tuple(
      x + 2 if x >= 0 else x for x in self.static_argnums
    )


class Remat(LiftedModule[M], metaclass=RematMeta):
  def __init__(
    self,
    *,
    module_constructor: tp.Callable[..., M],
    prevent_cse: bool = True,
    static_argnums: tp.Union[int, tuple[int, ...]] = (),
    policy: tp.Optional[tp.Callable[..., bool]] = None,
    # submodule args
    module_init_args: tuple[tp.Any, ...],
    module_init_kwargs: dict[str, tp.Any],
  ):
    self.options = RematOptions(
      prevent_cse=prevent_cse,
      static_argnums=static_argnums,
      policy=policy,
    )
    self.module_constructor = module_constructor
    self.remat_module = self.module_constructor(
      *module_init_args, **module_init_kwargs
    )

  @property
  def _submodule(self) -> M:
    return self.remat_module

  def _call(
    self,
    accessor: DelayedAccessor,
    *args,
    rngs: tp.Optional[rnglib.Rngs] = None,
  ) -> tp.Any:
    def remat_call_apply(module, *args, **kwargs):
      return accessor(module)(*args, **kwargs)

    return remat_apply(
      self.options,
      remat_call_apply,
      self.remat_module,
      args,
      rngs,
    )


class RematCall(tp.Protocol):
  def __call__(self, *args, rngs: tp.Optional[rnglib.Rngs]) -> tp.Any:
    ...


def remat_apply(
  options: RematOptions,
  f: RematCall,
  module: Module,
  args: tuple[tp.Any, ...],
  rngs: tp.Optional[rnglib.Rngs],
):
  _check_args(args)

  state, graphdef = module.split()
  keys = rngs.fork() if rngs is not None else None

  def _remat_fn(
    state: State,
    keys: tp.Optional[dict[str, jax.Array]],
    *args,
  ) -> tuple[tuple[State, GraphDef[Module]], tp.Any]:
    kwargs = {}
    if keys is not None:
      kwargs['rngs'] = rnglib.Rngs(keys)

    module = graphdef.merge(state)
    out = f(module, *args, **kwargs)

    state_and_def = module.split()

    return state_and_def, out

  state_and_def: tuple[State, GraphDef[Module]]
  state_and_def, out = jax.checkpoint(
    _remat_fn,
    prevent_cse=options.prevent_cse,
    static_argnums=options.static_argnums,
    policy=options.policy,
  )(state, keys, *args)

  module.update(state_and_def)

  return out


def remat(
  f: F,
  *,
  # variables: lift.CollectionFilter,
  # rngs: lift.PRNGSequenceFilter,
  prevent_cse: bool = True,
  static_argnums: tp.Union[int, tuple[int, ...]] = (),
  policy: tp.Optional[tp.Callable[..., bool]] = None,
  is_init: tp.Optional[bool] = None,
) -> F:
  if is_init is None:
    is_init = f.__name__ == '__init__'

  options = RematOptions(
    # variables=variables,
    # rngs=rngs,
    prevent_cse=prevent_cse,
    static_argnums=static_argnums,
    policy=policy,
  )

  if is_init:
    return f
  else:

    @functools.wraps(f)
    def remat_wrapper(
      module: Module, *args, rngs: tp.Optional[rnglib.Rngs] = None
    ):
      return remat_apply(options, f, module, args, rngs)

    return remat_wrapper  # type: ignore


# -------------------------------
# vmap
# -------------------------------


@dataclasses.dataclass
class VmapOptions:
  variable_axes: tp.Mapping[filterlib.Filter, int]
  broadcast_rngs: filterlib.Filter
  in_args_axes: tp.Any
  in_kwargs_axes: tp.Any
  out_axes: tp.Any
  axis_size: int | None
  axis_name: str | None
  spmd_axis_name: str | None
  vmap_metadata: tp.Mapping[str, tp.Any]


class VmapMeta(ModuleMeta):
  def __call__(
    self,
    module_constructor: tp.Callable[..., M],
    *,
    variable_axes: tp.Mapping[filterlib.Filter, int] = MappingProxyType({}),
    broadcast_rngs: filterlib.Filter = None,
    in_args_axes: tp.Any = 0,
    in_kwargs_axes: tp.Any = 0,
    out_axes: tp.Any = 0,
    axis_size: int | None = None,
    axis_name: str | None = None,
    spmd_axis_name: str | None = None,
    vmap_metadata: tp.Mapping[str, tp.Any] = MappingProxyType({}),
  ) -> tp.Callable[..., 'Vmap[M]']:
    super_call = super().__call__

    def _create_scan(*args, **kwargs) -> Scan[M]:
      _check_args(args)
      return super_call(
        module_constructor=module_constructor,
        module_init_args=args,
        module_init_kwargs=kwargs,
        variable_axes=variable_axes,
        broadcast_rngs=broadcast_rngs,
        in_args_axes=in_args_axes,
        in_kwargs_axes=in_kwargs_axes,
        out_axes=out_axes,
        axis_size=axis_size,
        axis_name=axis_name,
        spmd_axis_name=spmd_axis_name,
        vmap_metadata=vmap_metadata,
      )

    return _create_scan


class Vmap(LiftedModule[M], metaclass=VmapMeta):
  def __init__(
    self,
    module_constructor: tp.Callable[..., M],
    *,
    variable_axes: tp.Mapping[filterlib.Filter, int] = MappingProxyType({}),
    broadcast_rngs: filterlib.Filter = None,
    in_args_axes: tp.Any = 0,
    in_kwargs_axes: tp.Any = 0,
    out_axes: tp.Any = 0,
    axis_size: int | None = None,
    axis_name: str | None = None,
    spmd_axis_name: str | None = None,
    vmap_metadata: tp.Mapping[str, tp.Any] = MappingProxyType({}),
    # submodule args
    module_init_args: tuple[tp.Any, ...],
    module_init_kwargs: dict[str, tp.Any],
  ):
    self.module_constructor = module_constructor
    self.options = VmapOptions(
      variable_axes=variable_axes,
      broadcast_rngs=broadcast_rngs,
      in_args_axes=in_args_axes,
      in_kwargs_axes=in_kwargs_axes,
      out_axes=out_axes,
      axis_size=axis_size,
      axis_name=axis_name,
      spmd_axis_name=spmd_axis_name,
      vmap_metadata=vmap_metadata,
    )
    self.vmap_module = vmap_init(
      self.options, module_constructor, module_init_args, module_init_kwargs
    )

  @property
  def _submodule(self) -> M:
    return self.vmap_module

  def _call(
    self, accessor: DelayedAccessor, *args, **kwargs
  ) -> tuple[tp.Any, tp.Any]:
    _check_args(args)

    def vmap_call_apply(module, *args, **kwargs):
      return accessor(module)(*args, **kwargs)

    return vmap_apply(
      self.options,
      vmap_call_apply,
      self.vmap_module,
      args,
      kwargs,
    )


class VmapCall(tp.Protocol):
  def __call__(
    self,
    module: Module,
    *args: tp.Any,
    **kwargs: tp.Any,
  ) -> tp.Any:
    ...


def vmap_init(
  options: VmapOptions,
  module_constructor: tp.Callable[..., M],
  module_init_args: tuple[tp.Any, ...],
  module_init_kwargs: dict[str, tp.Any],
) -> M:
  if options.variable_axes and options.axis_size is None:
    raise ValueError('Cannot use variable_axes without specifying a length')

  _check_args(module_init_args)

  rngs = module_init_kwargs.pop('rngs', None)

  if rngs is not None and not isinstance(rngs, rnglib.Rngs):
    raise TypeError(f'Expected a Rngs, got {type(rngs).__name__}')

  if rngs is not None:
    if not isinstance(rngs, rnglib.Rngs):
      raise TypeError(f'Expected a Rngs, got {type(rngs).__name__}')
    forked_rngs = rngs.fork(
      {filterlib.Not(options.broadcast_rngs): options.axis_size}
    )
    split_keys, broadcast_keys = forked_rngs.splits, forked_rngs.broadcasts
    if split_keys and options.axis_size is None:
      raise ValueError('Cannot split RNGs without specifying a length')
  else:
    split_keys = None
    broadcast_keys = None

  graphdef: tp.Optional[GraphDef[M]] = None

  def _init_state(split_keys, broadcast_keys):
    nonlocal graphdef

    if split_keys is not None:
      assert broadcast_keys is not None
      module_init_kwargs['rngs'] = rnglib.Rngs(**split_keys, **broadcast_keys)

    module = module_constructor(*module_init_args, **module_init_kwargs)

    # lift module
    filters = (*options.variable_axes.keys(), ...)

    *states, graphdef = module.split(*filters)

    return tuple(states)

  if split_keys is not None or options.variable_axes:
    init_out_axes = (*options.variable_axes.values(), None)
    _init_state = jax.vmap(
      _init_state,
      in_axes=(0, None),
      out_axes=init_out_axes,
      axis_size=options.axis_size,
    )

  *axes_states, carry_state = _init_state(split_keys, broadcast_keys)
  graphdef = tp.cast(GraphDef[M], graphdef)

  # add additional axis name to Variable.sharding
  if spmd.PARTITION_NAME in options.vmap_metadata:
    axes_states = [
      spmd.add_axis(state, index, options.vmap_metadata)
      for state, index in zip(axes_states, options.variable_axes.values())
    ]

  module = graphdef.merge(*axes_states, carry_state)
  return module


def vmap_apply(
  options: VmapOptions,
  f: VmapCall,
  module: Module,
  args: tuple[tp.Any, ...],
  kwargs: dict[str, tp.Any],
) -> tp.Any:
  rngs = kwargs.pop('rngs', None)

  # split module state
  filters = (*options.variable_axes.keys(), ...)
  *vectorized_states, broadcast_state, graphdef = module.split(*filters)

  # infer length
  axis_sizes: tp.Set[int] = set()
  args_sizes = jax.tree_util.tree_map(
    lambda axis, node: jax.tree_util.tree_map(lambda x: x.shape[axis], node)
    if axis is not None
    else None,
    options.in_args_axes,
    args,
    is_leaf=lambda x: x is None,
  )
  kwargs_sizes = jax.tree_util.tree_map(
    lambda axis, node: jax.tree_util.tree_map(lambda x: x.shape[axis], node)
    if axis is not None
    else None,
    options.in_kwargs_axes,
    kwargs,
    is_leaf=lambda x: x is None,
  )
  axis_sizes.update(jax.tree_util.tree_leaves(args_sizes))
  axis_sizes.update(jax.tree_util.tree_leaves(kwargs_sizes))

  if len(axis_sizes) > 1:
    raise ValueError(
      'Inconsistent lengths between variable_axes states and '
      f'arguments: {axis_sizes}'
    )
  elif len(axis_sizes) == 0:
    if options.axis_size is None:
      raise ValueError(
        'Cannot infer length from variable_axes states or axes_arg, '
        'please specify `length`'
      )
    axis_size = options.axis_size
  else:
    axis_size = axis_sizes.pop()
    if options.axis_size is not None and options.axis_size != axis_size:
      raise ValueError(
        f'Specified axis_size {options.axis_size} is not the same as the'
        f' inferred length {axis_size}'
      )

  # split rng state
  if rngs is not None:
    if not isinstance(rngs, rnglib.Rngs):
      raise TypeError(f'Expected a Rngs, got {type(rngs).__name__}')

    forked_rngs = rngs.fork({filterlib.Not(options.broadcast_rngs): axis_size})
    split_keys, broadcast_keys = forked_rngs.splits, forked_rngs.broadcasts
  else:
    split_keys = None
    broadcast_keys = None

  moduledef_out: tp.Optional[GraphDef[Module]] = None

  keys_axes = 0
  states_axes = list(options.variable_axes.values())
  args_axes = options.in_args_axes
  kwargs_axes = options.in_kwargs_axes
  out_axes = options.out_axes

  @functools.partial(
    jax.vmap,
    in_axes=(keys_axes, states_axes, args_axes, kwargs_axes),
    out_axes=(None, states_axes, out_axes),
    axis_name=options.axis_name,
    axis_size=axis_size,
    spmd_axis_name=options.spmd_axis_name,
  )
  def vmap_fn(
    split_keys: dict[str, rnglib.RngStream] | None,
    vectorized_states: list[State],
    args: tuple[tp.Any, ...],
    kwargs: dict[str, tp.Any],
  ):
    nonlocal moduledef_out

    # merge rng state
    if split_keys is not None:
      assert broadcast_keys is not None
      kwargs['rngs'] = rnglib.Rngs(**split_keys, **broadcast_keys)

    # remove metadata axis name from Variable.sharding
    if spmd.PARTITION_NAME in options.vmap_metadata:
      vectorized_states = [
        spmd.remove_axis(state, index, options.vmap_metadata)
        for state, index in zip(
          vectorized_states, options.variable_axes.values()
        )
      ]

    # merge module state
    module = graphdef.merge(*vectorized_states, broadcast_state)

    output = f(module, *args, **kwargs)

    # split module state
    *vectorized_states_out, broadcast_state_out, moduledef_out = module.split(
      *filters
    )

    # add metadata axis name to Variable.sharding
    if spmd.PARTITION_NAME in options.vmap_metadata:
      vectorized_states_out = [
        spmd.add_axis(state, index, options.vmap_metadata)
        for state, index in zip(
          vectorized_states_out, options.variable_axes.values()
        )
      ]

    return broadcast_state_out, vectorized_states_out, output

  broadcast_state, vectorized_states, output = vmap_fn(
    split_keys, vectorized_states, args, kwargs
  )
  assert moduledef_out is not None

  module.update(((*vectorized_states, broadcast_state), moduledef_out))

  return output


def vmap(
  f: F,
  *,
  variable_axes: tp.Mapping[filterlib.Filter, int] = MappingProxyType({}),
  broadcast_rngs: filterlib.Filter = None,
  in_args_axes: tp.Any = 0,
  in_kwargs_axes: tp.Any = 0,
  out_axes: tp.Any = 0,
  axis_size: int | None = None,
  axis_name: str | None = None,
  spmd_axis_name: str | None = None,
  vmap_metadata: tp.Mapping[str, tp.Any] = MappingProxyType({}),
  is_init: tp.Optional[bool] = None,
) -> F:
  if is_init is None:
    is_init = f.__name__ == '__init__'

  options = VmapOptions(
    variable_axes=variable_axes,
    broadcast_rngs=broadcast_rngs,
    in_args_axes=in_args_axes,
    in_kwargs_axes=in_kwargs_axes,
    out_axes=out_axes,
    axis_size=axis_size,
    axis_name=axis_name,
    spmd_axis_name=spmd_axis_name,
    vmap_metadata=vmap_metadata,
  )

  if is_init:

    @functools.wraps(f)
    def vmap_init_wrapper(module: Module, *args, **kwargs):
      def module_constructor(*args, **kwargs):
        _check_args(args)
        f(module, *args, **kwargs)
        return module

      lifted_module = vmap_init(options, module_constructor, args, kwargs)
      module.update(lifted_module)

    wrapper = vmap_init_wrapper

  else:

    @functools.wraps(f)
    def vmap_apply_wrapper(module: Module, *args, **kwargs) -> tp.Any:
      _check_args(args)
      return vmap_apply(options, f, module, args, kwargs)

    wrapper = vmap_apply_wrapper

  return wrapper  # type: ignore
