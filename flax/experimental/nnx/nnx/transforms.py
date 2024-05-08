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

from flax.core.frozen_dict import FrozenDict

import jax
import jax.core
import jax.numpy as jnp
import jax.stages

from jax._src.tree_util import broadcast_prefix
from flax.experimental.nnx.nnx import (
  filterlib,
  graph,
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
StrInt = tp.TypeVar('StrInt', str, int)
AxisName = tp.Hashable
Leaves = tp.List[Leaf]
Index = int

def _normalize_sequence(
  x: StrInt | tp.Iterable[StrInt] | None, /
) -> tuple[StrInt, ...]:
  if x is None:
    return ()
  elif isinstance(x, (str, int)):
    return (x,)  # type: ignore
  else:
    return tuple(x)


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

@dataclasses.dataclass(frozen=True)
class JitStaticInputs:
  graphdef: GraphDef[tuple[tp.Any, ...]]
  ctx: graph.UpdateContext


jax.tree_util.register_static(JitStaticInputs)


@dataclasses.dataclass(frozen=True)
class JitStaticOutputs:
  graphdef: GraphDef[tuple[tp.Any, ...]]
  index_mapping: dict[Index, Index]


jax.tree_util.register_static(JitStaticOutputs)

def _default_constrain_object_state(state: State) -> State:
  state_spec = spmd.get_partition_spec(state)
  state = jax.lax.with_sharding_constraint(state, state_spec)
  return state


@dataclasses.dataclass
class JITOptions:
  in_shardings: tp.Any
  out_shardings: tp.Any
  static_argnums: tuple[int, ...]
  static_argnames: tuple[str, ...]
  donate_argnums: tuple[int, ...]
  donate_argnames: tuple[str, ...]
  keep_unused: bool
  device: tp.Optional[jax.Device]
  backend: tp.Optional[str]
  inline: bool
  abstracted_axes: tp.Optional[tp.Any]
  # nnx specific
  donate_state: bool
  constrain_state: tp.Callable[[State], State] | None

  @classmethod
  def from_jit_kwargs(
    cls,
    in_shardings: tp.Any,
    out_shardings: tp.Any,
    static_argnums: int | tp.Sequence[int] | None,
    static_argnames: str | tp.Iterable[str] | None,
    donate_argnums: int | tp.Sequence[int] | None,
    donate_argnames: str | tp.Iterable[str] | None,
    keep_unused: bool,
    device: tp.Optional[jax.Device],
    backend: tp.Optional[str],
    inline: bool,
    abstracted_axes: tp.Optional[tp.Any],
    donate_state: bool,
    constrain_state: bool | tp.Callable[[State], State],
  ):
    _static_argnums = _normalize_sequence(static_argnums)
    _static_argnames = _normalize_sequence(static_argnames)
    _donate_argnums = _normalize_sequence(donate_argnums)
    _donate_argnames = _normalize_sequence(donate_argnames)

    if donate_state:
      _donate_argnames = (*_donate_argnames, '_nnx_jit_state')

    if callable(constrain_state):
      _constrain_object_state = constrain_state
    elif constrain_state:
      _constrain_object_state = _default_constrain_object_state
    else:
      _constrain_object_state = None

    return cls(
      in_shardings=in_shardings,
      out_shardings=out_shardings,
      static_argnums=_static_argnums,
      static_argnames=_static_argnames,
      donate_argnums=_donate_argnums,
      donate_argnames=_donate_argnames,
      keep_unused=keep_unused,
      device=device,
      backend=backend,
      inline=inline,
      abstracted_axes=abstracted_axes,
      donate_state=donate_state,
      constrain_state=_constrain_object_state,
    )

  def get_jit_kwargs(self) -> dict[str, tp.Any]:
    kwargs = vars(self).copy()
    del kwargs['donate_state']
    del kwargs['constrain_state']
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
    static_argnums: int | tp.Sequence[int] | None = None,
    static_argnames: str | tp.Iterable[str] | None = None,
    donate_argnums: int | tp.Sequence[int] | None = None,
    donate_argnames: str | tp.Iterable[str] | None = None,
    keep_unused: bool = False,
    device: tp.Optional[jax.Device] = None,
    backend: tp.Optional[str] = None,
    inline: bool = False,
    abstracted_axes: tp.Optional[tp.Any] = None,
    # nnx specific
    donate_state: bool = False,
    constrain_state: bool | tp.Callable[[State], State] = False,
  ) -> tp.Callable[..., 'Jit[M]']:
    super_call = super().__call__

    def _create_jit(*args, **kwargs) -> Jit[M]:
      return super_call(
        module_constructor=module_constructor,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        donate_argnames=donate_argnames,
        keep_unused=keep_unused,
        device=device,
        backend=backend,
        inline=inline,
        abstracted_axes=abstracted_axes,
        # nnx specific
        donate_state=donate_state,
        constrain_state=constrain_state,
        # submodule args
        module_init_args=args,
        module_init_kwargs=kwargs,
      )

    return _create_jit


class JittedFn(tp.Protocol):
  def __call__(
    self,
    *args: tp.Any,
    _nnx_jit_static: JitStaticInputs,
    _nnx_jit_state: State,
    **kwargs: tp.Any,
  ) -> tuple[
    tp.Any, State, GraphDef[tuple[tuple[tp.Any, ...], tuple[tp.Any, ...]]]
  ]:
    ...


def get_jitted_fn(f, options: JITOptions) -> JittedFn:
  jit_kwargs = options.get_jit_kwargs()

  @functools.partial(jax.jit, **jit_kwargs)
  def jitted_fn(
    *args: tp.Any,
    _nnx_jit_static: JitStaticInputs,
    _nnx_jit_state: State,
    **kwargs: tp.Any,
  ) -> tuple[tp.Any, State, GraphDef[tuple[tp.Any, ...]]]:
    ctx = _nnx_jit_static.ctx
    graphdef = _nnx_jit_static.graphdef
    state: State = _nnx_jit_state

    if options.constrain_state is not None:
      state = options.constrain_state(state)

    input_graph_nodes = ctx.merge(graphdef, state)

    (args, kwargs) = graph.insert_graph_nodes((args, kwargs), input_graph_nodes)

    out = f(*args, **kwargs)

    out, output_graph_nodes = graph.extract_graph_nodes(out)

    graphdef, state = ctx.split((input_graph_nodes, output_graph_nodes))

    if options.constrain_state is not None:
      state = options.constrain_state(state)

    return out, state, graphdef

  return jitted_fn


def jit_apply(
  options: JITOptions,
  jitted_fn: JittedFn,
  args: tuple[tp.Any, ...],
  kwargs: dict[str, tp.Any],
) -> tp.Any:
  ctx = graph.UpdateContext()
  (args, kwargs), input_graph_nodes = graph.extract_graph_nodes((args, kwargs))
  graphdef, state = ctx.split(input_graph_nodes)

  out, output_state, output_graphdef = jitted_fn(
    *args,
    _nnx_jit_static=JitStaticInputs(graphdef, ctx),
    _nnx_jit_state=state,
    **kwargs,
  )
  input_graph_nodes, output_graph_nodes = ctx.update(
    output_graphdef, output_state
  )
  out = graph.insert_graph_nodes(out, output_graph_nodes)
  return out


class Jit(LiftedModule[M], metaclass=JITMeta):
  def __init__(
    self,
    module_constructor: tp.Callable[..., M],
    *,
    in_shardings: tp.Any = UNSPECIFIED,
    out_shardings: tp.Any = UNSPECIFIED,
    static_argnums: int | tp.Sequence[int] | None = None,
    static_argnames: str | tp.Iterable[str] | None = None,
    donate_argnums: int | tp.Sequence[int] | None = None,
    donate_argnames: str | tp.Iterable[str] | None = None,
    keep_unused: bool = False,
    device: tp.Optional[jax.Device] = None,
    backend: tp.Optional[str] = None,
    inline: bool = False,
    abstracted_axes: tp.Optional[tp.Any] = None,
    # nnx specific
    donate_state: bool = False,
    constrain_state: bool | tp.Callable[[State], State] = False,
    # submodule args
    module_init_args: tuple[tp.Any, ...],
    module_init_kwargs: dict[str, tp.Any],
  ):
    self.options = JITOptions.from_jit_kwargs(
      in_shardings=in_shardings,
      out_shardings=out_shardings,
      static_argnums=static_argnums,
      static_argnames=static_argnames,
      donate_argnums=donate_argnums,
      donate_argnames=donate_argnames,
      keep_unused=keep_unused,
      device=device,
      backend=backend,
      inline=inline,
      abstracted_axes=abstracted_axes,
      donate_state=donate_state,
      constrain_state=constrain_state,
    )
    self.accessor: tp.Optional[DelayedAccessor] = None

    def jit_call_module(module, *args, **kwargs):
      assert self.accessor is not None
      method = self.accessor(module)
      return method(*args, **kwargs)

    self.jitted_fn: JittedFn[M] = get_jitted_fn(jit_call_module, self.options)
    self.module_constructor = module_constructor
    self.jit_module = self.module_constructor(
      *module_init_args, **module_init_kwargs
    )

  @property
  def _submodule(self) -> M:
    return self.jit_module

  def _call(self, accessor: DelayedAccessor, *args, **kwargs) -> tp.Any:
    self.accessor = accessor
    try:
      out = jit_apply(
        self.options, self.jitted_fn, (self.jit_module, *args), kwargs
      )
    finally:
      self.accessor = None
    return out


def jit(
  fun: F,
  *,
  in_shardings: tp.Any = UNSPECIFIED,
  out_shardings: tp.Any = UNSPECIFIED,
  static_argnums: int | tp.Sequence[int] | None = None,
  static_argnames: str | tp.Iterable[str] | None = None,
  donate_argnums: int | tp.Sequence[int] | None = None,
  donate_argnames: str | tp.Iterable[str] | None = None,
  keep_unused: bool = False,
  device: tp.Optional[jax.Device] = None,
  backend: tp.Optional[str] = None,
  inline: bool = False,
  abstracted_axes: tp.Optional[tp.Any] = None,
  # nnx specific
  donate_state: bool = False,
  constrain_state: bool | tp.Callable[[State], State] = False,
) -> F:
  """
  Lifted version of ``jax.jit`` that can handle Modules / graph nodes as
  arguments.

  Args:
    fun: Function to be jitted. ``fun`` should be a pure function, as
      side-effects may only be executed once.

      The arguments and return value of ``fun`` should be arrays,
      scalars, or (nested) standard Python containers (tuple/list/dict) thereof.
      Positional arguments indicated by ``static_argnums`` can be anything at
      all, provided they are hashable and have an equality operation defined.
      Static arguments are included as part of a compilation cache key, which is
      why hash and equality operators must be defined.

      JAX keeps a weak reference to ``fun`` for use as a compilation cache key,
      so the object ``fun`` must be weakly-referenceable. Most :class:`Callable`
      objects will already satisfy this requirement.
    in_shardings: Pytree of structure matching that of arguments to ``fun``,
      with all actual arguments replaced by resource assignment specifications.
      It is also valid to specify a pytree prefix (e.g. one value in place of a
      whole subtree), in which case the leaves get broadcast to all values in
      that subtree.

      The ``in_shardings`` argument is optional. JAX will infer the shardings
      from the input :py:class:`jax.Array`'s and defaults to replicating the input
      if the sharding cannot be inferred.

      The valid resource assignment specifications are:
        - :py:class:`XLACompatibleSharding`, which will decide how the value
            will be partitioned. With this, using a mesh context manager is not
            required.
        - :py:obj:`None`, will give JAX the freedom to choose whatever sharding
          it wants.
          For in_shardings, JAX will mark is as replicated but this behavior
          can change in the future.
          For out_shardings, we will rely on the XLA GSPMD partitioner to
          determine the output shardings.

      The size of every dimension has to be a multiple of the total number of
      resources assigned to it. This is similar to pjit's in_shardings.
    out_shardings: Like ``in_shardings``, but specifies resource
      assignment for function outputs. This is similar to pjit's
      out_shardings.

      The ``out_shardings`` argument is optional. If not specified, :py:func:`jax.jit`
      will use GSPMD's sharding propagation to figure out what the sharding of the
      output(s) should be.
    static_argnums: An optional int or collection of ints that specify which
      positional arguments to treat as static (compile-time constant).
      Operations that only depend on static arguments will be constant-folded in
      Python (during tracing), and so the corresponding argument values can be
      any Python object.

      Static arguments should be hashable, meaning both ``__hash__`` and
      ``__eq__`` are implemented, and immutable. Calling the jitted function
      with different values for these constants will trigger recompilation.
      Arguments that are not arrays or containers thereof must be marked as
      static.

      If neither ``static_argnums`` nor ``static_argnames`` is provided, no
      arguments are treated as static. If ``static_argnums`` is not provided but
      ``static_argnames`` is, or vice versa, JAX uses
      :code:`inspect.signature(fun)` to find any positional arguments that
      correspond to ``static_argnames``
      (or vice versa). If both ``static_argnums`` and ``static_argnames`` are
      provided, ``inspect.signature`` is not used, and only actual
      parameters listed in either ``static_argnums`` or ``static_argnames`` will
      be treated as static.
    static_argnames: An optional string or collection of strings specifying
      which named arguments to treat as static (compile-time constant). See the
      comment on ``static_argnums`` for details. If not
      provided but ``static_argnums`` is set, the default is based on calling
      ``inspect.signature(fun)`` to find corresponding named arguments.
    donate_argnums: Specify which positional argument buffers are "donated" to
      the computation. It is safe to donate argument buffers if you no longer
      need them once the computation has finished. In some cases XLA can make
      use of donated buffers to reduce the amount of memory needed to perform a
      computation, for example recycling one of your input buffers to store a
      result. You should not reuse buffers that you donate to a computation, JAX
      will raise an error if you try to. By default, no argument buffers are
      donated.

      If neither ``donate_argnums`` nor ``donate_argnames`` is provided, no
      arguments are donated. If ``donate_argnums`` is not provided but
      ``donate_argnames`` is, or vice versa, JAX uses
      :code:`inspect.signature(fun)` to find any positional arguments that
      correspond to ``donate_argnames``
      (or vice versa). If both ``donate_argnums`` and ``donate_argnames`` are
      provided, ``inspect.signature`` is not used, and only actual
      parameters listed in either ``donate_argnums`` or ``donate_argnames`` will
      be donated.

      For more details on buffer donation see the
      `FAQ <https://jax.readthedocs.io/en/latest/faq.html#buffer-donation>`_.
    donate_argnames: An optional string or collection of strings specifying
      which named arguments are donated to the computation. See the
      comment on ``donate_argnums`` for details. If not
      provided but ``donate_argnums`` is set, the default is based on calling
      ``inspect.signature(fun)`` to find corresponding named arguments.
    keep_unused: If `False` (the default), arguments that JAX determines to be
      unused by `fun` *may* be dropped from resulting compiled XLA executables.
      Such arguments will not be transferred to the device nor provided to the
      underlying executable. If `True`, unused arguments will not be pruned.
    device: This is an experimental feature and the API is likely to change.
      Optional, the Device the jitted function will run on. (Available devices
      can be retrieved via :py:func:`jax.devices`.) The default is inherited
      from XLA's DeviceAssignment logic and is usually to use
      ``jax.devices()[0]``.
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the XLA backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.
    inline: Specify whether this function should be inlined into enclosing
      jaxprs (rather than being represented as an application of the xla_call
      primitive with its own subjaxpr). Default False.
    donate_state: Optional, bool. If True, the object state of the
      graph node's state will be donated to the computation. Default False.
    constrain_state: Optional, bool or callable. If True, the object
      state of the graph node's state will be constrained to the partition
      specified by the graph node's partition spec as computed by
      :func:`nnx.spmd.get_partition_spec`. If a callable, the object State will
      passed to the callable which must return the constrained object State. If
      False, the object state will not be constrained. Default False.

  Returns:
    A wrapped version of ``fun``, set up for just-in-time compilation.
  """
  options = JITOptions.from_jit_kwargs(
    in_shardings=in_shardings,
    out_shardings=out_shardings,
    static_argnums=static_argnums,
    static_argnames=static_argnames,
    donate_argnums=donate_argnums,
    donate_argnames=donate_argnames,
    keep_unused=keep_unused,
    device=device,
    backend=backend,
    inline=inline,
    abstracted_axes=abstracted_axes,
    donate_state=donate_state,
    constrain_state=constrain_state,
  )
  jitted_fn = get_jitted_fn(fun, options)

  @functools.wraps(fun)
  def jit_apply_wrapper(*args, **kwargs):
    return jit_apply(options, jitted_fn, args, kwargs)

  wrapper = jit_apply_wrapper
  wrapper.inner = jitted_fn

  return wrapper  # type: ignore


# -------------------------------
# grad
# -------------------------------


@dataclasses.dataclass
class GradOptions:
  argnums: tuple[int, ...]
  has_aux: bool
  holomorphic: bool
  allow_int: bool
  reduce_axes: tp.Sequence[AxisName]
  return_value: bool
  wrt: filterlib.Filter


class GradMeta(ModuleMeta):
  def __call__(
    self,
    module_constructor: tp.Callable[..., M],
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: tp.Sequence[AxisName] = (),
    return_value: bool = False,
    *,
    wrt: filterlib.Filter = variables.Param,
  ) -> tp.Callable[..., 'Grad[M]']:
    super_call = super().__call__

    def _create_grad(*args, **kwargs) -> Grad[M]:
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
    argnums: int | tp.Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: tp.Sequence[AxisName] = (),
    return_value: bool = False,
    *,
    wrt: filterlib.Filter = variables.Param,
    # submodule args
    module_init_args: tuple[tp.Any, ...],
    module_init_kwargs: dict[str, tp.Any],
  ):
    _argnums = _normalize_sequence(argnums)
    self.options = GradOptions(
      argnums=_argnums,
      has_aux=has_aux,
      holomorphic=holomorphic,
      allow_int=allow_int,
      reduce_axes=reduce_axes,
      return_value=return_value,
      wrt=wrt,
    )
    self.module_constructor = module_constructor
    self.grad_module = self.module_constructor(
      *module_init_args, **module_init_kwargs
    )

  @property
  def _submodule(self) -> M:
    return self.grad_module

  def _call(self, accessor: DelayedAccessor, *args, **kwargs) -> tp.Any:
    def grad_call_apply(module, *args, **kwargs):
      method = accessor(module)
      return method(*args, **kwargs)

    return grad_apply(self.options, grad_call_apply, (self.grad_module, *args))


def grad_apply(options: GradOptions, f, args: tuple[tp.Any, ...]):
  _, input_nodes = graph.extract_graph_nodes(args)

  _args = list(args)
  diff_graph_nodes: dict[int, tp.Any] = {
    i: arg
    for i, arg in enumerate(args)
    if i in options.argnums and graph.is_node(arg)
  }

  _, diff_state, _ = graph.split(diff_graph_nodes, options.wrt, ...)
  for i in diff_graph_nodes:
    _args[i] = diff_state[i]

  transform = jax.value_and_grad if options.return_value else jax.grad
  out_nodes = None

  argnums = options.argnums[0] if len(options.argnums) == 1 else options.argnums

  @functools.partial(
    transform,
    argnums=argnums,
    has_aux=True,
    holomorphic=options.holomorphic,
    allow_int=options.allow_int,
    reduce_axes=options.reduce_axes,
  )
  def grad_fn(*args):
    nonlocal out_nodes

    _args = list(args)
    for i, graph_node in diff_graph_nodes.items():
      diff_state: State = _args[i]
      graph.update(graph_node, diff_state)
      _args[i] = graph_node

    out = f(*_args)
    out, out_nodes = graph.extract_graph_nodes(out)

    _, updates, _ = graph.flatten((input_nodes, out_nodes))

    if options.has_aux:
      loss, aux = out
      out = (loss, (updates, aux))
    else:
      out = (out, updates)

    return out

  out = grad_fn(*_args)

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

  graph.update((input_nodes, out_nodes), updates)
  return out


def grad(
  f: tp.Callable[..., tp.Any],
  argnums: int | tp.Sequence[int] = 0,
  has_aux: bool = False,
  holomorphic: bool = False,
  allow_int: bool = False,
  reduce_axes: tp.Sequence[AxisName] = (),
  *,
  wrt: filterlib.Filter = variables.Param,
) -> tp.Callable[..., tp.Any]:
  """Lifted version of ``jax.grad`` that can handle Modules / graph nodes as
  arguments.

  The differentiable state of each graph node is defined by the `wrt` filter,
  which by default is set to `nnx.Param`. Internally the ``State`` of
  graph nodes is extracted, filtered according to `wrt` filter, and
  passed to the underlying ``jax.grad`` function. The gradients
  of graph nodes are of type ``State``.

  Example::

    >>> from flax.experimental import nnx
    ...
    >>> m = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> x = jnp.ones((1, 2))
    >>> y = jnp.ones((1, 3))
    ...
    >>> loss_fn = lambda m, x, y: jnp.mean((m(x) - y) ** 2)
    >>> grad_fn = nnx.grad(loss_fn, wrt=nnx.Param)
    ...
    >>> grads = grad_fn(m, x, y)
    >>> jax.tree_util.tree_map(jnp.shape, grads)
    State({
      'bias': VariableState(
        type=Param,
        value=(3,)
      ),
      'kernel': VariableState(
        type=Param,
        value=(2, 3)
      )
    })

  Args:
    fun: Function to be differentiated. Its arguments at positions specified by
      ``argnums`` should be arrays, scalars, graph nodes or standard Python
      containers. Argument arrays in the positions specified by ``argnums`` must
      be of inexact (i.e., floating-point or complex) type. It should return a
      scalar (which includes arrays with shape ``()`` but not arrays with shape
      ``(1,)`` etc.)
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default 0).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. If True, inputs and outputs must be complex. Default False.
    allow_int: Optional, bool. Whether to allow differentiating with
      respect to integer valued inputs. The gradient of an integer input will
      have a trivial vector-space dtype (float0). Default False.
    reduce_axes: Optional, tuple of axis names. If an axis is listed here, and
      ``fun`` implicitly broadcasts a value over that axis, the backward pass
      will perform a ``psum`` of the corresponding gradient. Otherwise, the
      gradient will be per-example over named axes. For example, if ``'batch'``
      is a named batch axis, ``grad(f, reduce_axes=('batch',))`` will create a
      function that computes the total gradient while ``grad(f)`` will create
      one that computes the per-example gradient.
    wrt: Optional, filterlib.Filter. Filter to extract the differentiable state
      of each graph node. Default is `nnx.Param`.

  """

  if f.__name__ == '__init__':
    raise ValueError('Cannot use `grad` with `__init__`')

  _argnums = _normalize_sequence(argnums)
  options = GradOptions(
    argnums=_argnums,
    wrt=wrt,
    has_aux=has_aux,
    holomorphic=holomorphic,
    allow_int=allow_int,
    reduce_axes=reduce_axes,
    return_value=False,
  )

  @functools.wraps(f)
  def grad_wrapper(*args):
    return grad_apply(options, f, args)

  return grad_wrapper  # type: ignore




def value_and_grad(
  f: tp.Callable[..., tp.Any],
  argnums: int | tp.Sequence[int] = 0,
  has_aux: bool = False,
  holomorphic: bool = False,
  allow_int: bool = False,
  reduce_axes: tp.Sequence[AxisName] = (),
  *,
  wrt: filterlib.Filter = variables.Param,
) -> tp.Callable[..., tp.Any]:
  if f.__name__ == '__init__':
    raise ValueError('Cannot use `value_and_grad` with `__init__`')

  _argnums = _normalize_sequence(argnums)
  options = GradOptions(
    argnums=_argnums,
    has_aux=has_aux,
    holomorphic=holomorphic,
    allow_int=allow_int,
    reduce_axes=reduce_axes,
    return_value=True,
    wrt=wrt,
  )

  @functools.wraps(f)
  def value_and_grad_wrapper(*args):
    return grad_apply(options, f, args)

  return value_and_grad_wrapper  # type: ignore


# -------------------------------
# scan
# -------------------------------

@dataclasses.dataclass
class ScanOptions:
  length: int | None
  reverse: bool
  unroll: int | bool
  _split_transpose: bool
  # extended api
  in_axes: tp.Any
  in_axes_kwargs: tp.Any
  out_axes: tp.Any
  carry_argnum: int
  # nnx specific
  state_axes: tp.Mapping[filterlib.Filter, int]
  split_rngs: filterlib.Filter
  transform_metadata: tp.Mapping[str, tp.Any]
  scan_output: bool


class ScanMeta(ModuleMeta):
  def __call__(
    self,
    module_constructor: tp.Callable[..., M],
    *,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
    # extended api
    in_axes: int | None | tp.Sequence[tp.Any] = 0,
    in_axes_kwargs: tp.Any = 0,
    out_axes: tp.Any = 0,
    carry_argnum: int = 1,
    # nnx specific
    state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
    split_rngs: filterlib.Filter = ...,
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
    scan_output: bool = True,
  ) -> tp.Callable[..., 'Scan[M]']:
    super_call = super().__call__

    def _create_scan(*args, **kwargs) -> Scan[M]:
      return super_call(
        module_constructor=module_constructor,
        module_init_args=args,
        module_init_kwargs=kwargs,
        # base api
        length=length,
        reverse=reverse,
        unroll=unroll,
        _split_transpose=_split_transpose,
        # extended api
        in_axes=in_axes,
        in_axes_kwargs=in_axes_kwargs,
        out_axes=out_axes,
        carry_argnum=carry_argnum,
        # nnx specific
        state_axes=state_axes,
        split_rngs=split_rngs,
        transform_metadata=transform_metadata,
        scan_output=scan_output,
      )

    return _create_scan


class Scan(LiftedModule[M], metaclass=ScanMeta):
  def __init__(
    self,
    module_constructor: tp.Callable[..., M],
    *,
    length: int | None = None,
    reverse: bool = False,
    unroll: int | bool = 1,
    _split_transpose: bool = False,
    # extended api
    in_axes: int | None | tp.Sequence[tp.Any] = 0,
    in_axes_kwargs: tp.Any = 0,
    out_axes: tp.Any = 0,
    carry_argnum: int = 1,
    # nnx specific
    state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
    split_rngs: filterlib.Filter = ...,
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
    scan_output: bool = True,
    # submodule args
    module_init_args: tuple[tp.Any, ...],
    module_init_kwargs: dict[str, tp.Any],
  ):
    self.module_constructor = module_constructor
    self.options = ScanOptions(
      length=length,
      reverse=reverse,
      unroll=unroll,
      _split_transpose=_split_transpose,
      # extended api
      in_axes=in_axes,
      in_axes_kwargs=in_axes_kwargs,
      out_axes=out_axes,
      carry_argnum=carry_argnum,
      # nnx specific
      state_axes=state_axes,
      split_rngs=split_rngs,
      transform_metadata=transform_metadata,
      scan_output=scan_output,
    )
    # use Vmap to handle initialisation
    vmapped_module = Vmap(
      module_constructor,
      in_axes=in_axes,
      out_axes=None,
      axis_name=None,
      axis_size=length,
      spmd_axis_name=None,
      state_axes=state_axes,
      split_rngs=split_rngs,
      in_axes_kwargs=in_axes_kwargs,
      transform_metadata=transform_metadata,
    )(*module_init_args, **module_init_kwargs)

    self.scan_module = vmapped_module.vmap_module

  @property
  def _submodule(self) -> M:
    return self.scan_module

  def _call(
    self, accessor: DelayedAccessor, *args, **kwargs
  ) -> tuple[tp.Any, tp.Any]:
    def scan_call_apply(module, *args, **kwargs):
      method = accessor(module)
      return method(*args, **kwargs)

    return scan_apply(
      self.options,
      scan_call_apply,
      (self._submodule, *args),
      kwargs,
    )


def scan_apply(
  options: ScanOptions,
  f: tp.Callable[..., tuple[C, B] | C],
  args: tuple[tp.Any, ...],
  kwargs: dict[str, tp.Any],
) -> tuple[C, B] | C:
  # extract nodes
  (args, kwargs), input_graph_nodes = graph.extract_graph_nodes((args, kwargs))
  input_rng_streams = rnglib.backup_keys(input_graph_nodes)

  # extract carry arg
  carry_arg, args = _extract_carry_arg(args, options.carry_argnum)

  ctx = graph.UpdateContext()
  # split module state
  filters = (*options.state_axes.keys(), ...)
  graphdef, rng_state, *scan_states, carry_state = ctx.split(
    input_graph_nodes, rnglib.RngState, *filters
  )

  # transpose axes arg
  flatdef, flat_scan, flat_carry = _transpose_and_split(
    (args, kwargs, scan_states),
    (
      options.in_axes,
      options.in_axes_kwargs,
      list(options.state_axes.values()),
    ),
  )

  # infer length
  lengths: set[int] = set(
    x.shape[axis]  # type: ignore
    for x, axis in zip(flat_scan, flatdef.flat_axes)
    if axis is not None
  )

  if len(lengths) > 1:
    raise ValueError(
      'Inconsistent lengths between state_axes states and '
      f'arguments: {lengths}'
    )
  elif len(lengths) == 0:
    if options.length is None:
      raise ValueError(
        'Cannot infer length from state_axes states or axes_arg, '
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
  split_keys, carry_keys = rnglib.fork(
    rng_state,
    options.split_rngs,
    length,
  )

  def scan_fn(
    carry: tuple[
      State,  # carry_keys
      State,  # carry_state
      tp.Any,  # carry_arg
    ],
    scan: tuple[
      State,  # split_keys
      list[jax.Array | None],  # flat_scan
    ],
  ):
    carry_keys, carry_state, carry_arg = carry
    split_keys, flat_scan = scan

    # merge args and kwargs
    args, kwargs, scan_states = _unflatten_splits(
      flatdef, flat_scan, flat_carry
    )
    # remove metadata axis name from Variable.sharding
    if spmd.PARTITION_NAME in options.transform_metadata:
      scan_states = [
        spmd.remove_axis(state, index, options.transform_metadata)
        for state, index in zip(scan_states, options.state_axes.values())
      ]

    # insert carry arg
    args = _insert_carry_arg(args, options.carry_argnum, carry_arg)

    # merge module state
    input_graph_nodes = ctx.merge(
      graphdef, *scan_states, carry_state, split_keys, carry_keys
    )
    (args, kwargs) = graph.insert_graph_nodes((args, kwargs), input_graph_nodes)

    out = f(*args, **kwargs)

    if options.scan_output:
      if not isinstance(out, tuple) or len(out) != 2:
        raise ValueError(
          'Expected a tuple of length 2 as the output of the scan function, '
          f'got {out}'
        )
      out = tp.cast(tuple[C, B], out)
      carry_arg_out, scan_args_out = out
    else:
      out = tp.cast(C, out)
      carry_arg_out = out
      scan_args_out = None

    (
      (carry_arg_out, scan_args_out),
      output_graph_nodes,
    ) = graph.extract_graph_nodes((carry_arg_out, scan_args_out))

    # split module state
    (
      graphdef_out,
      rng_state_out,
      *scan_states_out,
      carry_state_out,
    ) = ctx.split(
      (input_graph_nodes, output_graph_nodes),
      rnglib.RngState,
      *filters,
    )

    not_keys_out, split_keys_out, carry_keys_out = rng_state_out.split(
      rnglib.NotKey, options.split_rngs, ...
    )
    carry_keys_out = State.merge(not_keys_out, carry_keys_out)

    if 1 in carry_state_out:
      raise ValueError(
        f'Cannot add new carry state during scan, got {carry_state_out[1]}'
      )
    if 0 in carry_state_out:
      carry_state_out = carry_state_out[0]
      assert isinstance(carry_state_out, State)
    if 1 in carry_keys_out:
      raise ValueError(
        f'Cannot add new carry keys during scan, got {carry_keys_out[1]}'
      )
    if 0 in carry_keys_out:
      carry_keys_out = carry_keys_out[0]
      assert isinstance(carry_keys_out, State)

    # add metadata axis name to Variable.sharding
    if spmd.PARTITION_NAME in options.transform_metadata:
      scan_states_out = [
        spmd.add_axis(state, index, options.transform_metadata)
        for state, index in zip(scan_states_out, options.state_axes.values())
      ]

    carry_out = (carry_keys_out, carry_state_out, carry_arg_out)
    scan_out = (graphdef_out, scan_args_out, scan_states_out, split_keys_out)

    return carry_out, scan_out

  carry = (carry_keys, carry_state, carry_arg)
  scan = (split_keys, flat_scan)

  carry_out, scan_out = jax.lax.scan(
    scan_fn,
    carry,
    scan,
    length=length,
    reverse=options.reverse,
    unroll=options.unroll,
    _split_transpose=options._split_transpose,
  )
  carry_keys_out, carry_state_out, carry_arg_out = carry_out
  graphdef_out, scan_args_out, scan_states_out, split_keys_out = scan_out

  scan_args_out, scan_states_out = _transpose_tree(
    (scan_args_out, scan_states_out),
    (options.out_axes, list(options.state_axes.values())),
    axis_is_source=False,
  )

  if carry_state_out:
    carry_state_out = State({0: carry_state_out._mapping})
  if carry_keys_out:
    carry_keys_out = State({0: carry_keys_out._mapping})
  _, output_graph_nodes = ctx.update(
    graphdef_out,
    *scan_states_out,
    carry_state_out,
    carry_keys_out,
    split_keys_out,
  )

  carry_arg_out, scan_args_out = graph.insert_graph_nodes(
    (carry_arg_out, scan_args_out), output_graph_nodes
  )

  rnglib.restore_keys(input_rng_streams)

  if options.scan_output:
    scan_args_out = tp.cast(B, scan_args_out)
    return carry_arg_out, scan_args_out
  else:
    return carry_arg_out


@dataclasses.dataclass(frozen=True)
class FlatDef(tp.Generic[A]):
  type: type[A]
  treedef: jax.tree_util.PyTreeDef
  flat_axes: list[int | None]

jax.tree_util.register_static(FlatDef)

def _transpose_tree(tree: A, axes, /, *, axis_is_source: bool) -> A:
  flatdef, flat_transposes, _ = _transpose_and_split(
    tree, axes, allow_none=False, axis_is_source=axis_is_source
  )
  return flatdef.treedef.unflatten(flat_transposes)


def _transpose_and_split(
  tree: A, axes, /, *, allow_none: bool = True, axis_is_source: bool = True
) -> tuple[
  FlatDef[A],
  list[jax.Array | None],
  list[tp.Any],
]:
  flat_axes: list[int | None] = broadcast_prefix(
    axes, tree, is_leaf=lambda x: x is None
  )
  flat_tree, treedef = jax.tree.flatten(tree)

  flat_broadcasts: list[tp.Any] = []
  flat_transposes: list[jax.Array | None] = []

  for i, (axis, node) in enumerate(zip(flat_axes, flat_tree)):
    if axis is None:
      if not allow_none:
        raise ValueError('None axis not allowed')

      flat_broadcasts.append(node)
      flat_transposes.append(None)
    else:
      if not isinstance(node, jax.Array):
        raise TypeError(
          f'Expected a jax.Array, got {type(node).__name__} for axis {axis}'
        )
      # normalize axis
      if axis < 0:
        if axis < -len(node.shape):
          raise ValueError(
            f'Axis {axis} out of bounds for array with shape {node.shape}'
          )
        axis = len(node.shape) + axis
        flat_axes[i] = axis

      if axis_is_source:
        node = jnp.moveaxis(node, axis, 0)
      else:
        node = jnp.moveaxis(node, 0, axis)
      flat_broadcasts.append(None)
      flat_transposes.append(node)

  flatdef = FlatDef(type(tree), treedef, flat_axes)

  return flatdef, flat_transposes, flat_broadcasts

def _unflatten_splits(
  flatdef: FlatDef[A],
  flat_transposes: list[jax.Array | None],
  flat_broadcasts: list[tp.Any] | None = None,
  /,
  *,
  allow_none: bool = True,
) -> A:
  flat_axes = flatdef.flat_axes
  treedef = flatdef.treedef
  if flat_broadcasts is None:
    if allow_none:
      raise ValueError('flat_broadcasts must be provided if allow_none is True')
    flat_broadcasts = [None] * len(flat_axes)

  flat_tree = []
  for axis, transpose, broadcast in zip(
    flat_axes, flat_transposes, flat_broadcasts
  ):
    if axis is None:
      if not allow_none:
        raise ValueError('None axis not allowed')
      flat_tree.append(broadcast)
    else:
      if transpose is None:
        raise ValueError('None transpose not allowed')
      flat_tree.append(transpose)

  tree = treedef.unflatten(flat_tree)
  return tree


def _extract_carry_arg(
  args: tuple[tp.Any, ...], carry_argnum: int, /
) -> tuple[tp.Any, tuple[tp.Any, ...]]:
  # extract carry arg
  if len(args) < carry_argnum + 1:
    raise TypeError(
      f'Expected at least {carry_argnum + 1} positional arguments, '
      f'got {len(args)}'
    )

  args_ = list(args)
  carry_arg = args_[carry_argnum]
  args_[carry_argnum] = None
  args = tuple(args_)

  return carry_arg, args


def _insert_carry_arg(
  args: tuple[tp.Any, ...], carry_argnum: int, carry_arg: tp.Any, /
) -> tuple[tp.Any, ...]:
  args_ = list(args)
  args_[carry_argnum] = carry_arg
  args = tuple(args_)

  return args


def scan(
  f: F,
  *,
  length: int | None = None,
  reverse: bool = False,
  unroll: int | bool = 1,
  _split_transpose: bool = False,
  # extended api
  in_axes: int | None | tp.Sequence[tp.Any] = 0,
  in_axes_kwargs: tp.Any = 0,
  out_axes: tp.Any = 0,
  carry_argnum: int = 0,
  # nnx specific
  state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
  split_rngs: filterlib.Filter = ...,
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
  scan_output: bool = True,
) -> F:
  options = ScanOptions(
    length=length,
    reverse=reverse,
    unroll=unroll,
    _split_transpose=_split_transpose,
    in_axes=in_axes,
    in_axes_kwargs=in_axes_kwargs,
    out_axes=out_axes,
    carry_argnum=carry_argnum,
    state_axes=state_axes,
    split_rngs=split_rngs,
    transform_metadata=transform_metadata,
    scan_output=scan_output,
  )

  @functools.wraps(f)
  def scan_apply_wrapper(*args, **kwargs) -> C | tuple[C, tp.Any]:
    return scan_apply(options, f, args, kwargs)

  return scan_apply_wrapper  # type: ignore


# -------------------------------
# remat
# -------------------------------


class RematMeta(ModuleMeta):
  def __call__(
    self,
    module_constructor: tp.Callable[..., M],
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy: tp.Callable[..., bool] | None = None,
  ) -> tp.Callable[..., 'Remat[M]']:
    super_call = super().__call__

    def create_remat(*args, **kwargs) -> Remat[M]:
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
  static_argnums: int | tuple[int, ...]
  policy: tp.Callable[..., bool] | None

  def __post_init__(self):
    if isinstance(self.static_argnums, int):
      self.static_argnums = (self.static_argnums,)

    # add 1 as an offset to account for state parameter
    self.static_argnums = tuple(
      x + 1 if x >= 0 else x for x in self.static_argnums
    )


class Remat(LiftedModule[M], metaclass=RematMeta):
  def __init__(
    self,
    *,
    module_constructor: tp.Callable[..., M],
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy: tp.Callable[..., bool] | None = None,
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

  def _call(self, accessor: DelayedAccessor, *args) -> tp.Any:
    def remat_apply_call(module, *args):
      method = accessor(module)
      return method(*args)

    return remat_apply(
      self.options,
      remat_apply_call,
      (self.remat_module, *args),
    )


def remat_apply(
  options: RematOptions,
  f: tp.Callable[..., tp.Any],
  args: tuple[tp.Any, ...],
):
  ctx = graph.UpdateContext()
  args, input_nodes = graph.extract_graph_nodes(args)
  graphdef, state = ctx.split(input_nodes)

  def _remat_fn(state: State, *args):
    input_nodes = ctx.merge(graphdef, state)
    args = graph.insert_graph_nodes(args, input_nodes)
    out = f(*args)

    out, output_nodes = graph.extract_graph_nodes(out)
    new_graphdef, new_state = ctx.split((input_nodes, output_nodes))
    return (new_graphdef, new_state), out

  (new_graphdef, new_state), out = jax.checkpoint(
    _remat_fn,
    prevent_cse=options.prevent_cse,
    static_argnums=options.static_argnums,
    policy=options.policy,
  )(state, *args)

  _, output_nodes = ctx.update(new_graphdef, new_state)
  out = graph.insert_graph_nodes(out, output_nodes)

  return out


def remat(
  f: F,
  *,
  prevent_cse: bool = True,
  static_argnums: int | tuple[int, ...] = (),
  policy: tp.Callable[..., bool] | None = None,
) -> F:
  options = RematOptions(
    prevent_cse=prevent_cse,
    static_argnums=static_argnums,
    policy=policy,
  )

  @functools.wraps(f)
  def remat_wrapper(*args):
    return remat_apply(options, f, args)

  return remat_wrapper  # type: ignore


# -------------------------------
# vmap
# -------------------------------

@dataclasses.dataclass
class VmapOptions:
  in_axes: int | None | tp.Sequence[tp.Any]
  out_axes: tp.Any
  axis_name: AxisName | None
  axis_size: int | None
  spmd_axis_name: AxisName | tuple[AxisName, ...] | None
  # nnx specific
  state_axes: tp.Mapping[filterlib.Filter, int]
  split_rngs: filterlib.Filter
  in_axes_kwargs: tp.Any
  transform_metadata: tp.Mapping[str, tp.Any]


class VmapMeta(ModuleMeta):
  def __call__(
    self,
    module_constructor: tp.Callable[..., M],
    *,
    in_axes: int | None | tp.Sequence[tp.Any] = 0,
    out_axes: tp.Any = 0,
    axis_name: AxisName | None = None,
    axis_size: int | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
    # nnx specific
    in_axes_kwargs: tp.Any = 0,
    state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
    split_rngs: filterlib.Filter = ...,
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
  ) -> tp.Callable[..., 'Vmap[M]']:
    super_call = super().__call__

    def _create_vmap(*args, **kwargs) -> Scan[M]:
      return super_call(
        module_constructor=module_constructor,
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=axis_size,
        axis_name=axis_name,
        spmd_axis_name=spmd_axis_name,
        # nnx specific
        in_axes_kwargs=in_axes_kwargs,
        state_axes=state_axes,
        split_rngs=split_rngs,
        transform_metadata=transform_metadata,
        # submodule args
        module_init_args=args,
        module_init_kwargs=kwargs,
      )

    return _create_vmap


class Vmap(LiftedModule[M], metaclass=VmapMeta):
  def __init__(
    self,
    module_constructor: tp.Callable[..., M],
    *,
    in_axes: int | None | tp.Sequence[tp.Any] = 0,
    out_axes: tp.Any = 0,
    axis_name: AxisName | None = None,
    axis_size: int | None = None,
    spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
    # nnx specific
    in_axes_kwargs: tp.Any = 0,
    state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
    split_rngs: filterlib.Filter = ...,
    transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
    # submodule args
    module_init_args: tuple[tp.Any, ...],
    module_init_kwargs: dict[str, tp.Any],
  ):
    self.module_constructor = module_constructor
    self.options = VmapOptions(
      in_axes=in_axes,
      out_axes=out_axes,
      axis_name=axis_name,
      axis_size=axis_size,
      spmd_axis_name=spmd_axis_name,
      # nnx specific
      in_axes_kwargs=in_axes_kwargs,
      state_axes=state_axes,
      split_rngs=split_rngs,
      transform_metadata=transform_metadata,
    )

    (
      (module_init_args, module_init_kwargs),
      init_nodes,
    ) = graph.extract_graph_nodes((module_init_args, module_init_kwargs))

    def vmap_init(init_nodes):
      (args, kwargs) = graph.insert_graph_nodes(
        (module_init_args, module_init_kwargs), init_nodes
      )
      return module_constructor(*args, **kwargs)

    init_options = dataclasses.replace(
      self.options,
      in_axes=None,
      out_axes=None,
    )
    self.vmap_module = vmap_apply(init_options, vmap_init, (init_nodes,), {})

  @property
  def _submodule(self) -> M:
    return self.vmap_module

  def _call(self, accessor: DelayedAccessor, *args, **kwargs):
    def vmap_apply_call(module, *args, **kwargs):
      method = accessor(module)
      return method(*args, **kwargs)

    return vmap_apply(
      self.options,
      vmap_apply_call,
      (self._submodule, *args),
      kwargs,
    )

def vmap_apply(
  options: VmapOptions,
  f: tp.Callable[..., A],
  args: tuple[tp.Any, ...],
  kwargs: dict[str, tp.Any],
) -> A:
  (args, kwargs), input_graph_nodes = graph.extract_graph_nodes((args, kwargs))
  input_rng_streams = rnglib.backup_keys(input_graph_nodes)

  ctx = graph.UpdateContext()
  # split module state
  filters = (*options.state_axes.keys(), ...)
  graphdef, rng_state, *vectorized_states, broadcast_state = ctx.split(
    input_graph_nodes, rnglib.RngState, *filters
  )

  # infer length
  axis_sizes: tp.Set[int] = set()
  args_sizes = jax.tree_util.tree_map(
    lambda axis, node: jax.tree_util.tree_map(lambda x: x.shape[axis], node)
    if axis is not None
    else None,
    options.in_axes,
    args,
    is_leaf=lambda x: x is None,
  )
  kwargs_sizes = jax.tree_util.tree_map(
    lambda axis, node: jax.tree_util.tree_map(lambda x: x.shape[axis], node)
    if axis is not None
    else None,
    options.in_axes_kwargs,
    kwargs,
    is_leaf=lambda x: x is None,
  )
  axis_sizes.update(jax.tree_util.tree_leaves(args_sizes))
  axis_sizes.update(jax.tree_util.tree_leaves(kwargs_sizes))

  if len(axis_sizes) > 1:
    raise ValueError(
      'Inconsistent lengths between state_axes states and '
      f'arguments: {axis_sizes}'
    )
  elif len(axis_sizes) == 0:
    if options.axis_size is None:
      raise ValueError(
        'Cannot infer length from state_axes states or axes_arg, '
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

  split_keys, broadcast_keys = rnglib.fork(
    rng_state,
    options.split_rngs,
    axis_size,
  )

  keys_axes = 0
  states_axes = list(options.state_axes.values())
  args_axes = options.in_axes
  kwargs_axes = options.in_axes_kwargs
  out_axes = options.out_axes
  broadcast_state_axes = None
  graphdef_out_axes = None
  keys_axes_out = 0

  @functools.partial(
    jax.vmap,
    in_axes=(keys_axes, states_axes, args_axes, kwargs_axes),
    out_axes=(
      graphdef_out_axes,
      broadcast_state_axes,
      states_axes,
      keys_axes_out,
      out_axes,
    ),
    axis_name=options.axis_name,
    axis_size=axis_size,
    spmd_axis_name=options.spmd_axis_name,
  )
  def vmap_fn(
    split_keys: State,
    vectorized_states: list[State],
    args: tuple[tp.Any, ...],
    kwargs: dict[str, tp.Any],
  ):
    # remove metadata axis name from Variable.sharding
    if spmd.PARTITION_NAME in options.transform_metadata:
      vectorized_states = [
        spmd.remove_axis(state, index, options.transform_metadata)
        for state, index in zip(vectorized_states, options.state_axes.values())
      ]

    # merge module state
    input_graph_nodes = ctx.merge(
      graphdef, *vectorized_states, broadcast_state, split_keys, broadcast_keys
    )

    (args, kwargs) = graph.insert_graph_nodes((args, kwargs), input_graph_nodes)

    out = f(*args, **kwargs)

    out, output_graph_nodes = graph.extract_graph_nodes(out)

    # split module state
    (
      graphdef_out,
      rng_state_out,
      *vectorized_states_out,
      broadcast_state_out,
    ) = ctx.split(
      (input_graph_nodes, output_graph_nodes),
      rnglib.RngState,
      *filters,
    )

    not_keys_out, split_keys_out, broadcast_keys_out = rng_state_out.split(
      rnglib.NotKey, options.split_rngs, ...
    )

    broadcast_state_out = State.merge(
      broadcast_state_out, broadcast_keys_out, not_keys_out
    )

    # add metadata axis name to Variable.sharding
    if spmd.PARTITION_NAME in options.transform_metadata:
      vectorized_states_out = [
        spmd.add_axis(state, index, options.transform_metadata)
        for state, index in zip(
          vectorized_states_out, options.state_axes.values()
        )
      ]

    return (
      graphdef_out,
      broadcast_state_out,
      vectorized_states_out,
      split_keys_out,
      out,
    )

  (
    graphdef_out,
    broadcast_state,
    vectorized_states,
    split_keys_out,
    out,
  ) = vmap_fn(split_keys, vectorized_states, args, kwargs)

  _, output_graph_nodes = ctx.update(
    graphdef_out,
    *vectorized_states,
    broadcast_state,
    split_keys_out,
  )

  out = graph.insert_graph_nodes(out, output_graph_nodes)

  rnglib.restore_keys(input_rng_streams)

  return out


def vmap(
  f: F,
  *,
  in_axes: int | None | tp.Sequence[tp.Any] = 0,
  out_axes: tp.Any = 0,
  axis_name: AxisName | None = None,
  axis_size: int | None = None,
  spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None,
  # nnx specific
  in_axes_kwargs: tp.Any = 0,
  state_axes: tp.Mapping[filterlib.Filter, int] = FrozenDict({...: 0}),
  split_rngs: filterlib.Filter = ...,
  transform_metadata: tp.Mapping[str, tp.Any] = FrozenDict({}),
) -> F:
  options = VmapOptions(
    state_axes=state_axes,
    split_rngs=split_rngs,
    in_axes=in_axes,
    in_axes_kwargs=in_axes_kwargs,
    out_axes=out_axes,
    axis_size=axis_size,
    axis_name=axis_name,
    spmd_axis_name=spmd_axis_name,
    transform_metadata=transform_metadata,
  )

  @functools.wraps(f)
  def vmap_apply_wrapper(*args, **kwargs) -> tp.Any:
    return vmap_apply(options, f, args, kwargs)

  wrapper = vmap_apply_wrapper

  return wrapper  # type: ignore

# -------------------------------
# eval_shape
# -------------------------------


def eval_shape(
  f: tp.Callable[..., A],
  *args: tp.Any,
  **kwargs: tp.Any,
) -> A:
  (args, kwargs), input_nodes = graph.extract_graph_nodes((args, kwargs))
  graphdef, state = graph.split(input_nodes)

  @functools.wraps(f)
  def _eval_shape_fn(state: State, *args, **kwargs):
    input_nodes = graph.merge(graphdef, state)
    args, kwargs = graph.insert_graph_nodes((args, kwargs), input_nodes)
    out = f(*args, **kwargs)
    out, output_nodes = graph.extract_graph_nodes(out)
    graphdef_out, state_out = graph.split(output_nodes)
    return graphdef_out, state_out, out

  graphdef_out, state_out, out = jax.eval_shape(
    _eval_shape_fn, state, *args, **kwargs
  )

  output_nodes = graph.merge(graphdef_out, state_out)
  out = graph.insert_graph_nodes(out, output_nodes)
  return out