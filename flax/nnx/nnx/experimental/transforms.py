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
# pytype: skip-file
from __future__ import annotations

from abc import abstractmethod
import dataclasses
import functools
import typing as tp

from flax import struct
from flax.core.frozen_dict import FrozenDict
from flax.nnx.nnx import (
  filterlib,
  graph,
  rnglib,
  spmd,
  variables,
)
from flax.nnx.nnx.module import GraphDef, Module
from flax.nnx.nnx.proxy_caller import (
  CallableProxy,
  DelayedAccessor,
)
from flax.nnx.nnx.state import State
from flax.typing import Leaf
import jax
from jax._src.tree_util import broadcast_prefix
import jax.core
import jax.numpy as jnp
import jax.stages

A = tp.TypeVar('A')
C = tp.TypeVar('C')
B = tp.TypeVar('B')
F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])
G = tp.TypeVar('G', bound=tp.Callable[..., tp.Any])
M = tp.TypeVar('M', bound=Module)
MA = tp.TypeVar('MA', bound=Module)
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


class LiftedModule(tp.Generic[M], Module):  # type: ignore[ignored-abstractmethod]
  @abstractmethod
  def _call(self, accessor: DelayedAccessor, *args, **kwargs) -> tp.Any:
    pass

  @property
  @abstractmethod
  def _submodule(self) -> M:
    pass  # type: ignore[bad-return-type] # why pytype?

  def __call__(self, *args, **kwargs) -> tp.Any:
    return self.call(*args, **kwargs)  # type: ignore

  @property
  def call(self) -> tp.Any:
    module = self

    def check_and_call(accessor: DelayedAccessor, *args, **kwargs):
      return self._call(accessor, *args, **kwargs)

    proxy = CallableProxy(check_and_call)  # type: ignore[arg-type]

    while isinstance(module._submodule, LiftedModule):
      module = module._submodule
      proxy = proxy.call

    return proxy  # type: ignore


# -------------------------------
# jit
# -------------------------------

UNSPECIFIED = object()


def _default_constrain_state(state: State) -> State:
  state_spec = spmd.get_partition_spec(state)
  state = jax.lax.with_sharding_constraint(state, state_spec)
  return state


@dataclasses.dataclass(frozen=True)
class JitStaticInputs:
  graphdef: GraphDef[tuple[tp.Any, ...]]
  constrain_state: tp.Callable[[State], State] | None
  f: tp.Callable[..., tp.Any]


jax.tree_util.register_static(JitStaticInputs)


@dataclasses.dataclass(frozen=True)
class JitStaticOutputs:
  graphdef: GraphDef[tuple[tp.Any, ...]]
  index_mapping: dict[Index, Index]


jax.tree_util.register_static(JitStaticOutputs)


def _jitted_fn(
  *args: tp.Any,
  _nnx_jit_static: JitStaticInputs,
  _nnx_jit_state: State,
  **kwargs: tp.Any,
) -> tuple[tp.Any, State, GraphDef[tuple[tp.Any, ...]]]:
  ctx = graph.current_update_context('jit')
  graphdef = _nnx_jit_static.graphdef
  constrain_state = _nnx_jit_static.constrain_state
  f = _nnx_jit_static.f
  state: State = _nnx_jit_state

  if constrain_state is not None:
    state = constrain_state(state)

  input_graph_nodes = ctx.merge(graphdef, state)

  (args, kwargs) = graph.insert_graph_nodes((args, kwargs), input_graph_nodes)

  out = f(*args, **kwargs)

  out, output_graph_nodes = graph.extract_graph_nodes(out)

  graphdef, state = ctx.split((input_graph_nodes, output_graph_nodes))

  if constrain_state is not None:
    state = constrain_state(state)

  return out, state, graphdef


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

  _static_argnums = _normalize_sequence(static_argnums)
  _static_argnames = _normalize_sequence(static_argnames)
  _donate_argnums = _normalize_sequence(donate_argnums)
  _donate_argnames = _normalize_sequence(donate_argnames)

  if donate_state:
    _donate_argnames = (*_donate_argnames, '_nnx_jit_state')

  if callable(constrain_state):
    _constrain_state = constrain_state
  elif constrain_state:
    _constrain_state = _default_constrain_state
  else:
    _constrain_state = None

  jit_kwargs = {}
  if in_shardings is not UNSPECIFIED:
    jit_kwargs['in_shardings'] = in_shardings
  if out_shardings is not UNSPECIFIED:
    jit_kwargs['out_shardings'] = out_shardings

  jitted_fn = jax.jit(
    _jitted_fn,
    static_argnums=_static_argnums,
    static_argnames=_static_argnames,
    donate_argnums=_donate_argnums,
    donate_argnames=_donate_argnames,
    keep_unused=keep_unused,
    device=device,
    backend=backend,
    inline=inline,
    abstracted_axes=abstracted_axes,
    **jit_kwargs,
  )

  @functools.wraps(fun)
  @graph.update_context('jit')
  def jit_wrapper(*args, **kwargs):
    ctx = graph.current_update_context('jit')
    (args, kwargs), input_graph_nodes = graph.extract_graph_nodes(
      (args, kwargs)
    )
    graphdef, state = ctx.split(input_graph_nodes)
    out, output_state, output_graphdef = jitted_fn(
      *args,
      _nnx_jit_static=JitStaticInputs(graphdef, _constrain_state, fun),
      _nnx_jit_state=state,
      **kwargs,
    )
    input_graph_nodes, output_graph_nodes = ctx.merge(
      output_graphdef, output_state
    )
    out = graph.insert_graph_nodes(out, output_graph_nodes)
    return out

  jit_wrapper.inner = jitted_fn  # type: ignore

  return jit_wrapper  # type: ignore


class Jit(tp.Generic[M], LiftedModule[M]):
  @staticmethod
  def constructor(
    module_constructor: tp.Callable[..., MA],
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
  ) -> tp.Callable[..., 'Jit[MA]']:
    def _create_jit(*args, **kwargs):
      return Jit(
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
    @functools.partial(
      jit,
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
    def jit_call_module(
      module, *args, _nnx_jit_accessor: DelayedAccessor, **kwargs
    ):
      method = _nnx_jit_accessor(module)
      return method(*args, **kwargs)

    self.jitted_fn = jit_call_module
    self.module_constructor = module_constructor
    self.jit_module = self.module_constructor(
      *module_init_args, **module_init_kwargs
    )

  @property
  def _submodule(self) -> M:
    return self.jit_module

  def _call(self, accessor: DelayedAccessor, *args, **kwargs) -> tp.Any:
    out = self.jitted_fn(
      self.jit_module, *args, _nnx_jit_accessor=accessor, **kwargs
    )
    return out


# -------------------------------
# grad
# -------------------------------


def grad_fn(*args):
  f: tp.Callable[..., tp.Any]
  graphdef: GraphDef[tuple[dict[int, tp.Any], tuple[tp.Any, ...]]]
  non_diff_state: State
  has_aux: bool
  diff_args: list[int]
  ctx = graph.current_update_context('grad')
  *_args, f, graphdef, non_diff_state, has_aux, diff_args = args

  # rebuild diff_state from substates in args
  diff_state = State({})
  for i in diff_args:
    diff_state[i] = _args[i]
  diff_state = State({0: diff_state.raw_mapping})

  diff_graph_nodes, input_nodes = ctx.merge(
    graphdef, diff_state, non_diff_state
  )

  # add nodes to the args
  for i, arg in diff_graph_nodes.items():
    _args[i] = arg

  out = f(*_args)

  out, out_nodes = graph.extract_graph_nodes(out)

  graphdef_out, state_out = ctx.split((input_nodes, out_nodes))

  if has_aux:
    loss, aux = out
    out = (loss, (graphdef_out, state_out, aux))
  else:
    out = (out, (graphdef_out, state_out))

  return out


def _grad_general(
  f: tp.Callable[..., tp.Any],
  argnums: int | tp.Sequence[int],
  has_aux: bool,
  holomorphic: bool,
  allow_int: bool,
  reduce_axes: tp.Sequence[AxisName],
  wrt: filterlib.Filter,
  return_value: bool,
) -> tp.Callable[..., tp.Any]:
  @graph.update_context('grad')
  def grad_wrapper(*args):
    ctx: graph.UpdateContext = graph.current_update_context('grad')
    _argnums = _normalize_sequence(argnums)
    _, input_nodes = graph.extract_graph_nodes(args)

    _args = list(args)
    diff_graph_nodes: dict[int, tp.Any] = {
      i: arg
      for i, arg in enumerate(args)
      if i in _argnums and graph.is_node(arg)
    }

    def only_diff(path: tuple, value: tp.Any) -> bool:
      # diff_graph_nodes is the first element in the tuple
      return path[0] == 0

    graphdef, diff_state, non_diff_state = ctx.split(
      (diff_graph_nodes, input_nodes), filterlib.All(wrt, only_diff), ...
    )  # type: ignore[misc]

    # extract diff_state substates into the args
    diff_args: list[int] = []
    if 0 in diff_state:
      for i, diff_substate in diff_state[0].items():  # type: ignore
        assert isinstance(i, int)
        _args[i] = diff_substate
        diff_args.append(i)
    transform = jax.value_and_grad if return_value else jax.grad

    _argnums = _argnums[0] if len(_argnums) == 1 else _argnums

    out = transform(
      grad_fn,
      argnums=_argnums,
      has_aux=True,
      holomorphic=holomorphic,
      allow_int=allow_int,
      reduce_axes=reduce_axes,
    )(*_args, f, graphdef, non_diff_state, has_aux, diff_args)

    if return_value:
      if has_aux:
        (loss, (graphdef_out, state_out, aux)), grads = out
        out = (loss, aux), grads
      else:
        (loss, (graphdef_out, state_out)), grads = out
        out = loss, grads
    else:
      if has_aux:
        grads, (graphdef_out, state_out, aux) = out
        out = grads, aux
      else:
        out, (graphdef_out, state_out) = out

    input_nodes, out_nodes = ctx.merge(graphdef_out, state_out)

    out = graph.insert_graph_nodes(out, out_nodes)
    return out

  return grad_wrapper


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

    >>> from flax import nnx
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

  return _grad_general(
    f,
    argnums,
    has_aux,
    holomorphic,
    allow_int,
    reduce_axes,
    wrt,
    return_value=False,
  )


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
  return _grad_general(
    f,
    argnums,
    has_aux,
    holomorphic,
    allow_int,
    reduce_axes,
    wrt,
    return_value=True,
  )


class Grad(tp.Generic[M], LiftedModule[M]):
  @staticmethod
  def constructor(
    module_constructor: tp.Callable[..., MA],
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: tp.Sequence[AxisName] = (),
    return_value: bool = False,
    *,
    wrt: filterlib.Filter = variables.Param,
  ) -> tp.Callable[..., 'Grad[MA]']:
    def _create_grad(*args, **kwargs):
      return Grad(
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

  def __init__(
    self,
    module_constructor: tp.Callable[..., M],
    argnums: int | tp.Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: tp.Sequence[AxisName] = (),
    *,
    wrt: filterlib.Filter = variables.Param,
    # submodule args
    module_init_args: tuple[tp.Any, ...],
    module_init_kwargs: dict[str, tp.Any],
  ):
    self.module_constructor = module_constructor
    self.grad_module = self.module_constructor(
      *module_init_args, **module_init_kwargs
    )

    @functools.partial(
      grad,
      argnums=argnums,
      has_aux=has_aux,
      holomorphic=holomorphic,
      allow_int=allow_int,
      reduce_axes=reduce_axes,
      wrt=wrt,
    )
    def grad_call_apply(module, *args):
      *args, accessor = args
      method = accessor(module)
      return method(*args)

    self.grad_apply = grad_call_apply

  @property
  def _submodule(self) -> M:
    return self.grad_module

  def _call(self, accessor: DelayedAccessor, *args) -> tp.Any:
    return self.grad_apply(self.grad_module, *args, accessor)


# -------------------------------
# scan
# -------------------------------


@dataclasses.dataclass(frozen=True)
class FlatDef(tp.Generic[A]):
  type: type[A]
  treedef: jax.tree_util.PyTreeDef
  flat_axes: list[int | None]


jax.tree_util.register_static(FlatDef)


def _transpose_tree(tree: A, axes, /, *, move_front: bool) -> A:
  flatdef, flat_transposes, _ = _transpose_and_split(
    tree, axes, allow_none=False, move_front=move_front
  )
  return flatdef.treedef.unflatten(flat_transposes)


def _transpose_and_split(
  tree: A, axes, /, *, allow_none: bool = True, move_front: bool = True
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

      if node.shape == ():
        raise ValueError(f'Cannot map over a scalar array, got {node}')
      elif axis >= len(node.shape):
        raise ValueError(
          f'Axis {axis} out of bounds for array with shape {node.shape}'
        )

      if move_front:
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


@struct.dataclass
class ScanBroadcasts(tp.Generic[C, B]):
  flatdef: FlatDef[
    tuple[tuple[tp.Any, ...], dict[str, tp.Any], list[State]]
  ] = struct.field(pytree_node=False)
  flat_carry: list[tp.Any] = struct.field(pytree_node=True)
  graphdef: GraphDef[tuple[tp.Any, ...]] = struct.field(pytree_node=False)
  filters: tuple[filterlib.Filter, ...] = struct.field(pytree_node=False)
  f: tp.Callable[..., tuple[C, B] | C] = struct.field(pytree_node=False)
  # options
  carry_argnum: int = struct.field(pytree_node=False)
  state_axes: tp.Mapping[filterlib.Filter, int] = struct.field(
    pytree_node=False
  )
  split_rngs: filterlib.Filter = struct.field(pytree_node=False)
  transform_metadata: tp.Mapping[str, tp.Any] = struct.field(pytree_node=False)
  scan_output: bool = struct.field(pytree_node=False)


def scan_fn(
  carry: tuple[
    State,  # split_rng_state
    State,  # broadcast_rng_state
    State,  # carry_state
    tp.Any,  # carry_arg
    ScanBroadcasts[C, B],  # broadcasts
  ],
  scan: tuple[
    list[jax.Array | None],  # flat_scan
  ],
):
  split_rng_state, broadcast_rng_state, carry_state, carry_arg, broadcasts = (
    carry
  )
  (flat_scan,) = scan
  flatdef = broadcasts.flatdef
  flat_carry = broadcasts.flat_carry
  graphdef, filters = broadcasts.graphdef, broadcasts.filters
  f = broadcasts.f
  ctx = graph.current_update_context('scan')

  # merge args and kwargs
  args, kwargs, scan_states = _unflatten_splits(flatdef, flat_scan, flat_carry)
  # remove metadata axis name from Variable.sharding
  if spmd.PARTITION_NAME in broadcasts.transform_metadata:
    scan_states = [
      spmd.remove_axis(state, index, broadcasts.transform_metadata)
      for state, index in zip(scan_states, broadcasts.state_axes.values())
    ]

  # insert carry arg
  args = _insert_carry_arg(args, broadcasts.carry_argnum, carry_arg)

  # merge module state
  input_graph_nodes = ctx.merge(
    graphdef, *scan_states, carry_state, split_rng_state, broadcast_rng_state
  )
  (args, kwargs) = graph.insert_graph_nodes((args, kwargs), input_graph_nodes)

  out = f(*args, **kwargs)

  if broadcasts.scan_output:
    if not isinstance(out, tuple) or len(out) != 2:
      raise ValueError(
        'Expected a tuple of length 2 as the output of the scan function, '
        f'got {out}'
      )
    out = tp.cast(tuple[C, B], out)  # type: ignore[invalid-annotation]
    carry_arg_out, scan_args_out = out
  else:
    out = tp.cast(C, out)  # type: ignore[invalid-annotation]
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
  ) = ctx.split(  # type: ignore[misc]
    (input_graph_nodes, output_graph_nodes),
    rnglib.RngState,
    *filters,
  )

  split_rng_state_out, broadcast_rng_state_out = rng_state_out.split(
    broadcasts.split_rngs, ...
  )

  def _extract_carry_state(state: State, /):
    if 1 in state:
      raise ValueError(
        f'Cannot add new carry state during scan, got {state[1]}'
      )
    if 0 in state:
      _state = state[0]
      assert isinstance(_state, State)
      state = _state

    return state

  carry_state_out = _extract_carry_state(carry_state_out)
  split_rng_state_out = _extract_carry_state(split_rng_state_out)
  broadcast_rng_state_out = _extract_carry_state(broadcast_rng_state_out)

  # override  broadcast_rng_state_out to keep the same state
  # for the next iteration
  broadcast_rng_state_out = broadcast_rng_state

  # add metadata axis name to Variable.sharding
  if spmd.PARTITION_NAME in broadcasts.transform_metadata:
    scan_states_out = [
      spmd.add_axis(state, index, broadcasts.transform_metadata)
      for state, index in zip(scan_states_out, broadcasts.state_axes.values())
    ]

  carry_out = (
    split_rng_state_out,
    broadcast_rng_state_out,
    carry_state_out,
    carry_arg_out,
    broadcasts,
  )
  scan_out = (graphdef_out, scan_args_out, scan_states_out)

  return carry_out, scan_out


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
  @functools.wraps(f)
  @graph.update_context('scan')
  def scan_apply_wrapper(*args, **kwargs):
    # extract nodes
    (args, kwargs), input_graph_nodes = graph.extract_graph_nodes(
      (args, kwargs)
    )
    input_rng_streams = rnglib.backup_keys(input_graph_nodes)

    # extract carry arg
    carry_arg, args = _extract_carry_arg(args, carry_argnum)

    ctx = graph.current_update_context('scan')
    # split module state
    filters = (*state_axes.keys(), ...)
    graphdef, rng_state, *scan_states, carry_state = ctx.split(  # type: ignore[misc]
      input_graph_nodes, rnglib.RngState, *filters
    )

    # transpose axes arg
    flatdef, flat_scan, flat_carry = _transpose_and_split(
      (args, kwargs, scan_states),
      (in_axes, in_axes_kwargs, list(state_axes.values())),
    )

    # infer length
    lengths: set[int] = set(
      x.shape[0]  # type: ignore
      for x, axis in zip(flat_scan, flatdef.flat_axes)
      if axis is not None
    )

    if len(lengths) > 1:
      raise ValueError(
        'Inconsistent lengths between state_axes states and '
        f'arguments: {lengths}'
      )
    elif len(lengths) == 0:
      if length is None:
        raise ValueError(
          'Cannot infer length from state_axes states or axes_arg, '
          'please specify `length`'
        )
      infered_length = length
    else:
      infered_length = lengths.pop()
      if length is not None and length != infered_length:
        raise ValueError(
          f'Specified length {length} is not the same as the inferred '
          f'length {infered_length}'
        )

    # split rng state
    split_rng_state, broadcast_rng_state = rng_state.split(split_rngs, ...)

    broadcasts = ScanBroadcasts(
      flatdef,
      flat_carry,
      graphdef,
      filters,
      f,
      # options
      carry_argnum,
      state_axes,
      split_rngs,
      transform_metadata,
      scan_output,
    )
    carry = (
      split_rng_state,
      broadcast_rng_state,
      carry_state,
      carry_arg,
      broadcasts,
    )
    scan = (flat_scan,)

    carry_out, scan_out = jax.lax.scan(
      scan_fn,
      carry,
      scan,
      length=infered_length,
      reverse=reverse,
      unroll=unroll,
      _split_transpose=_split_transpose,
    )
    (
      split_rng_state_out,
      broadcast_rng_state_out,
      carry_state_out,
      carry_arg_out,
      broadcasts,
    ) = carry_out
    graphdef_out, scan_args_out, scan_states_out = scan_out

    scan_args_out, scan_states_out = _transpose_tree(
      (scan_args_out, scan_states_out),
      (out_axes, list(state_axes.values())),
      move_front=False,
    )

    if carry_state_out:
      carry_state_out = State({0: carry_state_out._mapping})
    if split_rng_state_out:
      split_rng_state_out = State({0: split_rng_state_out._mapping})
    if broadcast_rng_state_out:
      broadcast_rng_state_out = State({0: broadcast_rng_state_out._mapping})

    _, output_graph_nodes = ctx.merge(
      graphdef_out,
      *scan_states_out,
      carry_state_out,
      split_rng_state_out,
      broadcast_rng_state_out,
    )

    carry_arg_out, scan_args_out = graph.insert_graph_nodes(
      (carry_arg_out, scan_args_out), output_graph_nodes
    )

    rnglib.restore_keys(input_rng_streams)

    if scan_output:
      scan_args_out = tp.cast(B, scan_args_out)
      return carry_arg_out, scan_args_out
    else:
      return carry_arg_out

  return scan_apply_wrapper  # type: ignore


class Scan(tp.Generic[M], LiftedModule[M]):
  @staticmethod
  def constructor(
    module_constructor: tp.Callable[..., MA],
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
  ) -> tp.Callable[..., 'Scan[MA]']:
    def _create_scan(*args, **kwargs):
      return Scan(
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
    # use Vmap to handle initialisation
    vmapped_module = Vmap.constructor(
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

    @functools.partial(
      scan,
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
    def scan_call_apply(
      module, *args, _nnx_scan_accessor: DelayedAccessor, **kwargs
    ):
      method = _nnx_scan_accessor(module)
      return method(*args, **kwargs)

    self.scan_fn = scan_call_apply

  @property
  def _submodule(self) -> M:
    return self.scan_module

  def _call(
    self, accessor: DelayedAccessor, *args, **kwargs
  ) -> tuple[tp.Any, tp.Any]:
    return self.scan_fn(
      self._submodule, *args, _nnx_scan_accessor=accessor, **kwargs
    )


# -------------------------------
# remat
# -------------------------------


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


class Remat(tp.Generic[M], LiftedModule[M]):
  @staticmethod
  def constructor(
    module_constructor: tp.Callable[..., MA],
    prevent_cse: bool = True,
    static_argnums: int | tuple[int, ...] = (),
    policy: tp.Callable[..., bool] | None = None,
  ) -> tp.Callable[..., 'Remat[MA]']:
    def create_remat(*args, **kwargs):
      return Remat(
        module_constructor=module_constructor,
        module_init_args=args,
        module_init_kwargs=kwargs,
        prevent_cse=prevent_cse,
        static_argnums=static_argnums,
        policy=policy,
      )

    return create_remat

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


@graph.update_context('remat')
def remat_apply(
  options: RematOptions,
  f: tp.Callable[..., tp.Any],
  args: tuple[tp.Any, ...],
):
  ctx = graph.current_update_context('remat')
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

  _, output_nodes = ctx.merge(new_graphdef, new_state)
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


class Vmap(tp.Generic[M], LiftedModule[M]):
  @staticmethod
  def constructor(
    module_constructor: tp.Callable[..., MA],
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
  ) -> tp.Callable[..., 'Vmap[MA]']:
    def _create_vmap(*args, **kwargs):
      return Vmap(
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


@graph.update_context('vmap')
def vmap_apply(
  options: VmapOptions,
  f: tp.Callable[..., A],
  args: tuple[tp.Any, ...],
  kwargs: dict[str, tp.Any],
) -> A:
  (args, kwargs), input_graph_nodes = graph.extract_graph_nodes((args, kwargs))
  input_rng_streams = rnglib.backup_keys(input_graph_nodes)

  ctx = graph.current_update_context('vmap')
  # split module state
  filters = (*options.state_axes.keys(), ...)
  graphdef, rng_state, *vectorized_states, broadcast_state = ctx.split(  # type: ignore[misc]
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

  split_keys, split_counts, broadcast_keys, broadcast_counts = rnglib.fork(
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
      graphdef,
      *vectorized_states,
      broadcast_state,
      split_keys,
      split_counts,
      broadcast_keys,
      broadcast_counts,
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
    ) = ctx.split(  # type: ignore[misc]
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

  _, output_graph_nodes = ctx.merge(
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


# -------------------------------
# cond
# -------------------------------


@dataclasses.dataclass(frozen=True)
class CondStaticInputs(tp.Generic[A]):
  true_fun: tp.Callable[..., A]
  false_fun: tp.Callable[..., A]


jax.tree_util.register_static(CondStaticInputs)


def _cond_fun(
  is_true: bool,
  static_inputs: CondStaticInputs[A],
  graphdef: GraphDef[tuple[tp.Any, ...]],
  state: State,
):
  ctx = graph.current_update_context('cond')
  fn = static_inputs.true_fun if is_true else static_inputs.false_fun
  operands = ctx.merge(graphdef, state)
  out = fn(*operands)
  graphdef_out, state_out = ctx.split((operands, out))
  return graphdef_out, state_out


def _cond_true_fun(
  static_inputs: CondStaticInputs[A],
  graphdef: GraphDef[tuple[tp.Any, ...]],
  state: State,
):
  return _cond_fun(True, static_inputs, graphdef, state)


def _cond_false_fun(
  static_inputs: CondStaticInputs[A],
  graphdef: GraphDef[tuple[tp.Any, ...]],
  state: State,
):
  return _cond_fun(False, static_inputs, graphdef, state)


@graph.update_context('cond')
def cond(
  pred,
  true_fun: tp.Callable[..., A],
  false_fun: tp.Callable[..., A],
  *operands,
  **kwargs,
) -> A:
  ctx: graph.UpdateContext = graph.current_update_context('cond')
  graphdef, state = ctx.split(operands)
  graphdef_out, state_out = jax.lax.cond(
    pred,
    _cond_true_fun,
    _cond_false_fun,
    CondStaticInputs(true_fun=true_fun, false_fun=false_fun),
    graphdef,
    state,
    **kwargs,
  )
  _operands_out, out = ctx.merge(graphdef_out, state_out)
  return out
