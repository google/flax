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

from flax.nnx.nnx import (
  extract,
  filterlib,
  graph,
  spmd,
  variables,
)
from flax.nnx.nnx.module import GraphDef, Module
from flax.nnx.nnx.proxy_caller import (
  CallableProxy,
  DelayedAccessor,
)
from flax.nnx.nnx.state import State
from flax.nnx.nnx.transforms import general
from flax.typing import Leaf
import jax
import jax.core
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

class Missing:
  pass


MISSING = Missing()


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


def jit_fn(
  *args,
  _nnx_jit_static: JitStaticInputs,
  _nnx_jit_state: State,
  **kwargs,
) -> tuple[tp.Any, State, GraphDef[tuple[tp.Any, ...]]]:
  ctx = graph.current_update_context('jit')
  graphdef = _nnx_jit_static.graphdef
  constrain_state = _nnx_jit_static.constrain_state
  f = _nnx_jit_static.f
  state: State = _nnx_jit_state

  if constrain_state is not None:
    state = constrain_state(state)

  input_graph_nodes = ctx.merge(graphdef, state)

  (args, kwargs) = extract.insert_graph_nodes((args, kwargs), input_graph_nodes)

  out = f(*args, **kwargs)

  out, output_graph_nodes = extract.extract_graph_nodes(out)

  graphdef, state = ctx.split((input_graph_nodes, output_graph_nodes))

  if constrain_state is not None:
    state = constrain_state(state)

  return out, state, graphdef


@tp.overload
def jit(
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
) -> tp.Callable[[F], F]: ...
@tp.overload
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
) -> F: ...
def jit(
  fun: F | Missing = MISSING,
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
) -> F | tp.Callable[[F], F]:
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
        - :py:class:`Sharding`, which will decide how the value
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

  if isinstance(fun, Missing):
    return functools.partial(
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
    jit_fn,
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
    (args, kwargs), input_graph_nodes = extract.extract_graph_nodes(
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
    out = extract.insert_graph_nodes(out, output_graph_nodes)
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
  ) -> tp.Callable[..., Jit[MA]]:
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
  *args, f, graphdef, non_diff_state, has_aux, diff_args = args

  # rebuild diff_state from substates in args
  diff_state = State({})
  for i in diff_args:
    diff_state[i] = args[i]
  diff_state: graph.GraphState = State({0: diff_state.raw_mapping})

  diff_graph_nodes, input_nodes = ctx.merge(
    graphdef, diff_state, non_diff_state
  )

  # add nodes to the args
  for i, arg in diff_graph_nodes.items():
    args[i] = arg

  # add other nodes to the args
  args = extract.insert_graph_nodes(args, input_nodes)

  out = f(*args)

  out, out_nodes = extract.extract_graph_nodes(out)

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
    diff_graph_nodes: dict[int, tp.Any] = {
      i: arg
      for i, arg in enumerate(args)
      if i in _argnums and graph.is_node(arg)
    }
    args, input_nodes = extract.extract_graph_nodes(args)
    args = list(args)

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
        args[i] = diff_substate
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
    )(*args, f, graphdef, non_diff_state, has_aux, diff_args)

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

    out = extract.insert_graph_nodes(out, out_nodes)
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
    >>> import jax.numpy as jnp
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
  ) -> tp.Callable[..., Grad[MA]]:
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
  ) -> tp.Callable[..., Remat[MA]]:
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
  args, input_nodes = extract.extract_graph_nodes(args)
  graphdef, state = ctx.split(input_nodes)

  def _remat_fn(state: State, *args):
    input_nodes = ctx.merge(graphdef, state)
    args = extract.insert_graph_nodes(args, input_nodes)
    out = f(*args)

    out, output_nodes = extract.extract_graph_nodes(out)
    new_graphdef, new_state = ctx.split((input_nodes, output_nodes))
    return (new_graphdef, new_state), out

  (new_graphdef, new_state), out = jax.checkpoint(
    _remat_fn,
    prevent_cse=options.prevent_cse,
    static_argnums=options.static_argnums,
    policy=options.policy,
  )(state, *args)

  _, output_nodes = ctx.merge(new_graphdef, new_state)
  out = extract.insert_graph_nodes(out, output_nodes)

  return out

@tp.overload
def remat(
  *,
  prevent_cse: bool = True,
  static_argnums: int | tuple[int, ...] = (),
  policy: tp.Callable[..., bool] | None = None,
) -> tp.Callable[[F], F]: ...
@tp.overload
def remat(
  f: F,
  *,
  prevent_cse: bool = True,
  static_argnums: int | tuple[int, ...] = (),
  policy: tp.Callable[..., bool] | None = None,
) -> F: ...
def remat(
  f: F | Missing = MISSING,
  *,
  prevent_cse: bool = True,
  static_argnums: int | tuple[int, ...] = (),
  policy: tp.Callable[..., bool] | None = None,
) -> F | tp.Callable[[F], F]:
  if isinstance(f, Missing):
    return functools.partial(
      remat,
      prevent_cse=prevent_cse,
      static_argnums=static_argnums,
      policy=policy,
    )

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
# eval_shape
# -------------------------------


def eval_shape(
  f: tp.Callable[..., A],
  *args: tp.Any,
  **kwargs: tp.Any,
) -> A:
  (args, kwargs), input_nodes = extract.extract_graph_nodes((args, kwargs))
  graphdef, state = graph.split(input_nodes)

  @functools.wraps(f)
  def _eval_shape_fn(state: State, *args, **kwargs):
    input_nodes = graph.merge(graphdef, state)
    args, kwargs = extract.insert_graph_nodes((args, kwargs), input_nodes)
    out = f(*args, **kwargs)
    out, output_nodes = extract.extract_graph_nodes(out)
    graphdef_out, state_out = graph.split(output_nodes)
    return graphdef_out, state_out, out

  graphdef_out, state_out, out = jax.eval_shape(
    _eval_shape_fn, state, *args, **kwargs
  )

  output_nodes = graph.merge(graphdef_out, state_out)
  out = extract.insert_graph_nodes(out, output_nodes)
  return out


# -------------------------------
# cond
# -------------------------------

@general.split_inputs(ctx_tag='cond')
def cond(
  pred,
  true_fun: tp.Callable[..., A],
  false_fun: tp.Callable[..., A],
  *operands,
  **kwargs,
) -> A:
  return jax.lax.cond(
    pred,
    general.merge_inputs(true_fun, ctx_tag='cond'),
    general.merge_inputs(false_fun, ctx_tag='cond'),
    *operands,
    **kwargs,
  )
