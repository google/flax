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
# pytype: skip-file
from __future__ import annotations

from abc import abstractmethod
import dataclasses
import functools
import inspect
import typing as tp

from flax.core import FrozenDict
from flax.nnx import (
  extract,
  graph,
)
from flax.nnx.module import Module
from flax.nnx.proxy_caller import (
  CallableProxy,
  DelayedAccessor,
)
from flax.nnx.transforms import general
from flax.typing import MISSING, Leaf, Missing
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
T = tp.TypeVar('T')
StrInt = tp.TypeVar('StrInt', str, int)
AxisName = tp.Hashable
Leaves = tp.List[Leaf]
Index = int


@tp.overload
def resolve_kwargs(
  fun: tp.Callable[..., tp.Any],
  args: tuple,
  kwargs: dict[str, tp.Any],
) -> tuple: ...
@tp.overload
def resolve_kwargs() -> tp.Callable[[F], F]: ...
def resolve_kwargs(
  fun: tp.Callable[..., tp.Any] | Missing = MISSING,
  args: tuple | Missing = MISSING,
  kwargs: dict[str, tp.Any] | Missing = MISSING,
) -> tuple | tp.Callable[[F], F]:
  if isinstance(fun, Missing):

    def resolve_kwargs_decorator(f):
      @functools.wraps(f)
      def resolve_kwargs_wrapper(*args, **kwargs):
        args = resolve_kwargs(f, args, kwargs)
        return f(*args)

      return resolve_kwargs_wrapper

    return resolve_kwargs_decorator  # type: ignore

  if isinstance(args, Missing):
    raise ValueError('args must be provided')
  if isinstance(kwargs, Missing):
    raise ValueError('kwargs must be provided')

  if isinstance(fun, functools.partial):
    # functools.partial should have an opaque signature.
    fun = lambda *args, **kwargs: None
  ba = inspect.signature(fun).bind(*args, **kwargs)
  ba.apply_defaults()
  if ba.kwargs:
    raise TypeError('keyword arguments could not be resolved to positions')
  else:
    return ba.args



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
# eval_shape
# -------------------------------


def eval_shape(
  f: tp.Callable[..., A],
  *args: tp.Any,
  **kwargs: tp.Any,
) -> A:
  args, kwargs = extract.to_tree((args, kwargs))

  @functools.wraps(f)
  def _eval_shape_fn(*args, **kwargs):
    args, kwargs = extract.from_tree((args, kwargs))
    out = f(*args, **kwargs)
    return extract.to_tree(out)

  out = jax.eval_shape(_eval_shape_fn, *args, **kwargs)
  return extract.from_tree(out)


# -------------------------------
# cond and switch
# -------------------------------


@general.split_inputs(ctxtag='cond')
def cond(
  pred,
  true_fun: tp.Callable[..., A],
  false_fun: tp.Callable[..., A],
  *operands,
  **kwargs,
) -> A:
  return jax.lax.cond(
    pred,
    general.merge_inputs(true_fun, ctxtag='cond'),
    general.merge_inputs(false_fun, ctxtag='cond'),
    *operands,
    **kwargs,
  )


@general.split_inputs(ctxtag='switch')
def switch(
  index,
  branches: tp.Sequence[tp.Callable[..., A]],
  *operands,
) -> A:
  return jax.lax.switch(
    index,
    [general.merge_inputs(f, ctxtag='switch') for f in branches],
    *operands,
  )


# -------------------------------
# while_loop
# -------------------------------


@dataclasses.dataclass(eq=False)
class WhileLoopCondFn:
  f: tp.Callable[..., tp.Any]

  def __post_init__(self):
    functools.update_wrapper(self, self.f)

  def __call__(self, pure_val):
    val = extract.from_tree(pure_val)
    out = self.f(val)
    return out


def _add_fake_index_mapping(tree: tp.Any):
  def per_node_state(ns: extract.NodeStates | tp.Any):
    global_index_mapping = {}
    if not isinstance(ns, extract.NodeStates):
      return ns
    assert isinstance(ns._graphdef, graph.NodeDef)

    def per_node_def(nd: graph.NodeDef | tp.Any):
      if nd.index >= 0:
        global_index_mapping[nd.index] = nd.index
      for sub_nd in nd.subgraphs.values():
        per_node_def(sub_nd)
      for l in nd.leaves.values():
        if isinstance(l, graph.NodeRef) and l.index >= 0:
          global_index_mapping[l.index] = l.index
      return

    per_node_def(ns._graphdef)
    return dataclasses.replace(ns, _graphdef=dataclasses.replace(
        ns._graphdef,
        index_mapping=FrozenDict(global_index_mapping)
      ))

  return jax.tree.map(per_node_state, tree,
                      is_leaf=lambda x: isinstance(x, extract.NodeStates))


def _remove_index_mapping(tree: tp.Any):
  '''Remove a fake index_mapping for the input to match that of the output.'''
  def per_node_state(ns: extract.NodeStates | tp.Any):
    if not isinstance(ns, extract.NodeStates):
      return ns
    assert isinstance(ns._graphdef, graph.NodeDef)
    return dataclasses.replace(ns, _graphdef=dataclasses.replace(
      ns._graphdef, index_mapping=None
    ))

  return jax.tree.map(per_node_state, tree,
                      is_leaf=lambda x: isinstance(x, extract.NodeStates))


@dataclasses.dataclass(eq=False)
class WhileLoopBodyFn:
  f: tp.Callable[..., tp.Any]

  def __post_init__(self):
    functools.update_wrapper(self, self.f)

  @graph.update_context('while_loop_body')
  def __call__(self, pure_val):
    # Removing the dummy index mapping being added outside of body function.
    pure_val_in = _remove_index_mapping(pure_val)

    val = extract.from_tree(pure_val_in, ctxtag='while_loop_body')
    out = self.f(val)
    pure_out = extract.to_tree(out, ctxtag='while_loop_body')

    try:
      jax.tree.map(lambda a, b: None, pure_val, pure_out)
    except ValueError as e:
      msg = ("nnx.while_loop requires body function's input and output to "
             "have the same reference and pytree structure, but they differ. "
             "If the mismatch comes from `index_mapping` field, you might "
             "have modified reference structure within the body function, "
             "which is not allowed."
             f"Detail of the mismatch: \n {str(e)}")
      raise ValueError(msg)

    return pure_out


@graph.update_context('while_loop')
def while_loop(cond_fun: tp.Callable[[T], tp.Any],
               body_fun: tp.Callable[[T], T],
               init_val: T) -> T:
  """NNX transform of `jax.lax.while_loop`.

  See: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html

  Caution: for the NNX internal reference tracing mechanism to work, you cannot
  change the reference structure of `init_val` inside `body_fun`.

  Args:
    cond_fun: a function for the continue condition of the while loop, taking a
      single input of type `T` and outputting a boolean.
    body_fun: a function that takes an input of type `T` and outputs an `T`.
      Note that both data and modules of `T` must have the same reference
      structure between inputs and outputs.
    init_val: the initial input for cond_fun and body_fun. Must be of type `T`.

  """

  pure_init_val = extract.to_tree(init_val, ctxtag='while_loop')

  # Adding the expected reference mapping to `pure_init_val` to match
  # `body_fun`'s output pytree structure, to make JAX while_loop happy.
  pure_init_val = _add_fake_index_mapping(pure_init_val)

  pure_out = jax.lax.while_loop(
    WhileLoopCondFn(cond_fun),
    WhileLoopBodyFn(body_fun),
    pure_init_val,
  )
  out = extract.from_tree(pure_out, ctxtag='while_loop')
  return out
