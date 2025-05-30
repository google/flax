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

import dataclasses
from functools import partial
import typing as tp
from typing import Any

from flax import linen
from flax import nnx
from flax.core import FrozenDict
from flax.core import meta
from flax.nnx import graph
from flax.nnx import variablelib
from flax.nnx.bridge import variables as bv
from flax.nnx.bridge import module as bdg_module
from flax.nnx.module import Module
from flax.nnx.object import Object
from flax.nnx.rnglib import Rngs
from flax.nnx.statelib import State
import jax
from jax import tree_util as jtu
from flax import config

M = tp.TypeVar('M', bound=Module)

# Flax-like style is NNX
@dataclasses.dataclass
class Functional(tp.Generic[M]):
  module_type: tp.Type[M]
  graphdef: tp.Optional[graph.GraphDef[M]]
  args: tuple[tp.Any, ...]
  kwargs: dict[str, tp.Any]

  def init(self, *, rngs: tp.Optional[Rngs] = None) -> State:
    kwargs = {}
    if rngs is not None:
      kwargs['rngs'] = rngs
    module = self.module_type(*self.args, **self.kwargs, **kwargs)
    graphdef, state = nnx.split(module)
    self.graphdef = graphdef
    return state  # type: ignore

  def apply(self, *states: tp.Any):
    assert self.graphdef is not None
    return self.graphdef.apply(*states)


def functional(cls: tp.Type[M]) -> tp.Callable[..., Functional[M]]:
  def _functional_constructor(*args: tp.Any, **kwargs: tp.Any) -> Functional[M]:
    return Functional(cls, None, args, kwargs)

  return _functional_constructor


def _set_initializing(module: Module, initializing: bool):
  for k, value in graph.iter_graph(module):
    if isinstance(value, Object):
      value._object__state._initializing = initializing


def lazy_init(fn: Module | tp.Callable[..., tp.Any], *args, **kwargs):
  """To run through an arbitrary nnx.Module method and initialize all its needed state.

  Here used to trigger initialization of all `LinenToNNX` module variables."""
  if isinstance(fn, Module):
    module = fn
    assert callable(fn)
  else:
    if not (hasattr(fn, '__self__') and isinstance(fn.__self__, Module)):
      raise ValueError(f'{fn = } needs to be a method of an NNX Module.')
    module = fn.__self__
  _set_initializing(module, True)
  try:
    _ = fn(*args, **kwargs)
  finally:
    _set_initializing(module, False)
  return fn

PYTREE_DEFAULT = 'auto' if config.flax_mutable_array else None

class ToNNX(Module):
  """A wrapper to turn any Linen module into an NNX module.

  The result NNX module can be used standalone with all NNX APIs, or as a submodule of
  another NNX module.

  Since Linen module initialization requires a sample input, you need to call `lazy_init`
  with an argument to initialize the variables.

  Example::

    >>> from flax import linen as nn, nnx
    >>> import jax
    >>> linen_module = nn.Dense(features=64)
    >>> x = jax.numpy.ones((1, 32))
    >>> # Like Linen init(), initialize with a sample input
    >>> model = nnx.bridge.ToNNX(linen_module, rngs=nnx.Rngs(0)).lazy_init(x)
    >>> # Like Linen apply(), but using NNX's direct call method
    >>> y = model(x)
    >>> model.kernel.shape
    (32, 64)

  Args:
    module: The Linen Module instance.
    rngs: The `nnx.Rngs` instance being passed to any NNX module.

  Returns:
    A stateful NNX module that behaves the same as the wrapped Linen module.
  """

  __data__ = 'auto'

  def __init__(
    self,
    module: linen.Module,
    rngs: tp.Optional[Rngs] = None,
  ):
    self.module = module
    self.rngs = rngs

  def lazy_init(self, *args, **kwargs):
    """A shortcut of calling `nnx.bridge.lazy_init()` upon this module."""
    return lazy_init(self, *args, **kwargs)

  def __getattr__(self, name: str):
    if hasattr(super(), name):
      return super().__getattribute__(name)
    maybe_method = getattr(self.module.__class__, name, None)
    if callable(maybe_method):
      method = partial(self.__call__, method=maybe_method)
      method.__self__ = self
      return method
    return super().__getattribute__(name)

  def __call__(
    self, *args: Any, rngs: tp.Optional[Rngs] = None,
    method: tp.Callable[..., Any] | str | None = None, **kwargs: Any
  ) -> Any:

    # Shape-based lazy init of the flax variables
    if not rngs:
      rngs = self.rngs
    if self._object__state.initializing:
      _rngs = (
        {name: stream() for name, stream in rngs.items()} if rngs else {}
      )
      # rename default to params
      if 'params' not in _rngs and 'default' in _rngs:
        _rngs['params'] = _rngs.pop('default')
      out, variables = self.module.init_with_output(_rngs, *args, method=method, **kwargs)

      nnx_attrs = bv.linen_vars_to_nnx_attrs(variables)
      for attr_name, value in nnx_attrs.items():
        setattr(self, attr_name, value)

    else:
      nnx_attrs = {k: v for k, v in vars(self).items()
                   if k not in ['module', 'rngs', '_object__state']}
      variables = bv.nnx_attrs_to_linen_vars(nnx_attrs)

      _rngs = (
        {name: stream() for name, stream in rngs.items()} if rngs else {}
      )

      # Get `mutable` from top level bridge.Module context if any
      if (m := bdg_module.current_module()) is not None:
        assert m.scope is not None
        mutable = m.scope.mutable
        if 'mutable' in kwargs and kwargs['mutable'] != mutable:
          raise ValueError(
            f"Multiple `mutable` arguments detected: {mutable} at top level vs "
            f"{kwargs['mutable']} in ToNNX() call")
        kwargs['mutable'] = mutable

      out = self.module.apply(variables, *args, rngs=_rngs, method=method, **kwargs)

    # Split out the updates if `mutable` is passed into the Flax module
    if kwargs.get('mutable', False) != False:
      out, updates = out
      nnx_attrs = bv.linen_vars_to_nnx_attrs(updates)
      for attr_name, value in nnx_attrs.items():
        if hasattr(self, attr_name) and isinstance(value, dict):
          original_tree = getattr(self, attr_name)
          setattr(self, attr_name, original_tree | value)
        else:
          setattr(self, attr_name, value)

    return out


def linen_rngs_dict(linen_module: linen.Module, add_default: bool = False):
  """Given a module, split out one of its every active RNG key collections."""
  assert linen_module.scope is not None, 'linen_rngs_dict() must be called inside a Linen module.'
  rngs: dict[str, tp.Any] = {
      name: linen_module.make_rng(name)
      for name in linen_module.scope.rngs.keys()
  }
  if add_default and 'default' not in rngs:
    rngs['default'] = 0
  return rngs


class ToLinen(linen.Module):
  """A wrapper to turn any NNX module into a Linen module.

  The result Linen module can be used standalone with all Linen APIs, or as a
  submodule of
  another Linen module.

  Since NNX modules are stateful and owns the state, we only create it once
  during init
  time, and will track its state and static data as separate variables.

  Example::

    >>> from flax import linen as nn, nnx
    >>> import jax
    >>> model = nnx.bridge.ToLinen(nnx.Linear, args=(32, 64))
    >>> x = jax.numpy.ones((1, 32))
    >>> y, variables = model.init_with_output(jax.random.key(0), x)
    >>> y.shape
    (1, 64)
    >>> variables['params']['kernel'].shape
    (32, 64)
    >>> # The static GraphDef of the underlying NNX module
    >>> variables.keys()
    dict_keys(['params'])

  Args:
    nnx_class: The NNX Module class (not instance!).
    args: The arguments that normally would be passed in to create the NNX
      module.
    kwargs: The keyword arguments that normally would be passed in to create the
      NNX module.
    skip_rng: True if this NNX module doesn't need `rngs` arg during
      initialization (not common).

  Returns:
    A stateful NNX module that behaves the same as the wrapped Linen module.
  """
  nnx_class: tp.Callable[..., Module]
  args: tp.Sequence = ()
  kwargs: tp.Mapping[str, tp.Any] = FrozenDict({})
  skip_rng: bool = False
  metadata_fn: tp.Callable[[variablelib.VariableState], tp.Any] | None = (
      bv.to_linen_var
  )

  @linen.compact
  def __call__(self, *args, **kwargs):
    module_kwargs = dict(self.kwargs)
    maybe_add_default = not self.is_initializing()
    def _module_kwargs():
      if not self.skip_rng:
        module_kwargs['rngs'] = nnx.Rngs(
            **linen_rngs_dict(self, add_default=maybe_add_default)
        )
      return module_kwargs

    # init codepath
    if self.is_initializing():
      module = self.nnx_class(*self.args, **_module_kwargs())
      # TODO: add lazy_init here in case there's an `ToNNX` submodule under `module`.
      # update linen variables before call module to save initial state
      self._update_variables(module)
      out = module(*args, **kwargs)
      return out

    # create state
    def maybe_unbox(x):
      if isinstance(x, meta.AxisMetadata):
        return x.unbox()
      return x
    states = jtu.tree_map(
        maybe_unbox,
        list(self.variables.values()),
        is_leaf=lambda x: isinstance(x, meta.AxisMetadata),
    )
    if not states:
      states = ({},)

    # update module state
    module = nnx.eval_shape(
        lambda: self.nnx_class(*self.args, **_module_kwargs())
    )
    nnx.update(module, *states)
    nnx.reseed(
        module, **linen_rngs_dict(self, add_default=maybe_add_default)
    )  # reseed with keys from linen apply call.

    out = module(*args, **kwargs)
    self._update_variables(module)
    return out

  def _update_variables(self, module):
    """Store the NNX module's graph def and state inside Linen module variables."""
    state = nnx.state(module, nnx.Not(nnx.RngState))

    collection_flat_state: dict[str, list[tuple[tuple[str, ...], tp.Any]]] = {}

    # group state by collection
    for path, leaf in nnx.to_flat_state(state):
      type_ = leaf.type if isinstance(leaf, nnx.VariableState) else type(leaf)
      collection = variablelib.variable_name_from_type(
          type_, allow_register=True
      )
      if collection not in collection_flat_state:
        collection_flat_state[collection] = []
      collection_flat_state[collection].append((path, leaf))

    # update linen variables
    for collection, flat_state in collection_flat_state.items():
      if self.is_mutable_collection(collection):

        def _to_linen_var(x):
          if isinstance(x, nnx.VariableState):
            if self.metadata_fn:
              return self.metadata_fn(x)
            else:
              return x.value
          return x

        collection_state = nnx.traversals.unflatten_mapping(flat_state)
        collection_state = jax.tree.map(
            _to_linen_var,
            collection_state,
            is_leaf=lambda x: isinstance(x, nnx.VariableState),
        )
        for k, v in collection_state.items():
          self.put_variable(collection, k, v)


def to_linen(
    nnx_class: tp.Callable[..., Module],
    *args,
    metadata_fn: (
        tp.Callable[[variablelib.VariableState], tp.Any] | None
    ) = bv.to_linen_var,
    name: str | None = None,
    **kwargs,
):
  """Shortcut of `nnx.bridge.ToLinen` if user is not changing any of its default fields."""
  return ToLinen(
      nnx_class,
      args=args,
      kwargs=FrozenDict(kwargs),
      metadata_fn=metadata_fn,
      name=name,
  )
