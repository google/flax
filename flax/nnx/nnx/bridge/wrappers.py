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
import typing as tp
from typing import Any

from flax import nnx
from flax import linen
from flax.nnx.nnx import graph
from flax.nnx.nnx import variables as variableslib
from flax.nnx.nnx.module import GraphDef, Module
from flax.nnx.nnx.rnglib import Rngs
from flax.nnx.nnx.state import State
from flax.nnx.nnx.object import Object
import jax
from jax import tree_util as jtu

M = tp.TypeVar('M', bound=Module)


# Flax-like style is NNX
@dataclasses.dataclass
class Functional(tp.Generic[M]):
  module_type: tp.Type[M]
  graphdef: tp.Optional[GraphDef[M]]
  args: tuple[tp.Any, ...]
  kwargs: dict[str, tp.Any]

  def init(self, *, rngs: tp.Optional[Rngs] = None) -> State:
    kwargs = {}
    if rngs is not None:
      kwargs['rngs'] = rngs
    module = self.module_type(*self.args, **self.kwargs, **kwargs)
    graphdef, state = nnx.split(module)
    self.graphdef = graphdef
    return state

  def apply(self, *states: tp.Any):
    assert self.graphdef is not None
    return self.graphdef.apply(*states)


def functional(cls: tp.Type[M]) -> tp.Callable[..., Functional[M]]:
  def _functional_constructor(*args: tp.Any, **kwargs: tp.Any) -> Functional[M]:
    return Functional(cls, None, args, kwargs)

  return _functional_constructor


def _set_initializing(module: Module, initializing: bool):
  for _, value in graph.iter_graph(module):
    if isinstance(value, Object):
      value._object__state._initializing = initializing


def lazy_init(fn: Module | tp.Callable[..., tp.Any], *args, **kwargs):
  """To run through an arbitrary nnx.Module method and initialize all its needed state.

  Here used to trigger initialization of all `LinenToNNX` module variables."""
  if isinstance(fn, Module):
    module = fn
    assert callable(fn)
  else:
    assert hasattr(fn, '__self__') and isinstance(fn.__self__, Module), f'{fn = } needs to be a method of an NNX Module.'
    module = fn.__self__
  _set_initializing(module, True)
  try:
    _ = fn(*args, **kwargs)
  finally:
    _set_initializing(module, False)
  return fn


class LinenToNNX(Module):
  def __init__(
    self,
    module: linen.Module,
    rngs: tp.Optional[Rngs] = None,
  ):
    self.module = module
    self.rngs = rngs
    self.linen_collections: set[str] = set()

  def lazy_init(self, *args, **kwargs):
    return lazy_init(self, *args, **kwargs)

  def __call__(
    self, *args: Any, rngs: tp.Optional[Rngs] = None,
    method: tp.Callable[..., Any] | str | None = None, **kwargs: Any
  ) -> Any:

    # Shape-based lazy init of the flax variables
    if not rngs:
      rngs = self.rngs
    if self._object__state.initializing:
      _rngs = (
        {name: stream.key.raw_value for name, stream in rngs.items()}
        if rngs
        else {}
      )
      # rename default to params
      if 'params' not in _rngs and 'default' in _rngs:
        _rngs['params'] = _rngs.pop('default')
      out, variables = self.module.init_with_output(_rngs, *args, method=method, **kwargs)
      def nn_var_to_nnx_state(kp, v):
        assert isinstance(kp[0], jtu.DictKey)
        vtype = variableslib.variable_type(kp[0].key)
        return vtype(v)
      for col, tree in jtu.tree_map_with_path(nn_var_to_nnx_state, variables).items():
        self._setattr(col, tree)
        self.linen_collections.add(col)

    else:
      variables = {col: jax.tree.map(lambda v: v.value, getattr(self, col))
                   for col in self.linen_collections}
      _rngs = (
        {name: stream() for name, stream in rngs.items()} if rngs else {}
      )
      out = self.module.apply(variables, *args, rngs=_rngs, method=method, **kwargs)

    # Split out the updates if `mutable` is passed into the Flax module
    if kwargs.get('mutable', False) != False:
      out, updates = out
      for collection, value in updates.items():
        self._setattr(collection, jax.tree.map(variableslib.variable_type(collection), value))

    return out


class NNXToLinen(linen.Module):
  module: Module

  def setup(self):
    ...

  def __call__(self, *args, **kwargs):
    ...
