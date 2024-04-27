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

import inspect
import typing as tp

import jax
import jax.numpy as jnp
import optax

from flax.experimental.nnx.nnx.graph import Key
from flax.experimental.nnx.nnx.module import GraphDef, Module
from flax.experimental.nnx.nnx.proxy_caller import ApplyCaller
from flax.experimental.nnx.nnx.rnglib import Rngs
from flax.experimental.nnx.nnx.state import State
from flax.training.train_state import struct

A = tp.TypeVar('A')
M = tp.TypeVar('M', bound=Module)
TS = tp.TypeVar('TS', bound='TrainState')


class Dict(Module, tp.Mapping[str, A]):
  @tp.overload
  def __init__(self, iterable: tp.Iterable[tp.Tuple[str, A]], /):
    ...

  @tp.overload
  def __init__(
    self, mapping: tp.Optional[tp.Mapping[str, A]] = None, /, **kwargs: A
  ):
    ...

  def __init__(self, *args, **kwargs):
    for name, value in dict(*args, **kwargs).items():
      setattr(self, name, value)

  def __getitem__(self, key) -> A:
    return getattr(self, key)

  def __setitem__(self, key, value):
    setattr(self, key, value)

  def __getattr__(self, key) -> A:
    return super().__getattribute__(key)

  def __setattr__(self, key, value):
    super().__setattr__(key, value)

  def __iter__(self) -> tp.Iterator[str]:
    return (k for k in vars(self) if k != '_graph_node__state')

  def __len__(self) -> int:
    return len(vars(self))


class List(Module, tp.Generic[A]):
  def __init__(self, elems: tp.Iterable[A], /):
    i = 0
    for i, value in enumerate(elems):
      setattr(self, str(i), value)
    self._length = i + 1

  def __getitem__(self, key: int) -> A:
    if key >= len(self) or key < -len(self):
      raise IndexError(f'index {key} out of range for {self}')
    if key < 0:
      key = self._length + key
    return getattr(self, str(key))

  def __setitem__(self, key: int, value: A):
    if key >= len(self):
      raise IndexError(f'index {key} out of range for {self}')
    setattr(self, str(key), value)

  def __iter__(self) -> tp.Iterator[A]:
    for i in range(len(self)):
      yield getattr(self, str(i))

  def __len__(self) -> int:
    return self._length

  def _graph_node_flatten(self):
    nodes: list[tuple[Key, tp.Any]] = sorted(
      (int(key), value)
      for key, value in vars(self).items()
      if key not in ('_graph_node__state', '_length')
    )
    nodes.append(('_length', self._length))
    return nodes, type(self)

  def _graph_node_set_key(self, key: Key, value: tp.Any):
    if isinstance(key, int):
      key = str(key)
    return super()._graph_node_set_key(key, value)

  def _graph_node_pop_key(self, key: Key):
    if isinstance(key, int):
      key = str(key)
    return super()._graph_node_pop_key(key)


class Sequential(List):
  def __call__(self, *args, rngs: tp.Optional[Rngs] = None, **kwargs) -> tp.Any:
    output: tp.Any = None

    for i, f in enumerate(self):
      if not callable(f):
        raise TypeError(f'Sequence[{i}] is not callable: {f}')
      if i > 0:
        if isinstance(output, tp.Tuple):
          args = output
          kwargs = {}
        elif isinstance(output, dict):
          args = ()
          kwargs = output
        else:
          args = (output,)
          kwargs = {}
      if rngs is not None and has_keyword_arg(f, 'rngs'):
        kwargs['rngs'] = rngs

      output = f(*args, **kwargs)

    return output


class ModuleDefApply(tp.Protocol, tp.Generic[M]):
  def __call__(
    self, state: State, *states: State
  ) -> ApplyCaller[tuple[State, GraphDef[M]]]:
    ...


class TrainState(tp.Generic[M], struct.PyTreeNode):
  graphdef: GraphDef[M]
  params: State
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState
  step: jax.Array

  @classmethod
  def create(
    cls,
    graphdef: GraphDef[M],
    *,
    params: State,
    tx: optax.GradientTransformation,
    step: int = 0,
    **kwargs,
  ):
    return cls(
      graphdef=graphdef,
      params=params,
      tx=tx,
      opt_state=tx.init(params),
      step=jnp.asarray(step),
      **kwargs,
    )

  if tp.TYPE_CHECKING:

    def __getattr__(self, key: str) -> tp.Any:
      ...

  def apply(
    self, state: tp.Union[State, str], *states: tp.Union[State, str]
  ) -> ApplyCaller[tuple[GraphDef[M], State]]:
    states = (state, *states)

    _states: list[State] = []

    for _state in states:
      if isinstance(_state, str):
        _state_key = _state
        _state = getattr(self, _state_key)
        if not isinstance(_state, State):
          raise TypeError(
            f'Expected {self.__class__.__name__}.{_state_key} to be a State, got {type(_state)}'
          )
      _states.append(_state)

    return self.graphdef.apply(*_states)

  def apply_gradients(self: TS, grads: State, **kwargs) -> TS:
    updates, opt_state = self.tx.update(grads, self.opt_state, self.params)
    params = optax.apply_updates(self.params, updates)  # type: ignore
    step = self.step + 1
    return self.replace(
      params=params,
      opt_state=opt_state,
      step=step,
      **kwargs,
    )


def has_keyword_arg(func: tp.Callable[..., tp.Any], name: str) -> bool:
  """Return True if func has keyword-only arguments with the given name."""
  return any(
    param.name == name
    and param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD)
    for param in inspect.signature(func).parameters.values()
  )
