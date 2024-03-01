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

import typing as tp

import jax
import jax.tree_util as jtu

from flax import traverse_util
from flax.experimental.nnx.nnx import filterlib, reprlib
from flax.experimental.nnx.nnx.variables import Variable
from flax.typing import Path

A = tp.TypeVar('A')

Key = str
FlatState = dict[Path, Variable[Variable]]


class NestedStateRepr(reprlib.Representable):
  def __init__(self, state: State):
    self.state = state

  def __nnx_repr__(self):
    yield reprlib.Object('', value_sep=': ', start='{', end='}')

    for r in self.state.__nnx_repr__():
      if isinstance(r, reprlib.Object):
        continue
      yield r


class State(tp.MutableMapping[Key, tp.Any], reprlib.Representable):
  def __init__(
    self,
    mapping: tp.Union[
      tp.Mapping[Key, tp.Any],
      tp.Iterator[tp.Tuple[Key, tp.Any]],
    ],
    /,
  ):
    if tp.TYPE_CHECKING:
      self._mapping = dict(mapping)
    else:
      super().__setattr__('_mapping', dict(mapping))

  @property
  def raw_mapping(self) -> dict[Key, dict[str, tp.Any] | tp.Any]:
    return self._mapping

  def __getitem__(self, key: Key | int) -> Variable | State:
    if isinstance(key, int):
      key = str(key)
    value = self._mapping[key]
    if isinstance(value, Variable):
      return value
    return State(value)

  def __getattr__(self, key: Key) -> Variable | State:
    if '_mapping' not in vars(self) or key not in self._mapping:
      raise AttributeError(f'No attribute {key} in State')

    return self[key]

  def __setitem__(self, key: Key | int, value: Variable | State) -> None:
    if isinstance(key, int):
      key = str(key)

    if not isinstance(value, (Variable, State)):
      raise ValueError(
        f'Trying to set key {key} to a value'
        f' that is not a Variable or State, got: {value}.'
      )
    if isinstance(value, State):
      self._mapping[key] = value._mapping
    else:
      self._mapping[key] = value

  __setattr__ = __setitem__

  def __delitem__(self, key: Key) -> None:
    del self._mapping[key]

  def __iter__(self) -> tp.Iterator[Key]:
    return iter(self._mapping)

  def __len__(self) -> int:
    return len(self._mapping)

  def __nnx_repr__(self):
    yield reprlib.Object(type(self), value_sep=': ', start='({', end='})')

    for k, v in self.items():
      if isinstance(v, State):
        v = NestedStateRepr(v)
      yield reprlib.Attr(repr(k), v)

  def flat_state(self) -> dict[Key, Variable[Variable]]:
    return traverse_util.flatten_dict(self._mapping, sep='/')  # type: ignore

  @classmethod
  def from_flat_path(cls, flat_state: FlatState, /) -> State:
    nested_state = traverse_util.unflatten_dict(flat_state, sep='/')
    return cls(nested_state)

  @tp.overload
  def split(self, first: filterlib.Filter, /) -> 'State':
    ...

  @tp.overload
  def split(
    self,
    first: filterlib.Filter,
    second: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
  ) -> tp.Tuple['State', ...]:
    ...

  def split(
    self, first: filterlib.Filter, /, *filters: filterlib.Filter
  ) -> tp.Union['State', tp.Tuple['State', ...]]:
    filters = (first, *filters)
    *states, rest = _split_state(self, *filters)

    if rest:
      raise ValueError(
        'Non-exhaustive filters, got a non-empty remainder: '
        f'{list(rest.keys())}.\nUse `...` to match all remaining elements.'
      )

    if len(states) == 1:
      states = states[0]
    else:
      states = tuple(states)
    return states

  @tp.overload
  def extract(
    self,
    first: filterlib.Filter,
    /,
  ) -> 'State':
    ...

  @tp.overload
  def extract(
    self,
    first: filterlib.Filter,
    second: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
  ) -> tp.Tuple['State', ...]:
    ...

  def extract(
    self,
    first: filterlib.Filter,
    /,
    *filters: filterlib.Filter,
  ) -> tp.Union['State', tp.Tuple['State', ...]]:
    *states, _rest = _split_state(self, first, *filters)

    assert len(states) == len(filters) + 1

    if len(states) == 1:
      states = states[0]
    else:
      states = tuple(states)

    return states

  @staticmethod
  def merge(state: 'State', /, *states: 'State') -> 'State':
    states = (state, *states)

    if len(states) == 1:
      return states[0]

    new_state: FlatState = {}

    for state in states:
      new_state.update(state.flat_state())

    return State.from_flat_path(new_state)

  def __or__(self, other: 'State') -> 'State':
    if not other:
      return self
    return State.merge(self, other)

  def __sub__(self, other: 'State') -> 'State':
    if not other:
      return self

    self_flat = self.flat_state()
    other_flat = other.flat_state()
    diff = {k: v for k, v in self_flat.items() if k not in other_flat}

    return State.from_flat_path(diff)


def _state_flatten_with_keys(x: State):
  items = sorted(x._mapping.items(), key=lambda item: item[0])
  children = tuple((jtu.DictKey(key), value) for key, value in items)
  return children, tuple(x._mapping.keys())


def _state_unflatten(
  static: tp.Tuple[Path, ...] | None,
  leaves: tp.Tuple[Variable, ...] | tuple[dict[str, Variable]],
):
  return State(zip(static, leaves)) if static else State(leaves[0])


def _state_flatten(x: State):
  return (x._mapping,), None


jax.tree_util.register_pytree_with_keys(
  State,
  _state_flatten_with_keys,
  _state_unflatten,
  flatten_func=_state_flatten,
)


def _split_state(
  state: State,
  *filters: filterlib.Filter,
) -> tp.Tuple[State, ...]:
  for i, filter_ in enumerate(filters):
    if filter_ is ... and i != len(filters) - 1:
      raise ValueError(
        'Ellipsis `...` can only be used as the last filter, '
        f'got it at index {i}.'
      )
  predicates = tuple(map(filterlib.to_predicate, filters))

  flat_state = state.flat_state()

  # we have n + 1 states, where n is the number of predicates
  # the last state is for values that don't match any predicate
  flat_states: tp.Tuple[FlatState, ...] = tuple(
    {} for _ in range(len(predicates) + 1)
  )

  for path, value in flat_state.items():
    for i, predicate in enumerate(predicates):
      if predicate(path, value):
        flat_states[i][path] = value
        break
    else:
      # if we didn't break, set leaf to last state
      flat_states[-1][path] = value

  return tuple(State.from_flat_path(flat_state) for flat_state in flat_states)
