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

from flax.experimental.nnx.nnx import filterlib, reprlib
from flax.experimental.nnx.nnx.variables import Variable

A = tp.TypeVar('A')

Leaf = tp.Any
Path = str
StateDict = dict[Path, tp.Any]
StateMapping = tp.Mapping[Path, tp.Any]


class State(tp.MutableMapping[Path, Leaf], reprlib.Representable):
  __slots__ = ('_mapping',)

  def __init__(
    self,
    __input: tp.Union[
      tp.Mapping[Path, Variable[Leaf]],
      tp.Iterator[tp.Tuple[Path, Variable[Leaf]]],
    ],
    /,
  ):
    self._mapping = dict(__input)

  @property
  def variables(self) -> dict[Path, Variable[Leaf]]:
    return self._mapping

  def __getitem__(self, __key: Path) -> Leaf:
    return self._mapping[__key].value

  def __setitem__(self, __key: Path, __value: Leaf) -> None:
    self._mapping[__key] = self._mapping[__key].replace(value=__value)

  def __delitem__(self, __key: Path) -> None:
    del self._mapping[__key]

  def __iter__(self) -> tp.Iterator[Path]:
    return iter(self._mapping)

  def __len__(self) -> int:
    return len(self._mapping)

  def __nnx_repr__(self):
    yield reprlib.Object(type(self), value_sep=': ', start='({', end='})')

    for k, v in self.items():
      yield reprlib.Attr(repr(k), v)

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
      states = State(states[0])
    else:
      states = tuple(State(state) for state in states)
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
      states = State(states[0])
    else:
      states = tuple(State(state) for state in states)

    return states

  @staticmethod
  def merge(state: 'State', /, *states: 'State') -> 'State':
    states = (state, *states)

    if len(states) == 1:
      return states[0]

    new_state: StateDict = {}

    for state in states:
      new_state.update(state.variables)

    return State(new_state)

  def __or__(self, other: 'State') -> 'State':
    if not other:
      return self
    return State.merge(self, other)

  def __sub__(self, other: 'State') -> 'State':
    if not other:
      return self

    # create new State via __new__ to avoid __init__ sorting
    _mapping = {k: v for k, v in self._mapping.items() if k not in other}
    state = object.__new__(State)
    state._mapping = _mapping
    return state


def _state_flatten_with_keys(
  x: State,
):
  items = sorted(x._mapping.items(), key=lambda item: item[0])
  children = tuple((jtu.DictKey(key), value) for key, value in items)
  return children, tuple(x._mapping.keys())


def _state_unflatten(
  static: tp.Tuple[Path, ...] | None,
  leaves: tp.Tuple[Leaf, ...] | tuple[dict[str, Leaf]],
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
  state: StateMapping,
  *filters: filterlib.Filter,
) -> tp.Tuple[StateDict, ...]:
  for i, filter_ in enumerate(filters):
    if filter_ is ... and i != len(filters) - 1:
      raise ValueError(
        'Ellipsis `...` can only be used as the last filter, '
        f'got it at index {i}.'
      )
  predicates = tuple(map(filterlib.to_predicate, filters))

  # we have n + 1 states, where n is the number of predicates
  # the last state is for values that don't match any predicate
  states: tp.Tuple[StateDict, ...] = tuple(
    {} for _ in range(len(predicates) + 1)
  )

  if isinstance(state, State):
    items = state._mapping.items()
  else:
    items = state.items()

  for path, value in items:
    for i, predicate in enumerate(predicates):
      if predicate(path, value):
        states[i][path] = value
        break
    else:
      # if we didn't break, set leaf to last state
      states[-1][path] = value

  return states
