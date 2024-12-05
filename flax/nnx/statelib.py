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

import typing as tp

import jax

from flax.nnx import traversals
from flax.nnx import filterlib, reprlib
from flax.nnx import variablelib
from flax.typing import PathParts

A = tp.TypeVar('A')
K = tp.TypeVar('K', bound=tp.Hashable)
V = tp.TypeVar('V')

ExtractValueFn = tp.Callable[[tp.Any], tp.Any]
SetValueFn = tp.Callable[[V, tp.Any], V]


class NestedStateRepr(reprlib.Representable):
  def __init__(self, state: State):
    self.state = state

  def __nnx_repr__(self):
    yield reprlib.Object('', value_sep=': ', start='{', end='}')

    for r in self.state.__nnx_repr__():
      if isinstance(r, reprlib.Object):
        continue
      yield r

  def __treescope_repr__(self, path, subtree_renderer):
    children = {}
    for k, v in self.state.items():
      if isinstance(v, dict):
        v = NestedStateRepr(v)
      children[k] = v
    # Render as the dictionary itself at the same path.
    return subtree_renderer(children, path=path)

class FlatState(tp.Sequence[tuple[PathParts, V]], reprlib.PrettySequence):
  _keys: tuple[PathParts, ...]
  _values: list[V]

  def __init__(self, items: tp.Iterable[tuple[PathParts, V]]):
    keys, values = [], []
    for key, value in items:
      keys.append(key)
      values.append(value)
    self._keys = tuple(keys)
    self._values = values

  @tp.overload
  def __getitem__(self, index: int) -> tuple[PathParts, V]: ...
  @tp.overload
  def __getitem__(self, index: slice) -> FlatState[V]: ...
  def __getitem__(
    self, index: int | slice
  ) -> tuple[PathParts, V] | FlatState[V]:
    if isinstance(index, int):
      return self._keys[index], self._values[index]
    return FlatState(zip(self._keys[index], self._values[index]))

  def __len__(self) -> int:
    return len(self._keys)

  def __iter__(self) -> tp.Iterator[tuple[PathParts, V]]:
    return iter(zip(self._keys, self._values))

  def items(self) -> tp.Iterator[tuple[PathParts, V]]:
    return iter(zip(self._keys, self._values))


def _flat_state_pytree_flatten(x: FlatState[V]):
  return x._values, x._keys


def _flat_state_pytree_unflatten(
  keys: tuple[PathParts, ...], values: list[V]
) -> FlatState[V]:
  flat_state = object.__new__(FlatState)
  flat_state._keys = keys
  flat_state._values = values
  return flat_state


jax.tree_util.register_pytree_node(
  FlatState,
  _flat_state_pytree_flatten,
  _flat_state_pytree_unflatten,
)

State = dict


def map_state(f: tp.Callable[[tuple, tp.Any], tp.Any], state: State) -> State:
  flat_state = to_flat_state(state)
  result = [
    (path, f(path, variable_state)) for path, variable_state in flat_state
  ]
  return traversals.unflatten_mapping(result)


def to_flat_state(state: State) -> FlatState:
  return FlatState(traversals.flatten_to_sequence(state))


def from_flat_state(
  flat_state: tp.Mapping[PathParts, V] | tp.Iterable[tuple[PathParts, V]],
  /,
) -> State:
  if not isinstance(flat_state, tp.Mapping):
    flat_state = dict(flat_state)
  nested_state = traversals.unflatten_mapping(flat_state)
  return nested_state


def to_pure_dict(
  state, extract_fn: ExtractValueFn | None = None
) -> dict[str, tp.Any]:
  # Works for nnx.Variable and nnx.VariableState
  if extract_fn is None:
    extract_fn = lambda x: x.value if hasattr(x, 'value') else x
  return jax.tree.map(
    extract_fn,
    state,
    is_leaf=lambda x: isinstance(x, variablelib.VariableState),
  )


def replace_by_pure_dict(
  state, pure_dict: dict[str, tp.Any], replace_fn: SetValueFn | None = None
):
  def try_convert_int(x):
    try:
      return int(x)
    except ValueError:
      return x

  # Works for nnx.Variable and nnx.VariableState
  if replace_fn is None:
    replace_fn = lambda x, v: x.replace(v) if hasattr(x, 'replace') else v
  current_flat = traversals.flatten_mapping(state)
  for kp, v in traversals.flatten_mapping(pure_dict).items():
    kp = tuple(map(try_convert_int, kp))
    if kp not in current_flat:
      raise ValueError(f'key in pure_dict not available in state: {kp}')
    current_flat[kp] = replace_fn(current_flat[kp], v)
  state.update(traversals.unflatten_mapping(current_flat))


@tp.overload
def split_state(state: State, first: filterlib.Filter, /) -> State: ...


@tp.overload
def split_state(
  state: State,
  first: filterlib.Filter,
  second: filterlib.Filter,
  /,
  *filters: filterlib.Filter,
) -> tuple[State, ...]: ...


@tp.overload
def split_state(
  state: State, /, *filters: filterlib.Filter
) -> tp.Union[State, tuple[State, ...]]: ...


def split_state(  # type: ignore[misc]
  state: State, first: filterlib.Filter, /, *filters: filterlib.Filter
) -> tp.Union[State, tuple[State, ...]]:
  """Split a ``State`` into one or more ``State``'s. The
  user must pass at least one ``Filter`` (i.e. :class:`Variable`),
  and the filters must be exhaustive (i.e. they must cover all
  :class:`Variable` types in the ``State``).

  Example usage::

    >>> from flax import nnx

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.batchnorm = nnx.BatchNorm(2, rngs=rngs)
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
    ...   def __call__(self, x):
    ...     return self.linear(self.batchnorm(x))

    >>> model = Model(rngs=nnx.Rngs(0))
    >>> state = nnx.state(model)
    >>> param, batch_stats = nnx.split_state(state, nnx.Param, nnx.BatchStat)

  Arguments:
    first: The first filter
    *filters: The optional, additional filters to group the state into mutually exclusive substates.
  Returns:
    One or more ``States`` equal to the number of filters passed.
  """
  filters = (first, *filters)
  *states_, rest = _split_state(to_flat_state(state), *filters)

  if rest:
    raise ValueError(
      'Non-exhaustive filters, got a non-empty remainder: '
      f'{rest}.\nUse `...` to match all remaining elements.'
    )

  states: State | tuple[State, ...]
  if len(states_) == 1:
    states = states_[0]
  else:
    states = tuple(states_)
  return states  # type: ignore


@tp.overload
def filter_state(
  self,
  first: filterlib.Filter,
  /,
) -> State: ...


@tp.overload
def filter_state(
  self,
  first: filterlib.Filter,
  second: filterlib.Filter,
  /,
  *filters: filterlib.Filter,
) -> tuple[State, ...]: ...


def filter_state(
  self,
  first: filterlib.Filter,
  /,
  *filters: filterlib.Filter,
) -> tp.Union[State, tuple[State, ...]]:
  """Filter a ``State`` into one or more ``State``'s. The
  user must pass at least one ``Filter`` (i.e. :class:`Variable`).
  This method is similar to :meth:`split() <flax.nnx.State.state.split>`,
  except the filters can be non-exhaustive.

  Example usage::

    >>> from flax import nnx

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.batchnorm = nnx.BatchNorm(2, rngs=rngs)
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
    ...   def __call__(self, x):
    ...     return self.linear(self.batchnorm(x))

    >>> model = Model(rngs=nnx.Rngs(0))
    >>> state = nnx.state(model)
    >>> param = nnx.filter_state(state, nnx.Param)
    >>> batch_stats = nnx.filter_state(state, nnx.BatchStat)
    >>> param, batch_stats = nnx.filter_state(state, nnx.Param, nnx.BatchStat)

  Arguments:
    first: The first filter
    *filters: The optional, additional filters to group the state into mutually exclusive substates.
  Returns:
    One or more ``States`` equal to the number of filters passed.
  """
  *states_, _rest = _split_state(to_flat_state(self), first, *filters)

  assert len(states_) == len(filters) + 1

  states: State | tuple[State, ...]
  if len(states_) == 1:
    states = states_[0]
  else:
    states = tuple(states_)

  return states  # type: ignore


def merge_state(state: tp.Mapping, /, *states: tp.Mapping) -> State:
  """The inverse of :meth:`split() <flax.nnx.State.state.split>`.

  ``merge`` takes one or more ``State``'s and creates
  a new ``State``.

  Example usage::

    >>> from flax import nnx
    >>> import jax.numpy as jnp

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.batchnorm = nnx.BatchNorm(2, rngs=rngs)
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
    ...   def __call__(self, x):
    ...     return self.linear(self.batchnorm(x))

    >>> model = Model(rngs=nnx.Rngs(0))
    >>> params, batch_stats = nnx.state(model, nnx.Param, nnx.BatchStat)
    >>> params['linear']['bias'].value += 1

    >>> state = nnx.merge_state(params, batch_stats)
    >>> nnx.update(model, state)
    >>> assert (model.linear.bias.value == jnp.array([1, 1, 1])).all()

  Args:
    state: A ``State`` object.
    *states: Additional ``State`` objects.
  Returns:
    The merged ``State``.
  """
  if not states:
    if isinstance(state, dict):
      return state
    return dict(state)

  states = (state, *states)

  new_state: dict[PathParts, tp.Any] = {}

  for state in states:
    new_state.update(traversals.flatten_mapping(state))  # type: ignore[attribute-error] # pytype is wrong here

  return from_flat_state(new_state)




def diff(state: State, other: State) -> State:
  if not other:
    return state

  self_flat = to_flat_state(state)
  other_flat = to_flat_state(other)
  diff = {k: v for k, v in self_flat.items() if k not in other_flat}

  return from_flat_state(diff)


def _split_state(
  flat_state: FlatState[V],
  *filters: filterlib.Filter,
) -> tuple[State[PathParts, V], ...]:
  for i, filter_ in enumerate(filters):
    if filter_ in (..., True) and i != len(filters) - 1:
      remaining_filters = filters[i + 1 :]
      if not all(f in (..., True) for f in remaining_filters):
        raise ValueError(
          '`...` or `True` can only be used as the last filters, '
          f'got {filter_} it at index {i}.'
        )
  predicates = tuple(map(filterlib.to_predicate, filters))

  # we have n + 1 states, where n is the number of predicates
  # the last state is for values that don't match any predicate
  flat_states: tuple[list[tuple[PathParts, V]], ...] = tuple(
    [] for _ in range(len(predicates) + 1)
  )

  for path, value in flat_state:
    for i, predicate in enumerate(predicates):
      if predicate(path, value):
        flat_states[i].append((path, value))  # type: ignore[index] # mypy is wrong here?
        break
    else:
      # if we didn't break, set leaf to last state
      flat_states[-1].append((path, value))  # type: ignore[index] # mypy is wrong here?

  return tuple(from_flat_state(flat_state) for flat_state in flat_states)


def create_path_filters(state: State):
  flat_state = to_flat_state(state)
  value_paths: dict[tp.Any, set[PathParts]] = {}
  for path, value in flat_state:
    if isinstance(value, (variablelib.Variable, variablelib.VariableState)):
      value = value.value
    value_paths.setdefault(value, set()).add(path)
  return {filterlib.PathIn(*value_paths[value]): value for value in value_paths}