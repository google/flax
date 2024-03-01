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
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import functools
import typing as tp
from abc import ABCMeta
from functools import partial
from typing import Any

import jax
import jax.tree_util as jtu

from flax.experimental.nnx.nnx import reprlib, tracers

A = tp.TypeVar('A')
B = tp.TypeVar('B')
F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])
V = tp.TypeVar('V', bound='Variable[Any]')
GetValueHook = tp.Callable[['Variable[A]', A], A]
SetValueHook = tp.Callable[['Variable[A]', A], A]
CreateValueHook = tp.Callable[['Variable[A]', A], A]
AxisName = str
AxisIndex = int
AddAxisHook = tp.Callable[[V, AxisName, AxisIndex], None]
RemoveAxisHook = tp.Callable[[V, AxisName, AxisIndex], None]

VariableTypeCache: dict[str, tp.Type['Variable[tp.Any]']] = {}


class Empty:
  def __repr__(self):
    return 'Empty'

  def __eq__(self, other):
    return isinstance(other, Empty)

  def __hash__(self):
    return hash(Empty)


jtu.register_pytree_node(
  Empty,
  lambda empty: ((), None),
  lambda _0, _1: EMPTY,
)

EMPTY = Empty()


@dataclasses.dataclass
class VariableMetadata(tp.Generic[A]):
  raw_value: A
  set_value_hooks: tuple[SetValueHook[A], ...] = ()
  get_value_hooks: tuple[GetValueHook[A], ...] = ()
  create_value_hooks: tuple[CreateValueHook[A], ...] = ()
  add_axis_hooks: tuple[AddAxisHook['Variable[A]'], ...] = ()
  remove_axis_hooks: tuple[RemoveAxisHook['Variable[A]'], ...] = ()
  metadata: tp.Mapping[str, tp.Any] = dataclasses.field(default_factory=dict)


class Variable(tp.Generic[A], reprlib.Representable):
  raw_value: A
  set_value_hooks: tuple[SetValueHook[A], ...]
  get_value_hooks: tuple[GetValueHook[A], ...]
  create_value_hooks: tuple[CreateValueHook[A], ...]
  add_axis_hooks: tuple[AddAxisHook['Variable[A]'], ...]
  remove_axis_hooks: tuple[RemoveAxisHook['Variable[A]'], ...]
  _trace_state: tracers.TraceState

  def __init__(
    self,
    value: tp.Union[A, VariableMetadata[A]],
    set_value_hooks: tp.Union[
      SetValueHook[A], tp.Sequence[SetValueHook[A]]
    ] = (),
    get_value_hooks: tp.Union[
      GetValueHook[A], tp.Sequence[GetValueHook[A]]
    ] = (),
    create_value_hooks: tp.Union[
      CreateValueHook[A], tp.Sequence[CreateValueHook[A]]
    ] = (),
    add_axis_hooks: tp.Union[
      AddAxisHook['Variable[A]'], tp.Sequence[AddAxisHook['Variable[A]']]
    ] = (),
    remove_axis_hooks: tp.Union[
      RemoveAxisHook['Variable[A]'],
      tp.Sequence[RemoveAxisHook['Variable[A]']],
    ] = (),
    **metadata: tp.Any,
  ):
    vars(self)['_trace_state'] = tracers.TraceState()
    if set_value_hooks:
      if callable(set_value_hooks):
        set_value_hooks = (set_value_hooks,)
      else:
        set_value_hooks = tuple(set_value_hooks)
    else:
      set_value_hooks = ()
    if get_value_hooks:
      if callable(get_value_hooks):
        get_value_hooks = (get_value_hooks,)
      else:
        get_value_hooks = tuple(get_value_hooks)
    else:
      get_value_hooks = ()

    if create_value_hooks:
      if callable(create_value_hooks):
        create_value_hooks = (create_value_hooks,)
      else:
        create_value_hooks = tuple(create_value_hooks)
    else:
      create_value_hooks = ()

    if add_axis_hooks:
      if callable(add_axis_hooks):
        add_axis_hooks = (add_axis_hooks,)
      else:
        add_axis_hooks = tuple(add_axis_hooks)
    else:
      add_axis_hooks = ()

    if remove_axis_hooks:
      if callable(remove_axis_hooks):
        remove_axis_hooks = (remove_axis_hooks,)
      else:
        remove_axis_hooks = tuple(remove_axis_hooks)
    else:
      remove_axis_hooks = ()

    if isinstance(value, VariableMetadata):
      value_metadata = dict(value.metadata)
      if set_value_hooks and value.set_value_hooks:
        set_value_hooks = set_value_hooks + value.set_value_hooks
      elif value.set_value_hooks:
        set_value_hooks = value.set_value_hooks
      if get_value_hooks and value.get_value_hooks:
        get_value_hooks = get_value_hooks + value.get_value_hooks
      elif value.get_value_hooks:
        get_value_hooks = value.get_value_hooks
      if create_value_hooks and value.create_value_hooks:
        create_value_hooks = create_value_hooks + value.create_value_hooks
      elif value.create_value_hooks:
        create_value_hooks = value.create_value_hooks
      if add_axis_hooks and value.add_axis_hooks:
        add_axis_hooks = add_axis_hooks + value.add_axis_hooks
      elif value.add_axis_hooks:
        add_axis_hooks = value.add_axis_hooks
      if remove_axis_hooks and value.remove_axis_hooks:
        remove_axis_hooks = remove_axis_hooks + value.remove_axis_hooks
      elif value.remove_axis_hooks:
        remove_axis_hooks = value.remove_axis_hooks

      metadata.update(value_metadata)
      value = tp.cast(A, value.raw_value)

    if hasattr(self, 'on_get_value'):
      on_get_value = getattr(type(self), 'on_get_value')
      if on_get_value not in get_value_hooks:
        get_value_hooks = (on_get_value, *get_value_hooks)

    if hasattr(self, 'on_set_value'):
      on_set_value = getattr(type(self), 'on_set_value')
      if on_set_value not in set_value_hooks:
        set_value_hooks = (on_set_value, *set_value_hooks)

    if hasattr(self, 'on_create_value'):
      on_create_value = getattr(type(self), 'on_create_value')
      if on_create_value not in create_value_hooks:
        create_value_hooks = (on_create_value, *create_value_hooks)

    if hasattr(self, 'on_add_axis'):
      on_add_axis = getattr(type(self), 'on_add_axis')
      if on_add_axis not in add_axis_hooks:
        add_axis_hooks = (on_add_axis, *add_axis_hooks)

    if hasattr(self, 'on_remove_axis'):
      on_remove_axis = getattr(type(self), 'on_remove_axis')
      if on_remove_axis not in remove_axis_hooks:
        remove_axis_hooks = (on_remove_axis, *remove_axis_hooks)

    self.raw_value = value
    self.get_value_hooks = get_value_hooks
    self.set_value_hooks = set_value_hooks
    self.create_value_hooks = create_value_hooks
    self.add_axis_hooks = add_axis_hooks
    self.remove_axis_hooks = remove_axis_hooks
    vars(self).update(metadata)

    # run create_value hooks
    self.raw_value = self.create_value(self.raw_value)

  if tp.TYPE_CHECKING:

    def __getattr__(self, name: str) -> tp.Any:
      ...
  else:
    def __setattr__(self, name: str, value: Any) -> None:
      return self._setattr(name, value)

  def _setattr(self, name: str, value: tp.Any):
    if not self._trace_state.is_valid():
      raise ValueError(
        'Cannot mutate Variable from a different trace level'
      )

    object.__setattr__(self, name, value)

  def copy_from(self, other: 'Variable[A]') -> None:
    if not self.is_equivalent(other):
      raise ValueError(
        f'Cannot copy from incompatible container, '
        f'expected {type(self).__name__}, got {type(other).__name__}'
      )
    if self is other:
      return
    vars_dict = vars(self)
    vars_dict.clear()
    vars_dict.update(vars(other))

  @property
  def value(self) -> A:
    value = self.raw_value
    if self.get_value_hooks:
      for hook in self.get_value_hooks:
        value = hook(self, value)
    return value

  @value.setter
  def value(self, value: A):
    if isinstance(value, Variable):
      raise ValueError(
        'Cannot set value to a Variable, ' 'use `copy_from` method instead'
      )
    if self.set_value_hooks:
      for hook in self.set_value_hooks:
        value = hook(self, value)
    self.raw_value = value

  def create_value(self, value: A):
    for hook in self.create_value_hooks:
      value = hook(self, value)
    return value

  def add_axis(self, axis_name: AxisName, axis_index: AxisIndex):
    for hook in self.add_axis_hooks:
      hook(self, axis_name, axis_index)

  def remove_axis(self, axis_name: AxisName, axis_index: AxisIndex):
    for hook in self.remove_axis_hooks:
      hook(self, axis_name, axis_index)

  def __eq__(self, other: object) -> bool:
    return type(self) is type(other) and vars(other) == vars(self)

  @tp.overload
  def replace(self, *, value: B, **kwargs) -> 'Variable[B]':
    ...

  @tp.overload
  def replace(self, **kwargs) -> 'Variable[A]':
    ...

  def replace(self, **kwargs) -> 'Variable[tp.Any]':
    # return `value` if it is a Variable
    if 'raw_value' in kwargs and isinstance(
      value := kwargs['raw_value'], Variable
    ):
      # remove value from kwargs
      kwargs.pop('raw_value')
      if not self.is_equivalent(value):
        raise ValueError(
          'Cannot replace value from incompatible container, '
          f'expected {type(self).__name__}, got {type(value).__name__}'
        )
      # if kwargs aren't empty, recursively call replace
      # else return variable value
      if kwargs:
        return value.replace(**kwargs)
      else:
        return value

    # get and update attributes
    attributes = vars(self).copy()
    attributes.update(**kwargs)
    # return new instance with updated attributes
    obj = object.__new__(type(self))
    vars(obj).update(attributes)
    return obj

  def is_equivalent(self, other: tp.Any) -> bool:
    return type(self) is type(other)

  def copy(self: 'Variable[A]') -> 'Variable[A]':
    obj = object.__new__(type(self))
    attributes = vars(self).copy()
    attributes['_trace_state'] = tracers.TraceState()
    vars(obj).update(attributes)
    return obj

  def __nnx_repr__(self):
    yield reprlib.Object(type=type(self))
    for name, value in vars(self).items():
      if name.endswith('_hooks') or name == "_trace_state":
        continue
      yield reprlib.Attr(name, repr(value))

  def __init_subclass__(cls):
    super().__init_subclass__()

    jtu.register_pytree_with_keys(
      cls,
      partial(_variable_flatten, with_keys=True),  # type: ignore
      partial(_variable_unflatten, cls=cls),  # type: ignore
      flatten_func=partial(_variable_flatten, with_keys=False),  # type: ignore
    )

  # hooks API
  if tp.TYPE_CHECKING:

    def on_get_value(self, value: A) -> A:
      raise NotImplementedError

    def on_set_value(self, value: A) -> A:
      raise NotImplementedError

    def on_create_value(self, value: A) -> A:
      raise NotImplementedError

    def on_add_axis(self: V, axis_name: AxisName, axis_index: AxisIndex) -> V:
      raise NotImplementedError

    def on_remove_axis(
      self: V, axis_name: AxisName, axis_index: AxisIndex
    ) -> V:
      raise NotImplementedError


def _variable_flatten(x: Variable[tp.Any], *, with_keys: bool):
  attributes = vars(x).copy()
  del attributes['_trace_state']
  value = attributes.pop('raw_value')
  if with_keys:
    node = (jtu.GetAttrKey('raw_value'), value)
  else:
    node = value

  return (node,), attributes


def _variable_unflatten(
  metadata: tp.Mapping[str, tp.Any],
  children: tp.Tuple[A],
  *,
  cls: type[Variable[A]],
) -> Variable[A]:
  variable = object.__new__(cls)
  vars(variable).update(metadata, _trace_state=tracers.TraceState(), raw_value=children[0])
  return variable


jtu.register_pytree_with_keys(
  Variable,
  partial(_variable_flatten, with_keys=True),  # type: ignore
  partial(_variable_unflatten, cls=Variable),  # type: ignore
  flatten_func=partial(_variable_flatten, with_keys=False),  # type: ignore
)


class Param(Variable[A]):
  pass


class BatchStat(Variable[A]):
  pass


class Cache(Variable[A]):
  pass


class Intermediate(Variable[A]):
  pass


class Rng(Variable[jax.Array]):
  tag: str

  def __init__(self, value: jax.Array, *, tag: str, **metadata: tp.Any):
    super().__init__(value, tag=tag, **metadata)

  def on_get_value(self, value: jax.Array):
    self.raw_value, value = jax.random.split(value)
    return value


def with_metadata(
  initializer: F,
  set_value_hooks: tp.Union[SetValueHook[A], tp.Sequence[SetValueHook[A]]] = (),
  get_value_hooks: tp.Union[SetValueHook[A], tp.Sequence[SetValueHook[A]]] = (),
  create_value_hooks: tp.Union[
    CreateValueHook[A], tp.Sequence[CreateValueHook[A]]
  ] = (),
  add_axis_hooks: tp.Union[
    AddAxisHook['Variable[A]'], tp.Sequence[AddAxisHook['Variable[A]']]
  ] = (),
  remove_axis_hooks: tp.Union[
    RemoveAxisHook['Variable[A]'],
    tp.Sequence[RemoveAxisHook['Variable[A]']],
  ] = (),
  **metadata: tp.Any,
) -> F:
  if set_value_hooks:
    if callable(set_value_hooks):
      set_value_hooks = (set_value_hooks,)
    else:
      set_value_hooks = tuple(set_value_hooks)
  else:
    set_value_hooks = ()

  if get_value_hooks:
    if callable(get_value_hooks):
      get_value_hooks = (get_value_hooks,)
    else:
      get_value_hooks = tuple(get_value_hooks)
  else:
    get_value_hooks = ()

  if create_value_hooks:
    if callable(create_value_hooks):
      create_value_hooks = (create_value_hooks,)
    else:
      create_value_hooks = tuple(create_value_hooks)
  else:
    create_value_hooks = ()

  if add_axis_hooks:
    if callable(add_axis_hooks):
      add_axis_hooks = (add_axis_hooks,)
    else:
      add_axis_hooks = tuple(add_axis_hooks)
  else:
    add_axis_hooks = ()

  if remove_axis_hooks:
    if callable(remove_axis_hooks):
      remove_axis_hooks = (remove_axis_hooks,)
    else:
      remove_axis_hooks = tuple(remove_axis_hooks)
  else:
    remove_axis_hooks = ()

  @functools.wraps(initializer)
  def wrapper(*args):
    return VariableMetadata(
      initializer(*args),
      set_value_hooks=set_value_hooks,
      get_value_hooks=get_value_hooks,
      create_value_hooks=create_value_hooks,
      add_axis_hooks=add_axis_hooks,
      remove_axis_hooks=remove_axis_hooks,
      metadata=metadata,
    )

  return wrapper  # type: ignore


def variable_type(name: str) -> tp.Type[Variable[tp.Any]]:
  if name not in VariableTypeCache:
    VariableTypeCache[name] = type(name, (Variable,), {})
  return VariableTypeCache[name]


# add known variable type names
VariableTypeCache['params'] = Param
VariableTypeCache['batch_stats'] = BatchStat
VariableTypeCache['cache'] = Cache
VariableTypeCache['intermediates'] = Intermediate
