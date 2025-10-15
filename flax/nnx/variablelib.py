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

import dataclasses
import functools
from functools import partial
import threading
import typing as tp
from typing import Any
import warnings

from flax import config
from jax._src import hijax

import jax
import treescope  # type: ignore[import-untyped]

from flax import errors
from flax.core import spmd as core_spmd
from flax.nnx import reprlib, tracers, visualization
from flax.typing import MISSING, Missing, SizeBytes
import jax.tree_util as jtu
from jax._src.state.types import AbstractRef

A = tp.TypeVar('A')
B = tp.TypeVar('B')
C = tp.TypeVar('C')
F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])
P = tp.TypeVar('P', bound=property)
V = tp.TypeVar('V', bound='Variable[Any]')
GetValueHook = tp.Callable[['Variable[A]', A], A]
SetValueHook = tp.Callable[['Variable[A]', A], A]
CreateValueHook = tp.Callable[['Variable[A]', A], A]
AxisName = str
AxisIndex = int
AddAxisHook = tp.Callable[[V, AxisIndex, AxisName | None], None]
RemoveAxisHook = tp.Callable[[V, AxisIndex, AxisName | None], None]

# JAX array refs were renamed a few times between JAX v0.7.0 and v0.8.0.
# The following ensures we avoid an ImportError or DeprecationWarning.
if hasattr(jax, 'new_ref') and hasattr(jax, 'Ref'):
  # JAX v0.7.2 or newer
  from jax import Ref
elif hasattr(jax, 'array_ref') and hasattr(jax, 'ArrayRef'):
  # JAX v0.7.1
  from jax import ArrayRef as Ref  # type: ignore[import-untyped]
else:
  # JAX v0.7.0 or older
  from jax.experimental import MutableArray as Ref

@dataclasses.dataclass
class VariableContext(threading.local):
  variable_mode_stack: list[tp.Literal['lojax', 'hijax', 'ref']] = (
    dataclasses.field(default_factory=list)
  )


VARIABLE_CONTEXT = VariableContext()


def current_variable_mode() -> tp.Literal['lojax', 'hijax', 'ref']:
  """ """
  if VARIABLE_CONTEXT.variable_mode_stack:
    return VARIABLE_CONTEXT.variable_mode_stack[-1]
  match config.flax_variable_mode:
    case 'lojax' | 'hijax' | 'ref':
      return config.flax_variable_mode
    case other:
      raise ValueError(f'Unrecognized variable mode: {other}')


def variable_mode(value: tp.Literal['lojax', 'hijax', 'ref'], /):
  """ """
  if VARIABLE_CONTEXT.variable_mode_stack:
    prev_value = VARIABLE_CONTEXT.variable_mode_stack[-1]
    VARIABLE_CONTEXT.variable_mode_stack[-1] = value
  else:
    prev_value = None
    VARIABLE_CONTEXT.variable_mode_stack.append(value)
  return ModeContext(prev_value, value)


class ModeContext:
  def __init__(
    self,
    prev_value: tp.Literal['lojax', 'hijax', 'ref'] | None,
    new_value: tp.Literal['lojax', 'hijax', 'ref'],
  ):
    self.prev_value: tp.Literal['lojax', 'hijax', 'ref'] | None = prev_value
    self.new_value: tp.Literal['lojax', 'hijax', 'ref'] = new_value

  def __enter__(self):
    if self.prev_value is not None:
      VARIABLE_CONTEXT.variable_mode_stack.insert(-1, self.prev_value)

  def __exit__(self, exc_type, exc_value, traceback):
    VARIABLE_CONTEXT.variable_mode_stack.pop()

  def __call__(self, f: F) -> F:
    # undo eager stack change
    VARIABLE_CONTEXT.variable_mode_stack.pop()
    if self.prev_value is not None:
      VARIABLE_CONTEXT.variable_mode_stack.append(self.prev_value)

    @functools.wraps(f)
    def set_variable_mode_wrapper(*args, **kwargs):
      VARIABLE_CONTEXT.variable_mode_stack.append(self.new_value)
      try:
        return f(*args, **kwargs)
      finally:
        VARIABLE_CONTEXT.variable_mode_stack.pop()

    return set_variable_mode_wrapper  # type: ignore[return-value]


def is_array_ref(x) -> tp.TypeGuard[Ref]:
  return isinstance(x, jax.Array | AbstractRef | Ref) and isinstance(
    jax.typeof(x), AbstractRef | Ref
  )


@dataclasses.dataclass
class VariableMetadata(tp.Generic[A]):
  raw_value: A
  set_value_hooks: tuple[SetValueHook[A], ...] = ()
  get_value_hooks: tuple[GetValueHook[A], ...] = ()
  create_value_hooks: tuple[CreateValueHook[A], ...] = ()
  add_axis_hooks: tuple[AddAxisHook[Variable[A]], ...] = ()
  remove_axis_hooks: tuple[RemoveAxisHook[Variable[A]], ...] = ()
  metadata: tp.Mapping[str, tp.Any] = dataclasses.field(default_factory=dict)


# --------------------------------------------
# Variable
# --------------------------------------------


def _variable_operator(name: str) -> tp.Callable[[Variable[A], tp.Any], A]:
  def variable_operator_method(self, other):
    value = self.get_value()
    if isinstance(other, Variable):
      other = other.get_value()
    return getattr(value, name)(other)

  variable_operator_method.__name__ = name
  return variable_operator_method


def _variable_unary_operator(name: str) -> tp.Callable[[Variable[A]], A]:
  def variable_unary_operator_method(self):
    value = self.get_value()
    return getattr(value, name)()

  variable_unary_operator_method.__name__ = name
  return variable_unary_operator_method

@dataclasses.dataclass(frozen=True)
class BoxRepr(reprlib.Representable):
  box: hijax.Box

  def __nnx_repr__(self):
    yield reprlib.Object(type=type(self.box).__name__)
    yield reprlib.Attr('value', self.box.get())

class VariableMeta(type):
  def __new__(cls, cls_name, bases, attrs):
    if '__slots__' not in attrs:
      attrs['__slots__'] = ()
    return super().__new__(cls, cls_name, bases, attrs)


class Variable(tp.Generic[A], reprlib.Representable, metaclass=VariableMeta):
  """The base class for all ``Variable`` types. Create custom ``Variable``
  types by subclassing this class. Numerous NNX graph functions can filter
  for specific ``Variable`` types, for example, :func:`split`, :func:`state`,
  :func:`pop`, and :func:`State.filter`.

  Example usage::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> class CustomVariable(nnx.Variable):
    ...   pass

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
    ...     self.custom_variable = CustomVariable(jnp.ones((1, 3)))
    ...   def __call__(self, x):
    ...     return self.linear(x) + self.custom_variable
    >>> model = Model(rngs=nnx.Rngs(0))

    >>> linear_variables = nnx.state(model, nnx.Param)
    >>> jax.tree.map(jnp.shape, linear_variables)
    State({
      'linear': {
        'bias': Param(
          value=(3,)
        ),
        'kernel': Param(
          value=(2, 3)
        )
      }
    })

    >>> custom_variable = nnx.state(model, CustomVariable)
    >>> jax.tree.map(jnp.shape, custom_variable)
    State({
      'custom_variable': CustomVariable(
        value=(1, 3)
      )
    })

    >>> variables = nnx.state(model)
    >>> jax.tree.map(jnp.shape, variables)
    State({
      'custom_variable': CustomVariable(
        value=(1, 3)
      ),
      'linear': {
        'bias': Param(
          value=(3,)
        ),
        'kernel': Param(
          value=(2, 3)
        )
      }
    })
  """

  __slots__ = ('_raw_value', '_trace_state', '_var_metadata')
  _raw_value: A
  _trace_state: tracers.TraceState
  _var_metadata: dict[str, tp.Any]

  @property
  def mode(self) -> tp.Literal['lojax', 'hijax', 'ref']:
    if isinstance(self._raw_value, hijax.Box):
      return 'hijax'
    elif is_array_ref(self._raw_value):
      return 'ref'
    else:
      return 'lojax'

  @property
  def shape(self: Variable[jax.Array]) -> tuple[int, ...]:
    return self.get_value().shape

  def __init__(
    self,
    value: A | VariableMetadata[A],
    *,
    mode: tp.Literal['lojax', 'hijax', 'ref'] | None = None,
    eager_sharding: bool | None = None,
    **metadata: tp.Any,
  ):
    var_t = type(self)
    object.__setattr__(self, '_trace_state', tracers.TraceState())

    if isinstance(value, VariableMetadata):
      aux_metadata = dict(value.metadata)
      if 'mode' in aux_metadata:
        if mode is not None and mode != aux_metadata['mode']:
          raise ValueError(
            'Cannot specify mode both in VariableMetadata and as an '
            'argument to Variable constructor.'
          )
        mode = aux_metadata.pop('mode')
      if 'eager_sharding' in aux_metadata:
        if (
          eager_sharding is not None
          and eager_sharding != aux_metadata['eager_sharding']
        ):
          raise ValueError(
            'Cannot specify eager_sharding both in VariableMetadata and as '
            'an argument to Variable constructor.'
          )
        eager_sharding = aux_metadata.pop('eager_sharding')
      metadata.update(aux_metadata)
      value = tp.cast(A, value.raw_value)

    if any(is_array_ref(v) for v in jax.tree.leaves(value)):
      raise ValueError('Cannot pass a Ref directly into Variable constructor.')

    object.__setattr__(self, '_raw_value', value)

    if hasattr(var_t, 'on_get_value') and 'on_get_value' not in metadata:
      metadata['on_get_value'] = var_t.on_get_value

    if hasattr(var_t, 'on_set_value') and 'on_set_value' not in metadata:
      metadata['on_set_value'] = var_t.on_set_value

    if hasattr(var_t, 'on_create_value') and 'on_create_value' not in metadata:
      metadata['on_create_value'] = var_t.on_create_value

    if hasattr(var_t, 'on_add_axis') and 'on_add_axis' not in metadata:
      metadata['on_add_axis'] = var_t.on_add_axis

    if hasattr(var_t, 'on_remove_axis') and 'on_remove_axis' not in metadata:
      metadata['on_remove_axis'] = var_t.on_remove_axis

    if 'sharding' in metadata:
      metadata['sharding_names'] = metadata.pop('sharding')

    # run create_value hooks
    if 'on_create_value' in metadata:
      value = metadata['on_create_value'](self, value)

    if eager_sharding is None:
      eager_sharding = config.flax_always_shard_variable

    if mode is None:
      mode = current_variable_mode()

    metadata['mode'] = mode
    object.__setattr__(self, '_var_metadata', metadata)
    object.__setattr__(self, '_raw_value', value)
    # run create_value hook
    value = self.create_value(value)  # type: ignore
    # shard the _value if applicable
    if eager_sharding and 'sharding_names' in metadata:
      metadata['eager_sharding'] = eager_sharding
      value = core_spmd.shard_value(
        value,
        metadata['sharding_names'],
        metadata.get('sharding_rules', None),
        metadata.get('mesh', None),
      )

    if mode == 'hijax':
      value = hijax.Box(value)  # type: ignore
    elif mode == 'ref':
      value = jax.new_ref(value)  # type: ignore

    object.__setattr__(self, '_raw_value', value)

  @property
  def trace_state(self) -> tracers.TraceState:
    return self._trace_state

  def __getattr__(self, name: str) -> tp.Any:
    if name in object.__getattribute__(self, '_var_metadata'):
      return self._var_metadata[name]
    return getattr(self._raw_value, name)

  def __setattr__(self, name: str, value: tp.Any):
    if not self._trace_state.is_valid():
      raise errors.TraceContextError(
        f'Cannot mutate {type(self).__name__} from a different trace level'
      )
    try:
      object.__setattr__(self, name, value)
    except AttributeError as e:
      raise AttributeError(
        f'Cannot set attribute {name}. '
        f'To set Variable metadata use either:\n\n'
        f'  variable.set_metadata({name}=value)\n\nor\n\n'
        f"  variable.set_metadata('{name}', value)"
      ) from e

  def __delattr__(self, name: str):
    if not self._trace_state.is_valid():
      raise errors.TraceContextError(
        f'Cannot mutate {type(self).__name__} from a different trace level'
      )
    try:
      object.__delattr__(self, name)
    except AttributeError as e:
      raise AttributeError(
        f'Cannot delete attribute {name}. '
        f'To delete Variable metadata use:\n\n'
        f"  variable.del_metadata('{name}')"
      ) from e

  # NOTE(cgarciae): adding this for backward compatibility with VariableState
  @property
  def type(self):
    """The type of the variable."""

    return type(self)

  @tp.overload
  def get_metadata(self) -> dict[str, tp.Any]: ...
  @tp.overload
  def get_metadata(self, name: str, default: tp.Any = MISSING) -> tp.Any: ...
  def get_metadata(
    self, name: str | None = None, default: tp.Any = MISSING
  ) -> tp.Any:
    """Get metadata for the Variable.

    Args:
      name: The key of the metadata element to get. If not provided, returns
        the full metadata dictionary.
      default: The default value to return if the metadata key is not found. If
        not provided and the key is not found, raises a KeyError.
    """
    metadata = self._var_metadata.copy()
    if name is None:
      return metadata
    if name not in metadata and not isinstance(default, Missing):
      return default
    return metadata[name]

  @tp.overload
  def set_metadata(self, metadata: dict[str, tp.Any], /) -> None: ...
  @tp.overload
  def set_metadata(self, name: str, value: tp.Any, /) -> None: ...
  @tp.overload
  def set_metadata(self, **metadata: tp.Any) -> None: ...
  def set_metadata(self, *args, **kwargs) -> None:
    """Set metadata for the Variable.

    `set_metadata` can be called in 3 ways:

    1. By passing a dictionary of metadata as the first argument, this will replace
      the entire Variable's metadata.
    2. By passing a name and value as the first two arguments, this will set
      the metadata entry for the given name to the given value.
    3. By using keyword arguments, this will update the Variable's metadata
      with the provided key-value pairs.
    """
    if not self._trace_state.is_valid():
      raise errors.TraceContextError(
        f'Cannot mutate {type(self).__name__} from a different trace level'
      )
    if args and kwargs:
      raise TypeError(
        'Cannot mix positional and keyword arguments in set_metadata'
      )
    if len(args) == 1:
      metadata = dict(args[0])
      if 'mode' not in metadata:
        raise ValueError('metadata is missing required key `mode` key')
      if metadata['mode'] != self.mode:
        raise ValueError(
          f'Cannot change `mode` metadata, expected {self.mode}, '
          f'got {metadata["mode"]}'
        )
      self._var_metadata = metadata
    elif len(args) == 2:
      name, value = args
      if name == 'mode' and value != self.mode:
        raise ValueError(
          f'Cannot change `mode` metadata, expected {self.mode}, got {value}'
        )
      self._var_metadata[name] = value
    elif kwargs:
      if 'mode' in kwargs and kwargs['mode'] != self.mode:
        raise ValueError(
          f'Cannot change `mode` metadata, expected {self.mode}, '
          f'got {kwargs["mode"]}'
        )
      self._var_metadata.update(kwargs)
    else:
      raise TypeError(
        f'set_metadata takes either 1 or 2 arguments, or at least 1 keyword argument, '
        f'got args={args}, kwargs={kwargs}'
      )

  def del_metadata(self, name: str) -> None:
    """Delete a metadata entry for the Variable.

    Args:
      name: The key of the metadata element to delete.
    """
    if not self._trace_state.is_valid():
      raise errors.TraceContextError(
        f'Cannot mutate {type(self).__name__} from a different trace level'
      )
    if name == 'mode':
      raise ValueError('Cannot delete `mode` metadata')
    del self._var_metadata[name]

  def copy_from(self, other: Variable[A]) -> None:
    if type(self) is not type(other):
      raise ValueError(
        f'Cannot copy from incompatible container, '
        f'expected {type(self).__name__}, got {type(other).__name__}'
      )
    if self is other:
      return
    self._raw_value = other._raw_value
    self._var_metadata.clear()
    self._var_metadata.update(other.get_metadata())

  def update_from_state(self, variable_state: Variable[A]):
    object.__setattr__(self, '_raw_value', variable_state._raw_value)

    if self._var_metadata != variable_state._var_metadata:
      metadata = variable_state.get_metadata().copy()
      metadata['mode'] = self.mode
      object.__setattr__(self, '_var_metadata', metadata)

  @tp.final
  def get_raw_value(self) -> A:
    return self._raw_value

  @tp.final
  def set_raw_value(self, value: A):
    if not self._trace_state.is_valid():
      raise errors.TraceContextError(
        f'Cannot mutate {type(self).__name__} from a different trace level'
      )
    object.__setattr__(self, '_raw_value', value)

  @property
  def raw_value(self) -> A:
    warnings.warn(
      "'.raw_value' access is now deprecated. Use:\n\n"
      '  variable.get_raw_value()\n',
      DeprecationWarning,
      stacklevel=2,
    )
    return self.get_raw_value()

  @raw_value.setter
  def raw_value(self, value: A):
    warnings.warn(
      "'.raw_value' access is now deprecated. Use:\n\n"
      '  variable.set_raw_value(value)\n',
      DeprecationWarning,
      stacklevel=2,
    )
    self.set_raw_value(value)

  @property
  def value(self) -> A:
    warnings.warn(
      "'.value' access is now deprecated. For Variable[Array] instances use:\n\n"
      '  variable[...]\n\n'
      'For other Variable types use:\n\n'
      '  variable.get_value()\n',
      DeprecationWarning,
      stacklevel=2,
    )
    value = self._raw_value
    if is_array_ref(value):
      value = value[...]

    return self.get_value()

  @value.setter
  def value(self, value: A):
    warnings.warn(
      "'.value' access is now deprecated. For Variable[Array] instances use:\n\n"
      '  variable[...] = value\n\n'
      'For other Variable types use:\n\n'
      '  variable.set_value(value)\n',
      DeprecationWarning,
      stacklevel=2,
    )
    self.set_value(value)

  def create_value(self, value: A):
    return value

  def get_value(self) -> A:
    if isinstance(self._raw_value, hijax.Box):
      value = self._raw_value.get()
    else:
      value = jax.tree.map(lambda x: x, self._raw_value)  # make a copy
    if 'on_get_value' in self._var_metadata:
      value = self._var_metadata['on_get_value'](self, value)
    return value

  def set_value(self, value: A):
    value = jax.tree.map(lambda x: x, value)  # make a copy
    if isinstance(value, Variable):
      raise ValueError(
        'Cannot set value to a Variable, use `copy_from` method instead'
      )
    if 'on_set_value' in self._var_metadata:
      value = self._var_metadata['on_set_value'](self, value)

    if isinstance(self._raw_value, hijax.Box):
      self._raw_value.set(value)
    else:
      object.__setattr__(self, '_raw_value', value)

  def add_axis(self, axis_index: AxisIndex, axis_name: AxisName | None):
    if 'on_add_axis' in self._var_metadata:
      self._var_metadata['on_add_axis'](self, axis_index, axis_name)

  def remove_axis(self, axis_index: AxisIndex, axis_name: AxisName | None):
    if 'on_remove_axis' in self._var_metadata:
      self._var_metadata['on_remove_axis'](self, axis_index, axis_name)

  @tp.overload
  def replace(self, value: B, **kwargs) -> Variable[B]: ...

  @tp.overload
  def replace(self, **kwargs) -> Variable[A]: ...

  def replace(self, value: tp.Any = Missing, **kwargs) -> Variable[tp.Any]:
    if value is not Missing:
      kwargs['value'] = value

    if 'is_hijax' in kwargs and kwargs['is_hijax'] != self.is_hijax:
      raise ValueError(
        f'Cannot change `is_hijax` metadata, expected {self.is_hijax}, '
        f'got {kwargs["is_hijax"]}'
      )

    if 'raw_value' in kwargs:
      raise RuntimeError

    # return `value` if it is a Variable
    if 'value' in kwargs and isinstance(value := kwargs['value'], Variable):
      # remove value from kwargs
      kwargs.pop('value')
      if type(self) is not type(value):
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
    # return new instance with updated attributes
    obj = object.__new__(type(self))
    object.__setattr__(obj, '_trace_state', self._trace_state)
    object.__setattr__(obj, '_raw_value', kwargs.pop('value'))
    object.__setattr__(obj, '_var_metadata', self.get_metadata() | kwargs)
    return obj

  @classmethod
  def _new(
    cls,
    value: A,
    metadata: dict[str, tp.Any],
  ) -> Variable[A]:
    obj = object.__new__(cls)
    object.__setattr__(obj, '_trace_state', tracers.TraceState())
    object.__setattr__(obj, '_raw_value', value)
    object.__setattr__(obj, '_var_metadata', metadata)
    return obj

  @classmethod
  def from_metadata(
    cls,
    value: A,
    attributes: dict[str, tp.Any],
  ) -> Variable[A]:
    obj = cls._new(value, dict(attributes))
    return obj  # type: ignore[return-value]

  def copy(self: Variable[A]) -> Variable[A]:
    obj = object.__new__(type(self))
    object.__setattr__(obj, '_trace_state', tracers.TraceState())
    object.__setattr__(obj, '_raw_value', self._raw_value)
    object.__setattr__(obj, '_var_metadata', self.get_metadata().copy())
    return obj

  to_state = copy

  def __nnx_repr__(self):
    stats = SizeBytes.from_any(self._raw_value)
    if stats:
      comment = f' # {stats}'
    else:
      comment = ''

    yield reprlib.Object(type=type(self).__name__, comment=comment)
    if isinstance(self._raw_value, hijax.Box):
      yield reprlib.Attr('value', BoxRepr(self._raw_value))
    else:
      yield reprlib.Attr('value', self._raw_value)
    for name, value in self._var_metadata.items():
      if name == 'is_hijax' and value is False:
        continue
      yield reprlib.Attr(name, value)

  def __treescope_repr__(self, path, subtree_renderer):
    size_bytes = SizeBytes.from_any(self.get_value())
    if size_bytes:
      stats_repr = f' # {size_bytes}'
      first_line_annotation = treescope.rendering_parts.comment_color(
        treescope.rendering_parts.text(f'{stats_repr}')
      )
    else:
      first_line_annotation = None

    children = {'value': self.get_value(), **self._var_metadata}
    return visualization.render_object_constructor(
      object_type=type(self),
      attributes=children,
      path=path,
      subtree_renderer=subtree_renderer,
      first_line_annotation=first_line_annotation,
    )

  # hooks API
  if tp.TYPE_CHECKING:

    def on_get_value(self, value: A) -> A: ...

    def on_set_value(self, value: A) -> A: ...

    def on_create_value(self, value: A) -> A: ...

    def on_add_axis(
      self: V, axis_index: AxisIndex, axis_name: AxisName | None
    ) -> V: ...

    def on_remove_axis(
      self: V, axis_index: AxisIndex, axis_name: AxisName | None
    ) -> V: ...

  def __jax_array__(self):
    return self.get_value()

  # pickle support
  def __getstate__(self):
    return {
      '_raw_value': self._raw_value,
      '_trace_state': self._trace_state,
      '_var_metadata': self._var_metadata,
    }

  def __setstate__(self, state):
    object.__setattr__(self, '_raw_value', state['_raw_value'])
    object.__setattr__(self, '_trace_state', state['_trace_state'])
    object.__setattr__(self, '_var_metadata', state['_var_metadata'])

  # --------------------------------------------
  # proxy methods
  # --------------------------------------------
  @tp.overload
  def __getitem__(self: Variable[jax.Array], key) -> jax.Array: ...
  @tp.overload
  def __getitem__(self: Variable[dict[tp.Any, B]], key) -> B: ...
  @tp.overload
  def __getitem__(self: Variable[list[B]], key: int) -> B: ...
  @tp.overload
  def __getitem__(self: Variable[tuple[B, ...]], key: int) -> B: ...
  @tp.overload
  def __getitem__(self, key) -> tp.Any: ...
  def __getitem__(self, key):
    return self.get_value()[key]  # type: ignore

  def __setitem__(self, key, item_value) -> None:
    value = self.get_value()
    if isinstance(value, jax.Array):
      value = value.at[key].set(item_value)  # type: ignore[assignment]
    else:
      value[key] = item_value  # type: ignore
    self.set_value(value)  # type: ignore

  def __delitem__(self, key) -> None:
    value = self.get_value()
    del value[key]  # type: ignore
    self.set_value(value)  # type: ignore

  def __call__(self, *args, **kwargs) -> tp.Any:
    return self.get_value()(*args, **kwargs)  # type: ignore

  def __len__(self) -> int:
    return len(self.get_value())  # type: ignore

  def __iter__(self) -> tp.Iterator:
    return iter(self.get_value())  # type: ignore

  def __contains__(self, item) -> bool:
    return item in self.get_value()  # type: ignore

  # binary operators
  __add__ = _variable_operator('__add__')
  __sub__ = _variable_operator('__sub__')
  __mul__ = _variable_operator('__mul__')
  __matmul__ = _variable_operator('__matmul__')
  __truediv__ = _variable_operator('__truediv__')
  __floordiv__ = _variable_operator('__floordiv__')
  __mod__ = _variable_operator('__mod__')
  __pow__ = _variable_operator('__pow__')
  __lshift__ = _variable_operator('__lshift__')
  __rshift__ = _variable_operator('__rshift__')
  __and__ = _variable_operator('__and__')
  __xor__ = _variable_operator('__xor__')
  __or__ = _variable_operator('__or__')
  __radd__ = _variable_operator('__radd__')
  __rsub__ = _variable_operator('__rsub__')
  __rmul__ = _variable_operator('__rmul__')
  __rmatmul__ = _variable_operator('__rmatmul__')
  __rtruediv__ = _variable_operator('__rtruediv__')
  __rfloordiv__ = _variable_operator('__rfloordiv__')
  __rmod__ = _variable_operator('__rmod__')
  __rpow__ = _variable_operator('__rpow__')
  __rlshift__ = _variable_operator('__rlshift__')
  __rrshift__ = _variable_operator('__rrshift__')
  __rand__ = _variable_operator('__rand__')
  __rxor__ = _variable_operator('__rxor__')
  __ror__ = _variable_operator('__ror__')

  # in-place operators
  def __iadd__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable[...] += x` instead.'
    )

  def __isub__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable[...] -= x` instead.'
    )

  def __imul__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable[...] *= x` instead.'
    )

  def __imatmul__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable[...] @= x` instead.'
    )

  def __itruediv__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable[...] /= x` instead.'
    )

  def __ifloordiv__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable[...] //= x` instead.'
    )

  def __imod__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable[...] %= x` instead.'
    )

  def __ipow__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable[...] **= x` instead.'
    )

  def __ilshift__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable[...] <<= x` instead.'
    )

  def __irshift__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable[...] >>= x` instead.'
    )

  def __iand__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable[...] &= x` instead.'
    )

  def __ixor__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable[...] ^= x` instead.'
    )

  def __ior__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable[...] |= x` instead.'
    )

  __neg__ = _variable_unary_operator('__neg__')
  __pos__ = _variable_unary_operator('__pos__')
  __abs__ = _variable_unary_operator('__abs__')
  __invert__ = _variable_unary_operator('__invert__')
  __complex__ = _variable_unary_operator('__complex__')
  __int__ = _variable_unary_operator('__int__')
  __float__ = _variable_unary_operator('__float__')
  __index__ = _variable_unary_operator('__index__')
  __trunc__ = _variable_unary_operator('__trunc__')
  __floor__ = _variable_unary_operator('__floor__')
  __ceil__ = _variable_unary_operator('__ceil__')

  def __round__(self, ndigits: int = 0) -> A:
    return self.get_value().__round__(ndigits)  # type: ignore

  # --------------------------------------------

  def __init_subclass__(cls) -> None:
    if '__slots__' not in vars(cls):
      cls.__slots__ = ()  # type: ignore[assignment]
    super().__init_subclass__()
    jax.tree_util.register_pytree_with_keys(
      cls,
      flatten_with_keys=_variable_flatten_with_keys,
      unflatten_func=partial(_variable_unflatten, cls),  # type: ignore
      flatten_func=_variable_flatten,
    )


def _variable_flatten_with_keys(x: Variable[tp.Any]):
  metadata = tuple(sorted(x._var_metadata.items()))
  node = (jtu.GetAttrKey('value'), x._raw_value)
  return (node,), metadata


def _variable_flatten(x: Variable[tp.Any]):
  metadata = tuple(sorted(x._var_metadata.items()))
  return (x._raw_value,), metadata


def _variable_unflatten(
  cls: type[Variable[tp.Any]],
  static: tuple[tuple[str, tp.Any], ...],
  children: tuple[tp.Any],
):
  return cls._new(children[0], dict(static))


jax.tree_util.register_pytree_with_keys(
  Variable,
  flatten_with_keys=_variable_flatten_with_keys,
  unflatten_func=partial(_variable_unflatten, Variable),  # type: ignore
  flatten_func=_variable_flatten,
)

VariableState = Variable


class Param(Variable[A]):
  """The canonical learnable parameter. All learnable parameters
  in NNX layer modules will have the ``Param`` :class:`Variable`
  type::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> layer = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    >>> jax.tree.map(jnp.shape, nnx.state(layer))
    State({
      'bias': Param(
        value=(3,)
      ),
      'kernel': Param(
        value=(2, 3)
      )
    })
  """

  pass


class BatchStat(Variable[A]):
  """The mean and variance batch statistics stored in
  the :class:`BatchNorm` layer. Note, these are not the
  learnable scale and bias parameters, but rather the
  running average statistics that are typically used
  during post-training inference::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> layer = nnx.BatchNorm(3, rngs=nnx.Rngs(0))
    >>> jax.tree.map(jnp.shape, nnx.state(layer))
    State({
      'bias': Param(
        value=(3,)
      ),
      'mean': BatchStat(
        value=(3,)
      ),
      'scale': Param(
        value=(3,)
      ),
      'var': BatchStat(
        value=(3,)
      )
    })
  """

  pass


class Cache(Variable[A]):
  """Autoregressive cache in :class:`MultiHeadAttention`::

  >>> from flax import nnx
  >>> import jax, jax.numpy as jnp

  >>> layer = nnx.MultiHeadAttention(
  ...   num_heads=2,
  ...   in_features=3,
  ...   qkv_features=6,
  ...   out_features=6,
  ...   decode=True,
  ...   rngs=nnx.Rngs(0),
  ... )

  >>> layer.init_cache((1, 3))
  >>> jax.tree.map(jnp.shape, nnx.state(layer, nnx.Cache))
  State({
    'cache_index': Cache(
      value=()
    ),
    'cached_key': Cache(
      value=(1, 2, 3)
    ),
    'cached_value': Cache(
      value=(1, 2, 3)
    )
  })
  """

  pass


class Intermediate(Variable[A]):
  """:class:`Variable` type that is typically used for
  :func:`Module.sow`::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.linear1 = nnx.Linear(2, 3, rngs=rngs)
    ...     self.linear2 = nnx.Linear(3, 4, rngs=rngs)
    ...   def __call__(self, x):
    ...     x = self.linear1(x)
    ...     self.sow(nnx.Intermediate, 'i', x)
    ...     x = self.linear2(x)
    ...     return x
    >>> model = Model(rngs=nnx.Rngs(0))

    >>> x = jnp.ones((1, 2))
    >>> y = model(x)
    >>> jax.tree.map(jnp.shape, nnx.state(model, nnx.Intermediate))
    State({
      'i': Intermediate(
        value=((1, 3),)
      )
    })
  """

  pass


class Perturbation(Intermediate[A]):
  """:class:`Variable` type that is typically used for
  :func:`Module.perturb`::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.linear1 = nnx.Linear(2, 3, rngs=rngs)
    ...     self.linear2 = nnx.Linear(3, 4, rngs=rngs)
    ...   def __call__(self, x):
    ...     x = self.linear1(x)
    ...     x = self.perturb('i', x)
    ...     x = self.linear2(x)
    ...     return x
    >>> model = Model(rngs=nnx.Rngs(0))

    >>> x = jnp.ones((1, 2))
    >>> y = model(x)
    >>> jax.tree.map(jnp.shape, nnx.state(model, nnx.Perturbation))
    State({
      'i': Perturbation(
        value=(1, 3)
      )
    })
  """

  pass


def with_metadata(
  initializer: F,
  set_value_hooks: tp.Union[SetValueHook[A], tp.Sequence[SetValueHook[A]]] = (),
  get_value_hooks: tp.Union[SetValueHook[A], tp.Sequence[SetValueHook[A]]] = (),
  create_value_hooks: tp.Union[
    CreateValueHook[A], tp.Sequence[CreateValueHook[A]]
  ] = (),
  add_axis_hooks: tp.Union[
    AddAxisHook[Variable[A]], tp.Sequence[AddAxisHook[Variable[A]]]
  ] = (),
  remove_axis_hooks: tp.Union[
    RemoveAxisHook[Variable[A]],
    tp.Sequence[RemoveAxisHook[Variable[A]]],
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


###################################################
### Variable type/class <-> string name mapping ###
###################################################
# Assumption: the mapping is 1-1 and unique.

VariableTypeCache: dict[str, tp.Type[Variable[tp.Any]]] = {}


def variable_type_from_name(
  name: str,
  /,
  *,
  base: type[Variable[tp.Any]] = Variable,
  allow_register: bool = False,
) -> tp.Type[Variable[tp.Any]]:
  """Given a Linen-style collection name, get or create its NNX Variable class."""
  if name not in VariableTypeCache:
    if not allow_register:
      raise ValueError(
        f'Name {name} is not registered in the registry. '
        'To register a new name, use register_variable_name() '
        'or set allow_register=True.'
      )
    VariableTypeCache[name] = type(name, (base,), {})
  return VariableTypeCache[name]


def variable_name_from_type(
  typ: tp.Type[Variable[tp.Any]], /, *, allow_register: bool = False
) -> str:
  """Given an NNX Variable type, get its Linen-style collection name.

  Should output the exact inversed result of `variable_type_from_name()`."""
  for name, t in VariableTypeCache.items():
    if typ == t:
      return name

  if not allow_register:
    raise ValueError(
      f'Type {typ} is not registered in the registry. '
      'To register a new type, use register_variable_name() '
      'or set allow_register=True.'
    )
  name = typ.__name__
  if name in VariableTypeCache:
    raise ValueError(
      'Name {name} is already registered in the registry as {VariableTypeCache[name]}. '
      'It cannot be linked with this type {typ}.'
    )
  register_variable_name(name, typ)
  return name


@tp.overload
def register_variable_name(
  name: str,
  typ: type[Variable[tp.Any]],
  *,
  overwrite: bool = False,
) -> type[Variable[tp.Any]]: ...


@tp.overload
def register_variable_name(
  name: str,
  *,
  overwrite: bool = False,
) -> tp.Callable[[type[Variable[tp.Any]]], type[Variable[tp.Any]]]: ...


def register_variable_name(
  name: str,
  typ: type[Variable[A]] | Missing = MISSING,
  *,
  overwrite=False,
) -> type[Variable[A]] | tp.Callable[[type[Variable[A]]], type[Variable[A]]]:
  """Register a pair of Linen collection name and its NNX type."""
  if isinstance(typ, Missing):
    return partial(register_variable_name, name, overwrite=overwrite)
  typ = tp.cast(type[Variable[A]], typ)
  if not overwrite and name in VariableTypeCache:
    raise ValueError(
      f'Name {name} already mapped to type {VariableTypeCache[name]}. '
      'To overwrite, call register_variable_name() with `overwrite=True`.'
    )
  VariableTypeCache[name] = typ
  return typ


# add known variable type names
register_variable_name('params', Param)
register_variable_name('batch_stats', BatchStat)
register_variable_name('cache', Cache)
register_variable_name('intermediates', Intermediate)
register_variable_name('perturbations', Perturbation)
