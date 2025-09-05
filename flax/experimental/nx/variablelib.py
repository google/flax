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

from __future__ import annotations

from functools import partial
import typing as tp

from flax.core import FrozenDict
import jax
import jax.numpy as jnp
from jax._src import core as jax_core
from jax._src.state.types import AbstractRef

A = tp.TypeVar('A')
B = tp.TypeVar('B')
V = tp.TypeVar('V', bound='Variable')

class Variable:
  """ """

  __slots__ = ('_value', '_metadata')

  _value: jax.Array | jax_core.MutableArray | AbstractRef
  _metadata: FrozenDict[str, tp.Hashable]

  def __init__(
    self,
    value: tp.Any,
    *,
    mutable: bool = False,
    **metadata: tp.Hashable,
  ):
    value = jnp.asarray(value)
    if mutable:
      object.__setattr__(self, '_value', jax_core.mutable_array(value))
    else:
      object.__setattr__(self, '_value', value)
    object.__setattr__(self, '_metadata', FrozenDict(metadata))

  @classmethod
  def from_array_metadata(cls, x, metadata: tp.Mapping[str, tp.Hashable]):
    variable = object.__new__(cls)
    object.__setattr__(variable, '_value', x)
    object.__setattr__(variable, '_metadata', metadata)
    return variable

  @property
  def raw_value(self):
    return self._value

  @property
  def mutable(self):
    return isinstance(
      jax.typeof(self._value), AbstractRef | jax_core.MutableArray
    )

  @property
  def shape(self) -> tuple[int, ...]:
    return self._value.shape

  @property
  def dtype(self) -> tp.Any:
    return self._value.dtype

  @property
  def metadata(self) -> FrozenDict[str, tp.Any]:
    return self._metadata

  def __repr__(self):
    type_str = type(self).__name__
    fields_strs = [
      f'value={self[...]!r}',
      f'shape={self.shape!r}',
      f'dtype={self.dtype!r}',
      f'mutable={self.mutable!r}',
    ]
    fields_strs.extend(f'{k}={v!r}' for k, v in self._metadata.items())
    return f'{type_str}(\n  ' + ',\n  '.join(fields_strs) + '\n)'

  def __getattr__(self, name: str) -> tp.Any:
    if name in self._metadata:
      return self._metadata[name]
    return getattr(self._value, name)

  if not tp.TYPE_CHECKING:

    def __setattr__(self, name: str, value: tp.Any):
      raise AttributeError

    def __delattr__(self, name: str):
      raise NotImplementedError

  def copy(
    self,
    value: tp.Any | None = None,
    *,
    mutable: bool | None = None,
    **kwargs,
  ):
    value = self[...] if value is None else value
    mutable = self.mutable if mutable is None else mutable
    metadata: dict[str, tp.Any] = dict(self._metadata)
    metadata.update(kwargs)
    return type(self)(
      value,
      mutable=mutable,
      **metadata,
    )


  # pickle support
  def __getstate__(self):
    return self._value, self._metadata

  def __setstate__(self, state):
    value, metadata = state
    object.__setattr__(self, '_value', value)
    object.__setattr__(self, '_metadata', metadata)

  # --------------------------------------------
  # proxy methods
  # --------------------------------------------

  def __getitem__(self, key) -> jax.Array:
    return self._value[key]

  def __setitem__(self, key, value) -> None:
    if not self.mutable:
      raise TypeError('Cannot set item on immutable Variable')
    self._value[key] = value  # type: ignore

  def __len__(self) -> int:
    return len(self[...])  # type: ignore

  def __iter__(self) -> tp.Iterator:
    return iter(self[...])  # type: ignore

  # --------------------------------------------

  def __jax_array__(self):
    return self[...]

  def __add__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__add__(other)  # type: ignore

  def __sub__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__sub__(other)  # type: ignore

  def __mul__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__mul__(other)  # type: ignore

  def __matmul__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__matmul__(other)  # type: ignore

  def __truediv__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__truediv__(other)  # type: ignore

  def __floordiv__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__floordiv__(other)  # type: ignore

  def __mod__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__mod__(other)  # type: ignore

  def __divmod__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__divmod__(other)  # type: ignore

  def __pow__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__pow__(other)  # type: ignore

  def __lshift__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__lshift__(other)  # type: ignore

  def __rshift__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__rshift__(other)  # type: ignore

  def __and__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__and__(other)  # type: ignore

  def __xor__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__xor__(other)  # type: ignore

  def __or__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__or__(other)  # type: ignore

  def __radd__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__radd__(other)  # type: ignore

  def __rsub__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__rsub__(other)  # type: ignore

  def __rmul__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__rmul__(other)  # type: ignore

  def __rmatmul__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__rmatmul__(other)  # type: ignore

  def __rtruediv__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__rtruediv__(other)  # type: ignore

  def __rfloordiv__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__rfloordiv__(other)  # type: ignore

  def __rmod__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__rmod__(other)  # type: ignore

  def __rdivmod__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__rdivmod__(other)  # type: ignore

  def __rpow__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__rpow__(other)  # type: ignore

  def __rlshift__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__rlshift__(other)  # type: ignore

  def __rrshift__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__rrshift__(other)  # type: ignore

  def __rand__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__rand__(other)  # type: ignore

  def __rxor__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__rxor__(other)  # type: ignore

  def __ror__(self, other) -> jax.Array:
    if isinstance(other, Variable):
      other = other[...]
    return self[...].__ror__(other)  # type: ignore

  def __neg__(self) -> jax.Array:
    return self[...].__neg__()  # type: ignore

  def __pos__(self) -> jax.Array:
    return self[...].__pos__()  # type: ignore

  def __abs__(self) -> jax.Array:
    return self[...].__abs__()  # type: ignore

  def __invert__(self) -> jax.Array:
    return self[...].__invert__()  # type: ignore

  def __complex__(self) -> jax.Array:
    return self[...].__complex__()  # type: ignore

  def __int__(self) -> jax.Array:
    return self[...].__int__()  # type: ignore

  def __float__(self) -> jax.Array:
    return self[...].__float__()  # type: ignore

  def __index__(self) -> jax.Array:
    return self[...].__index__()  # type: ignore

  def __round__(self, ndigits: int) -> jax.Array:
    return self[...].__round__(ndigits)  # type: ignore

  def __trunc__(self) -> jax.Array:
    return self[...].__trunc__()  # type: ignore

  def __floor__(self) -> jax.Array:
    return self[...].__floor__()  # type: ignore

  def __ceil__(self) -> jax.Array:
    return self[...].__ceil__()  # type: ignore

  # --------------------------------------------

  def __init_subclass__(cls):
    jax.tree_util.register_pytree_with_keys(
      cls,
      _flatten_variable_with_keys,
      partial(_unflatten_variable, cls),
      flatten_func=_flatten_variable,
    )


def _flatten_variable_with_keys(x: Variable):
  node = (jax.tree_util.GetAttrKey('value'), x._value)
  return (node,), x._metadata


def _flatten_variable(x: Variable):
  return (x._value,), x._metadata


def _unflatten_variable(
  cls: type[Variable],
  metadata: FrozenDict[str, tp.Any],
  children: tuple[tp.Any],
) -> Variable:
  variable = object.__new__(cls)
  object.__setattr__(variable, '_value', children[0])
  object.__setattr__(variable, '_metadata', metadata)
  return variable


jax.tree_util.register_pytree_with_keys(
  Variable,
  _flatten_variable_with_keys,
  partial(_unflatten_variable, Variable),
  flatten_func=_flatten_variable,
)

class Param(Variable):
  pass

class BatchStat(Variable):
  pass

class Cache(Variable):
  pass