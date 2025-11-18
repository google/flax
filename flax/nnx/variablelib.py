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
import itertools as it
import threading
import typing as tp
from typing import Any
import warnings

from flax import config
from flax import errors
from flax.core import spmd as core_spmd
from flax.nnx import reprlib, tracers, visualization
from flax.typing import MISSING, Missing, SizeBytes
import jax
from jax._src.state.types import AbstractRef
import jax.experimental
from jax.experimental import hijax
import jax.tree_util as jtu
import treescope  # type: ignore[import-untyped]

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
  variable_hijax_stack: list[bool] = dataclasses.field(default_factory=list)
  eager_shard_stack: list[bool] = dataclasses.field(default_factory=list)


VARIABLE_CONTEXT = VariableContext()


class UseEagerShardContext:
  def __init__(self, prev_value: bool | None, new_value: bool):
    self.prev_value: bool | None = prev_value
    self.new_value: bool = new_value

  def __enter__(self):
    if self.prev_value is not None:
      VARIABLE_CONTEXT.eager_shard_stack.insert(-1, self.prev_value)

  def __exit__(self, exc_type, exc_value, traceback):
    VARIABLE_CONTEXT.eager_shard_stack.pop()

  def __call__(self, f: F) -> F:
    # undo eager stack change
    VARIABLE_CONTEXT.eager_shard_stack.pop()
    if self.prev_value is not None:
      VARIABLE_CONTEXT.eager_shard_stack.append(self.prev_value)

    @functools.wraps(f)
    def use_eager_sharding_wrapper(*args, **kwargs):
      VARIABLE_CONTEXT.eager_shard_stack.append(self.new_value)
      try:
        return f(*args, **kwargs)
      finally:
        VARIABLE_CONTEXT.eager_shard_stack.pop()

    return use_eager_sharding_wrapper  # type: ignore[return-value]

def using_eager_sharding() -> bool:
  """Returns whether Variables are using eager sharding by default.

  Example::

    >>> from flax import nnx
    >>> nnx.use_eager_sharding(True)
    <...>
    >>> nnx.using_eager_sharding()
    True
    >>> nnx.use_eager_sharding(False)
    <...>
    >>> nnx.using_eager_sharding()
    False


  Returns:
    A boolean indicating if Variables are using eager sharding by default.
  """
  do_eager_sharding = config.flax_always_shard_variable
  if VARIABLE_CONTEXT.eager_shard_stack:
    do_eager_sharding = VARIABLE_CONTEXT.eager_shard_stack[-1]
  return do_eager_sharding


def use_eager_sharding(value: bool, /):
  """Sets whether Variables should use eager sharding by default or not.

  Example usage::

    >>> from flax import nnx
    >>> # Use eager sharding by default
    >>> nnx.use_eager_sharding(True)
    <...>
    >>> # Variable will now use eager sharding
    >>> nnx.using_eager_sharding()
    True

  It can also be used as a context manager to temporarily
  change the default behavior for a block of code::

    >>> nnx.use_eager_sharding(False)
    <...>
    >>> with nnx.use_eager_sharding(True):
    ...   nnx.using_eager_sharding()
    True
    >>> # it will reset outside
    >>> v = nnx.Variable(jax.numpy.ones((2, 3)))
    >>> nnx.using_eager_sharding()
    False

  Args:
    value: A boolean indicating if Variables should use eager sharding by default.

  Returns:
    A context manager that resets the context to the previous value.
  """
  if VARIABLE_CONTEXT.eager_shard_stack:
    prev_value = VARIABLE_CONTEXT.eager_shard_stack[-1]
    VARIABLE_CONTEXT.eager_shard_stack[-1] = value
  else:
    prev_value = None
    VARIABLE_CONTEXT.eager_shard_stack.append(value)
  return UseEagerShardContext(prev_value, value)

def using_hijax():
  """ """
  if VARIABLE_CONTEXT.variable_hijax_stack:
    return VARIABLE_CONTEXT.variable_hijax_stack[-1]

  return config.flax_hijax_variable


def use_hijax(value: bool, /):
  """ """
  if VARIABLE_CONTEXT.variable_hijax_stack:
    prev_value = VARIABLE_CONTEXT.variable_hijax_stack[-1]
    VARIABLE_CONTEXT.variable_hijax_stack[-1] = value
  else:
    prev_value = None
    VARIABLE_CONTEXT.variable_hijax_stack.append(value)
  return UseHijaxContext(prev_value, value)


class UseHijaxContext:
  def __init__(self, prev_value: bool | None, new_value: bool):
    self.prev_value: bool | None = prev_value
    self.new_value: bool = new_value

  def __enter__(self):
    if self.prev_value is not None:
      VARIABLE_CONTEXT.variable_hijax_stack.insert(-1, self.prev_value)

  def __exit__(self, exc_type, exc_value, traceback):
    VARIABLE_CONTEXT.variable_hijax_stack.pop()

  def __call__(self, f: F) -> F:
    # undo eager stack change
    VARIABLE_CONTEXT.variable_hijax_stack.pop()
    if self.prev_value is not None:
      VARIABLE_CONTEXT.variable_hijax_stack.append(self.prev_value)

    @functools.wraps(f)
    def use_hijax_wrapper(*args, **kwargs):
      VARIABLE_CONTEXT.variable_hijax_stack.append(self.new_value)
      try:
        return f(*args, **kwargs)
      finally:
        VARIABLE_CONTEXT.variable_hijax_stack.pop()

    return use_hijax_wrapper# type: ignore[return-value]


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


PyTreeDef = tp.Any

# ---------------------------------
# hijax
# ---------------------------------

def _new_hijax_variable(var_type: type[Variable]) -> HijaxVariable:
  variable = var_type._new(None, {})
  (), treedef = jax.tree.flatten(variable)
  return new_variable_p.bind(treedef=treedef, var_type=var_type)


def _get_hijax_state(hijax_var) -> Variable:
  tys: VariableQDD = jax.experimental.cur_qdd(hijax_var)
  leaf_vals = get_variable_p.bind(hijax_var, avals=tuple(tys.leaf_avals))
  variable = jax.tree.unflatten(tys.treedef, leaf_vals)
  return variable


def _set_hijax_state(hijax_var, variable: Variable):
  leaves, treedef = jax.tree.flatten(variable)
  set_variable_p.bind(
    hijax_var, *leaves, treedef=treedef, var_type=type(variable)
  )


def _new_hijax_from_variable(variable: Variable) -> HijaxVariable:
  hijax_var = _new_hijax_variable(type(variable))
  _set_hijax_state(hijax_var, variable)
  return hijax_var


@dataclasses.dataclass(frozen=True)
class VariableQDD:
  leaf_avals: tuple[hijax.AbstractValue, ...]
  treedef: PyTreeDef

  def to_tangent_qdd(self):
    leaf_avals = tuple(a.to_tangent_aval() for a in self.leaf_avals)
    return VariableQDD(leaf_avals, self.treedef)

  def normalize(self):
    leaf_types = tuple(a.normalize() for a in self.leaf_avals)
    return VariableQDD(leaf_types, self.treedef)


class VariableEffect(jax.core.Effect):
  ...


variable_effect = VariableEffect()
hijax.control_flow_allowed_effects.add_type(VariableEffect)


class NewVariable(hijax.HiPrimitive):
  def is_high(self, *, treedef, var_type) -> bool:
    return True  # type: ignore

  def abstract_eval(self, *, treedef, var_type: type[Variable]):
    variable = var_type._new(None, {})
    leaves, treedef = jax.tree.flatten(variable)
    qdd = VariableQDD(tuple(leaves), treedef)
    return hijax.AvalQDD(AbstractVariable(var_type), qdd), {variable_effect}  # type: ignore

  def to_lojax(self, *, treedef, var_type: type[Variable]):
    return HijaxVariable._new(None, {}, var_type)

  def jvp(_, primals, tangents, *, treedef, var_type):
    raise NotImplementedError('jvp not implemented for NewHijaxVariable')

  def transpose(_, *args, treedef, var_type):
    raise NotImplementedError('transpose not implemented for NewHijaxVariable')


new_variable_p = NewVariable(f'new_variable')


class SetVariable(hijax.HiPrimitive):
  multiple_results = True

  def is_high(self, *leaf_avals, treedef, var_type) -> bool:
    return True  # type: ignore

  # TODO: upstream this to Box
  def impl(self, hijax_var: HijaxVariable, *leaves, treedef, var_type):
    variable: Variable = jax.tree.unflatten(treedef, leaves)
    object.__setattr__(hijax_var, '_raw_value', variable._raw_value)
    object.__setattr__(hijax_var, '_metadata', variable._var_metadata)
    return []

  def abstract_eval(self, hijax_var_type, *leaf_avals, treedef, var_type):
    hijax_var_type.mutable_qdd.update(VariableQDD(leaf_avals, treedef))
    return [], {variable_effect}  # TODO better typechecking...

  def to_lojax(_, hijax_var: HijaxVariable, *leaves, treedef, var_type):
    variable: Variable = jax.tree.unflatten(treedef, leaves)
    object.__setattr__(hijax_var, '_raw_value', variable._raw_value)
    object.__setattr__(hijax_var, '_metadata', variable._var_metadata)
    return []

  def jvp(_, primals, tangents, *, treedef, var_type):
    variable: Variable
    variable, *vals = primals
    variable_dot: Variable
    variable_dot, *val_dots = tangents
    if type(variable_dot._raw_value) is hijax.Zero:
      raise Exception(
        "can't differentiate Variable._set operation, "
        'did you forget jax.lax.stop_gradient?'
      )
    set_variable_p.bind(
      variable, *vals, treedef=treedef, var_type=type(variable)
    )
    set_variable_p.bind(
      variable_dot, *val_dots, treedef=treedef, var_type=type(variable_dot)
    )
    return [], []

  def transpose(_, *args, treedef, var_type):
    raise NotImplementedError('transpose not implemented for SetHijaxVariable')


set_variable_p = SetVariable(f'set_variable')


class GetVariable(hijax.HiPrimitive):
  multiple_results = True

  def abstract_eval(self, abstract_var, *, avals):
    return avals, {variable_effect}

  def to_lojax(_, hijax_var: HijaxVariable, *, avals):
    return jax.tree.leaves(hijax_var._raw_value)

  def jvp(_, primals, tangents, *, avals):
    (box,), (variable_dot,) = primals, tangents
    return (
      get_variable_p.bind(box, avals=avals),
      get_variable_p.bind(
        variable_dot, avals=tuple(a.to_tangent_aval() for a in avals)
      ),
    )

  def transpose(_, *args):
    raise NotImplementedError('transpose not implemented for GetHijaxVariable')


get_variable_p = GetVariable(f'get_variable')


# ---------------------------------
# HijaxVariable
# ---------------------------------
def _variable_has_changed(old: Variable, new: Variable) -> bool:
  old_structure = jax.tree.structure(old)
  new_structure = jax.tree.structure(new)
  if old_structure != new_structure:  # type: ignore[operator]
    return True
  old_leaves = jax.tree.leaves(old)
  new_leaves = jax.tree.leaves(new)
  return any(o is not n for o, n in zip(old_leaves, new_leaves))


def _as_hijax_property(name: str, *, get: bool, set: bool) -> property:
  """Creates a property that operates on the hijax type."""

  def _getter_wrapper(hijax_var):
    variable = _get_hijax_state(hijax_var)
    old_state = jax.tree.map(lambda x: x, variable)
    out = getattr(variable, name)
    if _variable_has_changed(old_state, variable):
      _set_hijax_state(hijax_var, variable)
    return out

  def _setter_wrapper(hijax_var, value):
    variable = _get_hijax_state(hijax_var)
    setattr(variable, name, value)
    _set_hijax_state(hijax_var, variable)

  _hijax_property = property(
    fget=_getter_wrapper if get else None,
    fset=_setter_wrapper if set else None,
  )
  return _hijax_property  # type: ignore[return]


def _as_aval_property(p: property) -> hijax.aval_property:
  """Wraps a property `p` operate on the aval type."""
  _aval_property = hijax.aval_property(fget=p.fget)
  return _aval_property  # type: ignore[return]


def _as_hijax_attribute(name: str) -> property:
  """Creates a property that operates on the hijax type."""

  def _getter_wrapper(hijax_var):
    variable = _get_hijax_state(hijax_var)
    old_state = jax.tree.map(lambda x: x, variable)
    out = getattr(variable, name)
    if _variable_has_changed(old_state, variable):
      _set_hijax_state(hijax_var, variable)
    return out

  _getter_wrapper.__name__ = name
  _hijax_property = property(fget=_getter_wrapper)

  return _hijax_property  # type: ignore[return]


def _as_hijax_method(name: str) -> tp.Any:
  """Creates a method that operates on the hijax type."""

  def hijax_method_wrapper(hijax_var, *args, **kwargs):
    variable = _get_hijax_state(hijax_var)
    old_state = jax.tree.map(lambda x: x, variable)
    method = getattr(variable, name)
    out = method(*args, **kwargs)
    if _variable_has_changed(old_state, variable):
      _set_hijax_state(hijax_var, variable)
    return out

  hijax_method_wrapper.__name__ = name

  return hijax_method_wrapper


def _as_tracer_method(name: str):
  def op(self, hijax_var, *args, **kwargs):
    variable = _get_hijax_state(hijax_var)
    old_state = jax.tree.map(lambda x: x, variable)
    out = getattr(variable, name)(*args, **kwargs)
    if _variable_has_changed(old_state, variable):
      _set_hijax_state(hijax_var, variable)
    return out

  op.__name__ = name
  return op


class HijaxVariableMeta(type):
  def __instancecheck__(self, instance):
    if super().__instancecheck__(instance):
      return True

    if isinstance(instance, jax.core.Tracer):
      ty = jax.typeof(instance)
      return isinstance(ty, AbstractVariable)
    return False

jax.Ref
class HijaxVariable(
  tp.Generic[A], reprlib.Representable, metaclass=HijaxVariableMeta
):  # type: ignore
  __slots__ = ('_raw_value', '_metadata', '_var_type')
  _raw_value: A
  _metadata: dict[str, tp.Any]
  _var_type: type[Variable[tp.Any]]

  @classmethod
  def _new(
    cls,
    value,
    metadata: dict[str, tp.Any],
    var_type: type[Variable[A]],
  ):
    hijax_var = object.__new__(cls)
    object.__setattr__(hijax_var, '_raw_value', value)
    object.__setattr__(hijax_var, '_metadata', metadata)
    object.__setattr__(hijax_var, '_var_type', var_type)
    return hijax_var

  __init__ = _as_hijax_method('__init__')

  @property
  def value(self) -> A:
    raise NotImplementedError(
      'HijaxVariable.value property is not implemented. For Variable[Array] instances use:\n\n'
      '  variable[...]\n\n'
      'For other Variable types use:\n\n'
      '  variable.get_value()\n'
    )

  @value.setter
  def value(self, new_value: A):
    raise NotImplementedError(
      'HijaxVariable.value property is not implemented. For Variable[Array] instances use:\n\n'
      '  variable[...] = new_value\n\n'
      'For other Variable types use:\n\n'
      '  variable.set_value(new_value)\n'
    )

  @property
  def var_type(self) -> type[Variable[A]]:
    return self._var_type

  _trace_state = _as_hijax_property('_trace_state', get=True, set=False)
  _can_update = _as_hijax_property('_can_update', get=True, set=False)
  _check_can_update = _as_hijax_method('_check_can_update')
  __getattr__ = _as_hijax_method('__getattr__')
  __setattr__ = _as_hijax_method('__setattr__')
  __delattr__ = _as_hijax_method('__delattr__')
  type = _as_hijax_property('type', get=True, set=False)
  is_hijax = _as_hijax_property('is_hijax', get=True, set=False)
  has_ref = _as_hijax_property('has_ref', get=True, set=False)
  is_mutable = _as_hijax_property('is_mutable', get=True, set=False)
  get_metadata = _as_hijax_method('get_metadata')
  set_metadata = _as_hijax_method('set_metadata')

  def copy_from(self, other: Variable[A] | HijaxVariable[A]) -> None:
    if isinstance(other, HijaxVariable):
      other = _get_hijax_state(other)
    variable = _get_hijax_state(self)
    variable.copy_from(other)  # type: ignore[arg-type]
    _set_hijax_state(self, variable)

  def update_from_state(self, variable_state: Variable[A] | HijaxVariable[A]):
    if isinstance(variable_state, HijaxVariable):
      variable_state = _get_hijax_state(variable_state)
    variable = _get_hijax_state(self)
    variable.update_from_state(variable_state)  # type: ignore[arg-type]
    _set_hijax_state(self, variable)

  get_raw_value = _as_hijax_method('get_raw_value')
  set_raw_value = _as_hijax_method('set_raw_value')
  set_value = _as_hijax_method('set_value')
  get_value = _as_hijax_method('get_value')
  create_value = _as_hijax_method('create_value')
  set_raw_value = _as_hijax_method('set_raw_value')
  add_axis = _as_hijax_method('add_axis')
  remove_axis = _as_hijax_method('remove_axis')
  copy = _as_hijax_method('copy')
  replace = _as_hijax_method('replace')
  to_state = _as_hijax_method('to_state')

  @classmethod
  def from_metadata(cls, value: A, metadata: dict[str, tp.Any]):
    return cls._var_type.from_metadata(value, metadata)  # type: ignore[misc]

  __nnx_repr__ = _as_hijax_method('__nnx_repr__')
  __treescope_repr__ = _as_hijax_method('__treescope_repr__')

  # --------------------------------------------
  # proxy methods
  # --------------------------------------------
  __jax_array__ = _as_hijax_method('__jax_array__')
  __getitem__ = _as_hijax_method('__getitem__')
  __setitem__ = _as_hijax_method('__setitem__')
  __delitem__ = _as_hijax_method('__delitem__')
  __call__ = _as_hijax_method('__call__')
  __len__ = _as_hijax_method('__len__')
  __iter__ = _as_hijax_method('__iter__')
  __contains__ = _as_hijax_method('__contains__')
  __add__ = _as_hijax_method('__add__')
  __sub__ = _as_hijax_method('__sub__')
  __mul__ = _as_hijax_method('__mul__')
  __matmul__ = _as_hijax_method('__matmul__')
  __truediv__ = _as_hijax_method('__truediv__')
  __floordiv__ = _as_hijax_method('__floordiv__')
  __mod__ = _as_hijax_method('__mod__')
  __divmod__ = _as_hijax_method('__divmod__')
  __pow__ = _as_hijax_method('__pow__')
  __lshift__ = _as_hijax_method('__lshift__')
  __rshift__ = _as_hijax_method('__rshift__')
  __and__ = _as_hijax_method('__and__')
  __xor__ = _as_hijax_method('__xor__')
  __or__ = _as_hijax_method('__or__')
  __radd__ = _as_hijax_method('__radd__')
  __rsub__ = _as_hijax_method('__rsub__')
  __rmul__ = _as_hijax_method('__rmul__')
  __rmatmul__ = _as_hijax_method('__rmatmul__')
  __rtruediv__ = _as_hijax_method('__rtruediv__')
  __rfloordiv__ = _as_hijax_method('__rfloordiv__')
  __rmod__ = _as_hijax_method('__rmod__')
  __rdivmod__ = _as_hijax_method('__rdivmod__')
  __rpow__ = _as_hijax_method('__rpow__')
  __rlshift__ = _as_hijax_method('__rlshift__')
  __rrshift__ = _as_hijax_method('__rrshift__')
  __rand__ = _as_hijax_method('__rand__')
  __rxor__ = _as_hijax_method('__rxor__')
  __ror__ = _as_hijax_method('__ror__')
  __iadd__ = _as_hijax_method('__iadd__')
  __isub__ = _as_hijax_method('__isub__')
  __imul__ = _as_hijax_method('__imul__')
  __imatmul__ = _as_hijax_method('__imatmul__')
  __itruediv__ = _as_hijax_method('__itruediv__')
  __ifloordiv__ = _as_hijax_method('__ifloordiv__')
  __imod__ = _as_hijax_method('__imod__')
  __ipow__ = _as_hijax_method('__ipow__')
  __ilshift__ = _as_hijax_method('__ilshift__')
  __irshift__ = _as_hijax_method('__irshift__')
  __iand__ = _as_hijax_method('__iand__')
  __ixor__ = _as_hijax_method('__ixor__')
  __ior__ = _as_hijax_method('__ior__')
  __neg__ = _as_hijax_method('__neg__')
  __pos__ = _as_hijax_method('__pos__')
  __abs__ = _as_hijax_method('__abs__')
  __invert__ = _as_hijax_method('__invert__')
  __complex__ = _as_hijax_method('__complex__')
  __int__ = _as_hijax_method('__int__')
  __float__ = _as_hijax_method('__float__')
  __index__ = _as_hijax_method('__index__')
  __round__ = _as_hijax_method('__round__')
  __trunc__ = _as_hijax_method('__trunc__')
  __floor__ = _as_hijax_method('__floor__')
  __ceil__ = _as_hijax_method('__ceil__')

  # --------------------------------------------
  # hijax interface
  # --------------------------------------------

  def cur_qdd(self):
    return self.type_state()

  @property
  def ty(self):
    return AbstractVariable(self._var_type)

  def type_state(self):
    variable = self._var_type._new(self._raw_value, self._metadata)
    leaves, treedef = jax.tree.flatten(variable)
    leaf_avals = tuple(map(jax.typeof, leaves))
    return VariableQDD(leaf_avals, treedef)


hijax.register_hitype(HijaxVariable, lambda b: b.ty)


# ---------------------------------
# AbstractVariable
# ---------------------------------
class AbstractVariable(tp.Generic[A], hijax.MutableHiType):
  __slots__ = ['_var_type']
  _var_type: type[Variable[A]]
  # forwarded to value
  var_type = hijax.aval_property(lambda self: self.aval._var_type)
  is_hijax = _as_aval_property(HijaxVariable.is_hijax)
  has_ref = _as_aval_property(HijaxVariable.has_ref)
  is_mutable = _as_aval_property(HijaxVariable.is_mutable)
  _trace_state = _as_aval_property(HijaxVariable._trace_state)
  _can_update = _as_aval_property(HijaxVariable._can_update)
  _check_can_update = hijax.aval_method(HijaxVariable._check_can_update)

  def __init__(self, var_type: type[Variable[A]]):
    object.__setattr__(self, '_var_type', var_type)

  @property
  def dtype(self):
    raise AttributeError

  @property
  def ndim(self):
    raise AttributeError

  @property
  def size(self):
    raise AttributeError

  @property
  def shape(self):
    raise AttributeError

  def __getattr__(self, name: str):
    # Forward unknown attributes to the value
    if hasattr(AbstractVariable, name):
      raise AttributeError
    if name.startswith('_'):
      raise AttributeError
    return _as_aval_property(_as_hijax_attribute(name))

  # __setattr__ supported via __getattr__
  # __delattr__ CURRENTLY NOT SUPPORTED
  type = _as_aval_property(HijaxVariable.type)
  get_metadata = hijax.aval_method(HijaxVariable.get_metadata)
  set_metadata = hijax.aval_method(HijaxVariable.set_metadata)
  copy_from = hijax.aval_method(HijaxVariable.copy_from)
  update_from_state = hijax.aval_method(HijaxVariable.update_from_state)
  get_raw_value = hijax.aval_method(HijaxVariable.get_raw_value)
  set_raw_value = hijax.aval_method(HijaxVariable.set_raw_value)
  set_value = hijax.aval_method(HijaxVariable.set_value)
  get_value = hijax.aval_method(HijaxVariable.get_value)
  create_value = hijax.aval_method(HijaxVariable.create_value)
  set_raw_value = hijax.aval_method(HijaxVariable.set_raw_value)
  add_axis = hijax.aval_method(HijaxVariable.add_axis)
  remove_axis = hijax.aval_method(HijaxVariable.remove_axis)
  replace = hijax.aval_method(HijaxVariable.replace)

  @hijax.aval_method
  def from_metadata(self, value, metadata: dict[str, tp.Any]):
    aval: AbstractVariable = self.aval  # type: ignore
    variable = aval._var_type.from_metadata(value, metadata)
    return variable

  copy = hijax.aval_method(HijaxVariable.copy)
  replace = hijax.aval_method(HijaxVariable.replace)
  to_state = hijax.aval_method(HijaxVariable.to_state)

  def __str__(self):
    return f'{self._var_type.__name__}()'

  def __repr__(self):
    return f'{self._var_type.__name__}()'

  @hijax.aval_method
  def __treescope_repr__(self, path, subtree_renderer):
    raise NotImplementedError

  # ---------------------------------
  # proxy methods
  # ---------------------------------
  __jax_array__ = hijax.aval_method(HijaxVariable.__jax_array__)
  _getitem = _as_tracer_method('__getitem__')
  _setitem = _as_tracer_method('__setitem__')
  # __delitem__ CURRENTLY NOT SUPPORTED
  # __call__ CURRENTLY NOT SUPPORTED
  _len = _as_tracer_method('__len__')
  _iter = _as_tracer_method('__iter__')
  # __contains__ CURRENTLY NOT SUPPORTED
  _add = _as_tracer_method('__add__')
  _sub = _as_tracer_method('__sub__')
  _mul = _as_tracer_method('__mul__')
  _matmul = _as_tracer_method('__matmul__')
  _truediv = _as_tracer_method('__truediv__')
  _floordiv = _as_tracer_method('__floordiv__')
  _mod = _as_tracer_method('__mod__')
  _divmod = _as_tracer_method('__divmod__')
  _pow = _as_tracer_method('__pow__')
  _lshift = _as_tracer_method('__lshift__')
  _rshift = _as_tracer_method('__rshift__')
  _and = _as_tracer_method('__and__')
  _xor = _as_tracer_method('__xor__')
  _or = _as_tracer_method('__or__')
  _radd = _as_tracer_method('__radd__')
  _rsub = _as_tracer_method('__rsub__')
  _rmul = _as_tracer_method('__rmul__')
  _rmatmul = _as_tracer_method('__rmatmul__')
  _rtruediv = _as_tracer_method('__rtruediv__')
  _rfloordiv = _as_tracer_method('__rfloordiv__')
  _rmod = _as_tracer_method('__rmod__')
  _rdivmod = _as_tracer_method('__rdivmod__')
  _rpow = _as_tracer_method('__rpow__')
  _rlshift = _as_tracer_method('__rlshift__')
  _rrshift = _as_tracer_method('__rrshift__')
  _rand = _as_tracer_method('__rand__')
  _rxor = _as_tracer_method('__rxor__')
  _ror = _as_tracer_method('__ror__')
  # _iadd CURRENTLY NOT SUPPORTED
  # _isub CURRENTLY NOT SUPPORTED
  # _imul CURRENTLY NOT SUPPORTED
  # _imatmul CURRENTLY NOT SUPPORTED
  # _itruediv CURRENTLY NOT SUPPORTED
  # _ifloordiv CURRENTLY NOT SUPPORTED
  # _imod CURRENTLY NOT SUPPORTED
  # _ipow CURRENTLY NOT SUPPORTED
  # _ilshift CURRENTLY NOT SUPPORTED
  # _irshift CURRENTLY NOT SUPPORTED
  # _iand CURRENTLY NOT SUPPORTED
  # _ixor CURRENTLY NOT SUPPORTED
  # _ior CURRENTLY NOT SUPPORTED
  _neg = _as_tracer_method('__neg__')
  _pos = _as_tracer_method('__pos__')
  _abs = _as_tracer_method('__abs__')
  _invert = _as_tracer_method('__invert__')
  _complex = _as_tracer_method('__complex__')
  _int = _as_tracer_method('__int__')
  _float = _as_tracer_method('__float__')
  _index = _as_tracer_method('__index__')
  _round = _as_tracer_method('__round__')
  _trunc = _as_tracer_method('__trunc__')
  _floor = _as_tracer_method('__floor__')
  _ceil = _as_tracer_method('__ceil__')

  # --------------------------------
  # hijax interface
  # --------------------------------
  has_qdd = True

  def __hash__(self):
    return hash((AbstractVariable, self._var_type))

  def __eq__(self, other):
    return isinstance(other, AbstractVariable) and self._var_type == other._var_type

  def str_short(self, short_dtypes=False, **_) -> str:  # type: ignore
    return f'{self._var_type.__name__}()'

  # mutable interface
  def lo_ty_qdd(self, variable_state: VariableQDD) -> list:  # type: ignore
    return [lo_ty for t in variable_state.leaf_avals for lo_ty in t.lo_ty()]

  def new_from_loval(  # type: ignore[override]
    self, variable_state: VariableQDD, *lo_vals
  ) -> HijaxVariable:
    lo_vals_ = iter(lo_vals)
    hi_vals = [
      hi_ty.raise_val(*it.islice(lo_vals_, len(hi_ty.lo_ty())))  # type: ignore
      for hi_ty in variable_state.leaf_avals
    ]
    assert next(lo_vals_, None) is None
    variable: Variable = jax.tree.unflatten(variable_state.treedef, hi_vals)
    return HijaxVariable._new(
      variable._raw_value, variable._var_metadata, self._var_type
    )  # will be mutated

  def read_loval(self, variable_state: VariableQDD, variable) -> list:  # type: ignore
    leaf_vals, treedef = jax.tree.flatten(_get_hijax_state(variable))
    assert treedef == variable_state.treedef
    return [
      lo_val
      for hi_ty, hi_val in zip(variable_state.leaf_avals, leaf_vals)
      for lo_val in hi_ty.lower_val(hi_val)
    ]  # type: ignore

  def update_from_loval(  # type: ignore[override]
    self, box_state: VariableQDD, variable, *lo_vals
  ) -> None:
    lo_vals_ = iter(lo_vals)
    hi_vals = [
      hi_ty.raise_val(*it.islice(lo_vals_, len(hi_ty.lo_ty())))  # type: ignore
      for hi_ty in box_state.leaf_avals
    ]
    assert next(lo_vals_, None) is None
    _set_hijax_state(variable, jax.tree.unflatten(box_state.treedef, hi_vals))

  def to_tangent_aval(self):
    return AbstractVariable(self._var_type)


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


class VariableMeta(type):
  def __new__(cls, cls_name, bases, attrs):
    if '__slots__' not in attrs:
      attrs['__slots__'] = ()
    return super().__new__(cls, cls_name, bases, attrs)

  def __instancecheck__(self, instance):
    if super().__instancecheck__(instance):
      return True

    if isinstance(instance, jax.core.Tracer):
      ty = jax.typeof(instance)
      if isinstance(ty, AbstractVariable):
        return issubclass(ty._var_type, self)
    if isinstance(instance, HijaxVariable):
      return issubclass(instance._var_type, self)
    return False

  if not tp.TYPE_CHECKING:

    def __call__(cls, *args, **kwargs):
      return cls._variable_meta_call(*args, **kwargs)

  def _variable_meta_call(cls, *args, **kwargs):
    variable = super().__call__(*args, **kwargs)
    if variable.is_hijax:
      return _new_hijax_from_variable(variable)
    return variable


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
  required_metadata = frozenset([
      'is_hijax', 'has_ref', 'is_mutable', 'eager_sharding'
  ])

  @property
  def var_type(self):
    return type(self)

  @property
  def is_hijax(self) -> bool:
    return self._var_metadata['is_hijax']

  @property
  def has_ref(self) -> bool:
    return self._var_metadata['has_ref']

  @property
  def is_mutable(self) -> bool:
    return self._var_metadata['is_mutable']

  @property
  def shape(self: Variable[jax.Array]) -> tuple[int, ...]:
    return self.get_value().shape

  def __init__(
      self,
      value: A | VariableMetadata[A],
      *,
      is_hijax: bool | None = None,
      has_ref: bool = False,
      is_mutable: bool = True,
      eager_sharding: bool | None = None,
      **metadata: tp.Any,
  ):
    var_t = type(self)

    if isinstance(value, VariableMetadata):
      aux_metadata = dict(value.metadata)
      if 'is_hijax' in aux_metadata:
        if is_hijax is not None and is_hijax != aux_metadata['is_hijax']:
          raise ValueError(
            'Cannot specify is_hijax both in VariableMetadata and as an '
            'argument to Variable constructor.'
          )
        is_hijax = aux_metadata.pop('is_hijax')
      if 'has_ref' in aux_metadata:
        if has_ref is not None and has_ref != aux_metadata['has_ref']:
          raise ValueError(
            'Cannot specify has_ref both in VariableMetadata and as an '
            'argument to Variable constructor.'
          )
        has_ref = aux_metadata.pop('has_ref')
      if 'is_mutable' in aux_metadata:
        if is_mutable is not None and is_mutable != aux_metadata['is_mutable']:
          raise ValueError(
            'Cannot specify is_mutable both in VariableMetadata and as an '
            'argument to Variable constructor.'
          )
        is_mutable = aux_metadata.pop('is_mutable')
      if 'eager_sharding' in aux_metadata:
        if (
          eager_sharding is not None
          and eager_sharding != aux_metadata['eager_sharding']
        ):
          raise ValueError(
            'Cannot specify eager_sharding both in VariableMetadata and as '
            'an argument to Variable constructor.'
          )
        eager_sharding = aux_metadata['eager_sharding']
      metadata.update(aux_metadata)
      value = tp.cast(A, value.raw_value)

    if is_hijax is None:
      is_hijax = using_hijax()

    if eager_sharding is None:
      eager_sharding = using_eager_sharding()

    if is_hijax and not is_mutable:
      raise ValueError(
        'Cannot set is_hijax=True and is_mutable=False simultaneously.'
      )

    if has_ref and is_hijax:
      raise ValueError(
        'Cannot set has_ref=True and is_hijax=True simultaneously.'
      )

    if has_ref and not is_mutable:
      raise ValueError(
        'Cannot set has_ref=True and is_mutable=False simultaneously.'
      )

    if any(is_array_ref(v) for v in jax.tree.leaves(value)):
      raise ValueError('Cannot pass a Ref directly into Variable constructor.')

    metadata['is_hijax'] = is_hijax
    metadata['has_ref'] = has_ref
    metadata['is_mutable'] = is_mutable
    metadata['eager_sharding'] = eager_sharding
    object.__setattr__(self, '_trace_state', tracers.TraceState())
    object.__setattr__(self, '_var_metadata', metadata)
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

    object.__setattr__(self, '_raw_value', value)
    # run create_value hook
    value = self.create_value(value)  # type: ignore
    # shard the _value if applicable
    if eager_sharding and 'sharding_names' in metadata:
      value = core_spmd.shard_value(
        value,
        metadata['sharding_names'],
        metadata.get('sharding_rules', None),
        metadata.get('mesh', None),
      )
    if has_ref:
      value = jax.new_ref(value)  # type: ignore
    object.__setattr__(self, '_raw_value', value)

  @property
  def _can_update(self) -> bool:
    """Whether the Variable can be updated in-place in the current trace context."""
    if self.is_hijax:
      return self.is_mutable
    else:
      return self.is_mutable and self._trace_state.is_valid()

  def _check_can_update(self):
    if not self.is_mutable:
      raise errors.ImmutableVariableError(
        f'Cannot mutate {type(self).__name__} as it is marked as immutable.'
      )
    if not self.is_hijax and not self._trace_state.is_valid():
      raise errors.TraceContextError(
        f'Cannot mutate {type(self).__name__} from a different trace level'
      )

  def __getattr__(self, name: str) -> tp.Any:
    if name in object.__getattribute__(self, '_var_metadata'):
      return self._var_metadata[name]
    return getattr(object.__getattribute__(self, '_raw_value'), name)

  def __setattr__(self, name: str, value: tp.Any):
    self._check_can_update()
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
    self._check_can_update()
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
  def get_metadata(self, *, exclude_required: bool = False) -> dict[str, tp.Any]:
    ...

  @tp.overload
  def get_metadata(self, name: str, default: tp.Any = MISSING) -> tp.Any:
    ...

  def get_metadata(
    self,
    name: str | None = None,
    default: tp.Any = MISSING,
    *,
    exclude_required: bool | None = None,
  ) -> tp.Any:
    """Get metadata for the Variable.

    Args:
      name: The key of the metadata element to get. If not provided, returns
        the full metadata dictionary.
      default: The default value to return if the metadata key is not found. If
        not provided and the key is not found, raises a KeyError.
    """
    if name is not None and exclude_required is not None:
      raise TypeError(
        "Cannot specify both 'name' and 'exclude_required' arguments."
      )
    metadata = self._var_metadata.copy()
    if name is None:
      if not isinstance(default, Missing):
        raise TypeError(
          "Cannot provide a default value when 'name' is not provided. "
          f'Got default={default}'
        )
      if exclude_required:
        for key in self.required_metadata:
          metadata.pop(key, None)
      return metadata
    if name not in metadata and not isinstance(default, Missing):
      return default
    return metadata[name]

  @tp.overload
  def set_metadata(self, metadata: dict[str, tp.Any], /) -> None:
    ...

  @tp.overload
  def set_metadata(self, name: str, value: tp.Any, /) -> None:
    ...

  @tp.overload
  def set_metadata(self, **metadata: tp.Any) -> None:
    ...

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
    self._check_can_update()
    if args and kwargs:
      raise TypeError(
        'Cannot mix positional and keyword arguments in set_metadata'
      )
    if len(args) == 1:
      metadata = dict(args[0])
      if 'is_hijax' not in metadata:
        metadata['is_hijax'] = self.is_hijax
      if metadata['is_hijax'] != self.is_hijax:
        raise ValueError(
          f'Cannot change `is_hijax` metadata, expected {self.is_hijax}, '
          f'got {metadata["is_hijax"]}'
        )
      if 'has_ref' not in metadata:
        metadata['has_ref'] = self.has_ref
      if metadata['has_ref'] != self.has_ref:
        raise ValueError(
          f'Cannot change `has_ref` metadata, expected {self.has_ref}, '
          f'got {metadata["has_ref"]}'
        )
      if 'is_mutable' not in metadata:
        metadata['is_mutable'] = self.is_mutable
      if metadata['is_mutable'] != self.is_mutable:
        raise ValueError(
          f'Cannot change `is_mutable` metadata, expected {self.is_mutable}, '
          f'got {metadata["is_mutable"]}'
        )
      if 'eager_sharding' not in metadata:
        metadata['eager_sharding'] = self.eager_sharding
      if metadata['eager_sharding'] != self.eager_sharding:
        raise ValueError(
          f'Cannot change `eager_sharding` metadata, expected '
          f'{self.eager_sharding}, got {metadata["eager_sharding"]}'
        )
      self._var_metadata = metadata
    elif len(args) == 2:
      name, value = args
      if name == 'is_hijax' and value != self.is_hijax:
        raise ValueError(
          f'Cannot change `is_hijax` metadata, expected {self.is_hijax}, got {value}'
        )
      if name == 'has_ref' and value != self.has_ref:
        raise ValueError(
          f'Cannot change `has_ref` metadata, expected {self.has_ref}, got {value}'
        )
      if name == 'is_mutable' and value != self.is_mutable:
        raise ValueError(
          f'Cannot change `is_mutable` metadata, expected {self.is_mutable}, got {value}'
        )
      self._var_metadata[name] = value
    elif kwargs:
      if 'is_hijax' in kwargs and kwargs['is_hijax'] != self.is_hijax:
        raise ValueError(
          f'Cannot change `is_hijax` metadata, expected {self.is_hijax}, '
          f'got {kwargs["is_hijax"]}'
        )
      if 'has_ref' in kwargs and kwargs['has_ref'] != self.has_ref:
        raise ValueError(
          f'Cannot change `has_ref` metadata, expected {self.has_ref}, '
          f'got {kwargs["has_ref"]}'
        )
      if 'is_mutable' in kwargs and kwargs['is_mutable'] != self.is_mutable:
        raise ValueError(
          f'Cannot change `is_mutable` metadata, expected {self.is_mutable}, '
          f'got {kwargs["is_mutable"]}'
        )
      self._var_metadata.update(kwargs)
    else:
      raise TypeError(
        f'set_metadata takes either 1 or 2 arguments, or at least 1 keyword argument, '
        f'got args={args}, kwargs={kwargs}'
      )

  def has_metadata(self, name: str) -> bool:
    """Check if the Variable has a metadata entry for the given name.

    Args:
      name: The key of the metadata element to check.
    Returns:
      True if the metadata entry exists, False otherwise.
    """
    return name in self._var_metadata

  def del_metadata(self, name: str) -> None:
    """Delete a metadata entry for the Variable.

    Args:
      name: The key of the metadata element to delete.
    """
    self._check_can_update()
    if name in ('is_hijax', 'has_ref', 'is_mutable'):
      raise ValueError(f'Cannot delete `{name}` metadata')
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
    self._raw_value = variable_state._raw_value

    if self._var_metadata != variable_state._var_metadata:
      metadata = variable_state.get_metadata()
      metadata['is_hijax'] = self.is_hijax
      metadata['has_ref'] = self.has_ref
      metadata['is_mutable'] = self.is_mutable
      self._var_metadata = metadata

  @tp.final
  def get_raw_value(self) -> A:
    return self._raw_value

  # @tp.final
  def set_raw_value(self, value: A, *, _unsafe_bypass_check: bool = False):
    if not _unsafe_bypass_check:
      self._check_can_update()
    self._raw_value = value

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
      "'.raw_value' setter is now deprecated. Use:\n\n"
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
    return self.get_value()

  @value.setter
  def value(self, value: A):
    warnings.warn(
      "'.value' setter is now deprecated. For Variable[Array] instances use:\n\n"
      '  variable[...] = value\n\n'
      'For other Variable types use:\n\n'
      '  variable.set_value(value)\n',
      DeprecationWarning,
      stacklevel=2,
    )
    self.set_value(value)

  def create_value(self, value: A):
    return value

  def get_value(self, *, index: tp.Any = MISSING) -> A:
    value = jax.tree.map(lambda x: x, self._raw_value)  # make a copy
    if not isinstance(index, Missing):
      if is_array_ref(value):
        value = value[index]
      elif isinstance(value, jax.Array) and index is ...:
        pass  # skip trivial access
      else:
        value = value[index]
    elif is_array_ref(value):
      value = value[...]
    if 'on_get_value' in self._var_metadata:
      value = self._var_metadata['on_get_value'](self, value)
    return value  # type: ignore

  def set_value(self, value: A, *, index: tp.Any = MISSING):
    value = jax.tree.map(lambda x: x, value)  # make a copy
    if isinstance(value, Variable):
      raise ValueError(
        'Cannot set value to a Variable, use `copy_from` method instead'
      )
    if 'on_set_value' in self._var_metadata:
      value = self._var_metadata['on_set_value'](self, value)
    # update _raw_value
    if is_array_ref(self._raw_value):
      if isinstance(index, Missing):
        self._raw_value[...] = value
      else:
        self._raw_value[index] = value
    elif isinstance(self._raw_value, jax.Array) and (
      not isinstance(index, Missing)
    ):
      # check if its a full replace to av
      if (
        index == ...
        and isinstance(value, jax.Array)
        and value.shape == self._raw_value[index].shape
        and value.dtype == self._raw_value.dtype
        and (
          getattr(value, 'sharding', None)
          == getattr(self._raw_value, 'sharding', None)
        )
      ):
        self._raw_value = value
      else:
        self._raw_value = self._raw_value.at[index].set(value)  # type: ignore
    else:
      if isinstance(index, Missing):
        self._raw_value = value
      else:
        self._raw_value[index] = value  # type: ignore

  def add_axis(self, axis_index: AxisIndex, axis_name: AxisName | None):
    if 'on_add_axis' in self._var_metadata:
      self._var_metadata['on_add_axis'](self, axis_index, axis_name)

  def remove_axis(self, axis_index: AxisIndex, axis_name: AxisName | None):
    if 'on_remove_axis' in self._var_metadata:
      self._var_metadata['on_remove_axis'](self, axis_index, axis_name)

  @tp.overload
  def copy(self, value: B, **kwargs) -> Variable[B]:
    ...

  @tp.overload
  def copy(self, **kwargs) -> Variable[A]:
    ...

  def copy(
    self,
    value: tp.Any = MISSING,
    *,
    _copy_ref: bool = True,
    **updates,
  ) -> Variable[tp.Any]:
    assert 'raw_value' not in updates

    if updates.get('has_ref', False) and updates.get('is_hijax', False):
      raise ValueError(
        'Cannot set has_ref=True and is_hijax=True simultaneously.'
      )
    if not updates.get('is_mutable', True) and updates.get('has_ref', False):
      raise ValueError(
        'Cannot set has_ref=True and is_mutable=False simultaneously.'
      )
    if updates.get('is_mutable', False) and updates.get('is_hijax', False):
      raise ValueError(
        'Cannot set is_hijax=True and is_mutable=False simultaneously.'
      )
    new_metadata = self.get_metadata() | updates
    if updates.get('has_ref', False):
      new_metadata['is_hijax'] = False
      new_metadata.pop('was_hijax', None)
    if updates.get('is_hijax', False):
      new_metadata['has_ref'] = False
      new_metadata.pop('had_ref', None)
    if not updates.get('is_mutable', True) and self.is_mutable:
      new_metadata['has_ref'] = False
      new_metadata['is_hijax'] = False
      if self.has_ref:
        new_metadata['had_ref'] = True
      if self.is_hijax:
        new_metadata['was_hijax'] = True
    if updates.get('is_mutable', False) or updates.get('has_ref', False):
      new_metadata.pop('had_ref', None)
    if updates.get('is_mutable', False) or updates.get('is_hijax', False):
      new_metadata.pop('was_hijax', None)

    if not isinstance(value, Missing):
      pass
    elif 'value' in updates:
      value = updates.pop('value')
    else:
      value = self.get_raw_value()
    if _copy_ref and is_array_ref(value):
      value = value[...]

    if _copy_ref and (
      new_metadata['has_ref']
      or (new_metadata['is_mutable'] and self.get_metadata('had_ref', False))
    ):
      value = jax.new_ref(value)
      new_metadata['has_ref'] = True
    if new_metadata['is_mutable'] and self.get_metadata('was_hijax', False):
      new_metadata['is_hijax'] = True

    obj = self.from_metadata(value, new_metadata)
    return obj

  @classmethod
  def _new(
    cls,
    value: A,
    metadata: dict[str, tp.Any],
  ) -> Variable[A]:
    obj = object.__new__(cls)
    # skip __setattr__ for trace_state initialization
    object.__setattr__(obj, '_trace_state', tracers.TraceState())
    object.__setattr__(obj, '_var_metadata', metadata)
    object.__setattr__(obj, '_raw_value', value)
    return obj

  @classmethod
  def from_metadata(
    cls,
    value: A,
    attributes: dict[str, tp.Any],
  ) -> Variable[A]:
    variable = cls._new(value, dict(attributes))
    if attributes['is_hijax']:
      variable = _new_hijax_from_variable(variable)  # type: ignore[assignment]
    return variable  # type: ignore[return-value]

  replace = copy
  to_state = copy

  def __nnx_repr__(self):
    stats = SizeBytes.from_any(self._raw_value)
    if stats:
      comment = f' # {stats}'
    else:
      comment = ''

    yield reprlib.Object(type=type(self).__name__, comment=comment)
    yield reprlib.Attr('value', self.get_value())
    for name, value in self._var_metadata.items():
      if name == 'is_hijax' and not value:
        continue
      if name == 'has_ref' and not value:
        continue
      if name == 'is_mutable' and value:
        continue
      if name == 'eager_sharding' and value:
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
    # skip __setattr__ for trace_state initialization
    object.__setattr__(self, '_trace_state', state['_trace_state'])
    object.__setattr__(self, '_var_metadata', state['_var_metadata'])
    object.__setattr__(self, '_raw_value', state['_raw_value'])

  # --------------------------------------------
  # proxy methods
  # --------------------------------------------
  @tp.overload
  def __getitem__(self: Variable[jax.Array], key) -> jax.Array:
    ...

  @tp.overload
  def __getitem__(self: Variable[dict[tp.Any, B]], key) -> B:
    ...

  @tp.overload
  def __getitem__(self: Variable[list[B]], key: int) -> B:
    ...

  @tp.overload
  def __getitem__(self: Variable[tuple[B, ...]], key: int) -> B:
    ...

  @tp.overload
  def __getitem__(self, key) -> tp.Any:
    ...

  def __getitem__(self, key):
    return self.get_value(index=key)

  def __setitem__(self, key, value) -> None:
    self.set_value(value, index=key)

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

  def __eq__(self, other) -> bool:
    if isinstance(other, Variable):
      other = other.get_value()
    return self.get_value().__eq__(other)  # type: ignore

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
      'Use `variable.value @= x` instead.'
    )

  def __itruediv__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable.value /= x` instead.'
    )

  def __ifloordiv__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable.value //= x`` instead.'
    )

  def __imod__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable.value %= x` instead.'
    )

  def __ipow__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable.value **= x`` instead.'
    )

  def __ilshift__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable.value <<= x`` instead.'
    )

  def __irshift__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable.value >>= x`` instead.'
    )

  def __iand__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable.value &= x` instead.'
    )

  def __ixor__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable.value ^= x` instead.'
    )

  def __ior__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable.value |= x` instead.'
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
