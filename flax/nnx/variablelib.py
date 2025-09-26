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
from flax import config
from jax._src import hijax
from jax._src import core as jax_core
from jax._src import effects
from jax._src import ad_util
import itertools as it

import jax
import treescope  # type: ignore[import-untyped]

from flax import errors
from flax.core import spmd as core_spmd
from flax.nnx import reprlib, tracers, visualization
from flax.typing import Missing, SizeBytes
import jax.tree_util as jtu
import jax.numpy as jnp
from jax._src.state.types import AbstractRef

A = tp.TypeVar('A')
B = tp.TypeVar('B')
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

if hasattr(jax, 'array_ref') and hasattr(jax, 'ArrayRef'):
  from jax import array_ref  # type: ignore[import-untyped]
  from jax import ArrayRef  # type: ignore[import-untyped]
elif not tp.TYPE_CHECKING:
  from jax._src.core import mutable_array, MutableArray

  array_ref = mutable_array
  ArrayRef = MutableArray
  # temp workaround future proof correct repr
  MutableArray.__repr__ = lambda self: 'ArrayRef' + repr(self._buf)[5:]  # type: ignore[method-assign]


@dataclasses.dataclass
class VariableContext(threading.local):
  mutable_variable_stack: list[bool] = dataclasses.field(default_factory=list)
  hijax_variable_stack: list[bool | tp.Literal['mutable']] = dataclasses.field(
    default_factory=list
  )


VARIABLE_CONTEXT = VariableContext()


def using_refs() -> bool:
  """Returns whether Variables are using ArrayRefs by default.

  Example::

    >>> from flax import nnx
    ...
    >>> nnx.using_refs()
    False
    >>> nnx.use_refs(True)
    <...>
    >>> nnx.using_refs()
    True
    >>> nnx.use_refs(False)
    <...>
    >>> nnx.using_refs()
    False


  Returns:
    A boolean indicating if Variables are using ArrayRefs by default.
  """
  if VARIABLE_CONTEXT.mutable_variable_stack:
    return VARIABLE_CONTEXT.mutable_variable_stack[-1]
  else:
    return config.flax_array_ref


def use_refs(value: bool, /):
  """Sets whether Variables should use ArrayRefs by default or not.

  Example usage::

    >>> from flax import nnx
    >>> # Use ArrayRefs by default
    >>> nnx.use_refs(True)
    <...>
    >>> # Variable will now use ArrayRefs
    >>> v = nnx.Variable(jax.numpy.ones((2, 3)))
    >>> v.has_ref
    True
    >>> v.raw_value
    Ref(...)
    >>> nnx.use_refs(False)
    <...>

  It can also be used as a context manager to temporarily
  change the default behavior for a block of code::

    >>> nnx.use_refs(False)
    <...>
    >>> with nnx.use_refs(True):
    ...   v = nnx.Variable(jax.numpy.ones((2, 3)))
    ...   v.has_ref
    True
    >>> # it will reset outside
    >>> v = nnx.Variable(jax.numpy.ones((2, 3)))
    >>> v.has_ref
    False

  Additionally, it can be used as a function decorator to change
  the default behavior for a function::

    >>> @nnx.use_refs(True)
    ... def create_var():
    ...   return nnx.Variable(jax.numpy.ones((2, 3)))
    >>> v = create_var()
    >>> v.has_ref
    True

  Args:
    value: A boolean indicating if Variables should use ArrayRefs by default.

  Returns:
    A context manager that resets the context to the previous value.
  """
  if VARIABLE_CONTEXT.mutable_variable_stack:
    prev_value = VARIABLE_CONTEXT.mutable_variable_stack[-1]
    VARIABLE_CONTEXT.mutable_variable_stack[-1] = value
  else:
    prev_value = None
    VARIABLE_CONTEXT.mutable_variable_stack.append(value)
  return UseRefsContext(prev_value, value)


class UseRefsContext:
  def __init__(self, prev_value: bool | None, new_value: bool):
    self.prev_value: bool | None = prev_value
    self.new_value: bool = new_value

  def __enter__(self):
    if self.prev_value is not None:
      VARIABLE_CONTEXT.mutable_variable_stack.insert(-1, self.prev_value)

  def __exit__(self, exc_type, exc_value, traceback):
    VARIABLE_CONTEXT.mutable_variable_stack.pop()

  def __call__(self, f: F) -> F:
    # undo eager stack change
    VARIABLE_CONTEXT.mutable_variable_stack.pop()
    if self.prev_value is not None:
      VARIABLE_CONTEXT.mutable_variable_stack.append(self.prev_value)

    @functools.wraps(f)
    def use_refs_wrapper(*args, **kwargs):
      VARIABLE_CONTEXT.mutable_variable_stack.append(self.new_value)
      try:
        return f(*args, **kwargs)
      finally:
        VARIABLE_CONTEXT.mutable_variable_stack.pop()

    return use_refs_wrapper  # type: ignore[return]


# @contextlib.contextmanager
# def _clean_mutable_arrays_context(prev_value: bool | None):
#   if prev_value is not None:
#     VARIABLE_CONTEXT.mutable_variable_stack.insert(-1, prev_value)
#   try:
#     yield
#   finally:
#     VARIABLE_CONTEXT.mutable_variable_stack.pop()


def using_hijax() -> bool | tp.Literal['mutable']:
  """ """
  if VARIABLE_CONTEXT.hijax_variable_stack:
    return VARIABLE_CONTEXT.hijax_variable_stack[-1]
  elif config.flax_hijax_variable:
    return 'mutable'
  else:
    return False


def use_hijax(value: bool | tp.Literal['mutable'], /):
  """ """
  if VARIABLE_CONTEXT.hijax_variable_stack:
    prev_value = VARIABLE_CONTEXT.hijax_variable_stack[-1]
    VARIABLE_CONTEXT.hijax_variable_stack[-1] = value
  else:
    prev_value = None
    VARIABLE_CONTEXT.hijax_variable_stack.append(value)
  return UseHijaxContext(prev_value, value)


class UseHijaxContext:
  def __init__(
    self,
    prev_value: bool | tp.Literal['mutable'] | None,
    new_value: bool | tp.Literal['mutable'],
  ):
    self.prev_value: bool | tp.Literal['mutable'] | None = prev_value
    self.new_value: bool | tp.Literal['mutable'] = new_value

  def __enter__(self):
    if self.prev_value is not None:
      VARIABLE_CONTEXT.hijax_variable_stack.insert(-1, self.prev_value)

  def __exit__(self, exc_type, exc_value, traceback):
    VARIABLE_CONTEXT.hijax_variable_stack.pop()

  def __call__(self, f: F) -> F:
    # undo eager stack change
    VARIABLE_CONTEXT.hijax_variable_stack.pop()
    if self.prev_value is not None:
      VARIABLE_CONTEXT.hijax_variable_stack.append(self.prev_value)

    @functools.wraps(f)
    def use_hijax_wrapper(*args, **kwargs):
      VARIABLE_CONTEXT.hijax_variable_stack.append(self.new_value)
      try:
        return f(*args, **kwargs)
      finally:
        VARIABLE_CONTEXT.hijax_variable_stack.pop()

    return use_hijax_wrapper  # type: ignore[return]


def is_array_ref(x) -> tp.TypeGuard[ArrayRef]:
  return isinstance(x, jax.Array | AbstractRef | ArrayRef) and isinstance(
    jax.typeof(x), AbstractRef | ArrayRef
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


def _new_mutable_hijax_variable(
  var_type: type[Variable],
) -> MutableHijaxVariable:
  variable = var_type._new(None, {})
  (), treedef = jax.tree.flatten(variable)
  return new_mutable_hijax_variable_p.bind(treedef=treedef, var_type=var_type)


def _get_mutable_hijax_state(mutable_hijax_var) -> Variable:
  tys: MutableHijaxVariableQDD = jax_core.cur_qdd(mutable_hijax_var)
  leaf_vals = get_mutable_hijax_variable_p.bind(
    mutable_hijax_var, avals=tuple(tys.leaf_avals)
  )
  variable = jax.tree.unflatten(tys.treedef, leaf_vals)
  return variable


def _set_mutable_hijax_state(mutable_hijax_var, variable: Variable):
  leaves, treedef = jax.tree.flatten(variable)
  set_mutable_hijax_variable_p.bind(
    mutable_hijax_var, *leaves, treedef=treedef, var_type=type(variable)
  )


def _new_mutable_hijax_from_variable(
  variable: Variable,
) -> MutableHijaxVariable:
  mutable_hijax_var = _new_mutable_hijax_variable(type(variable))
  _set_mutable_hijax_state(mutable_hijax_var, variable)
  return mutable_hijax_var


def _get_mutable_hijax_var_type(mutable_hijax_var) -> type[Variable]:
  tys: MutableHijaxVariableQDD = jax_core.cur_qdd(mutable_hijax_var)
  return tys.var_type


## Variable implementation


@dataclasses.dataclass(frozen=True)
class MutableHijaxVariableQDD(jax_core.QuasiDynamicData):
  leaf_avals: tuple[jax_core.AbstractValue, ...]
  treedef: PyTreeDef
  var_type: type[Variable]

  def to_tangent_qdd(self):
    leaf_avals = tuple(a.to_tangent_aval() for a in self.leaf_avals)
    return MutableHijaxVariableQDD(leaf_avals, self.treedef, self.var_type)

  def normalize(self):
    leaf_types = tuple(a.normalize() for a in self.leaf_avals)
    return MutableHijaxVariableQDD(leaf_types, self.treedef, self.var_type)


class MutableHijaxVariableEffect(effects.Effect): ...


mutable_hijax_variable_effect = MutableHijaxVariableEffect()
effects.control_flow_allowed_effects.add_type(MutableHijaxVariableEffect)


class NewMutableHijaxVariable(hijax.HiPrimitive):
  def is_high(self, *, treedef, var_type) -> bool:
    return True  # type: ignore

  def abstract_eval(self, *, treedef, var_type: type[Variable]):
    variable = var_type._new(None, {})
    leaves, treedef = jax.tree.flatten(variable)
    qdd = MutableHijaxVariableQDD(tuple(leaves), treedef, var_type)
    return jax_core.AvalQDD(var_type._abstract_hijax_type(), qdd), {
      mutable_hijax_variable_effect
    }

  def to_lojax(self, *, treedef, var_type: type[Variable]):
    return var_type._mutable_hijax_type._new(None, {}, var_type)

  def jvp(_, primals, tangents, *, treedef):
    raise NotImplementedError('jvp not implemented for NewMutableHijaxVariable')

  def transpose(_, *args, treedef):
    raise NotImplementedError(
      'transpose not implemented for NewMutableHijaxVariable'
    )


new_mutable_hijax_variable_p = NewMutableHijaxVariable(f'new_variable')


class SetMutableHijaxVariable(hijax.HiPrimitive):
  multiple_results = True

  def is_high(self, *leaf_avals, treedef, var_type) -> bool:
    return True  # type: ignore

  def abstract_eval(self, hijax_var_type, *leaf_avals, treedef, var_type):
    hijax_var_type.mutable_qdd.update(
      MutableHijaxVariableQDD(leaf_avals, treedef, var_type)
    )
    return [], {mutable_hijax_variable_effect}  # TODO better typechecking...

  def to_lojax(
    _, mutable_hijax_var: MutableHijaxVariable, *leaves, treedef, var_type
  ):
    variable: Variable = jax.tree.unflatten(treedef, leaves)
    object.__setattr__(mutable_hijax_var, '_raw_value', variable.raw_value)
    object.__setattr__(mutable_hijax_var, '_metadata', variable._var_metadata)
    return []

  def jvp(_, primals, tangents, *, treedef):
    variable, *vals = primals
    variable_dot, *val_dots = tangents
    if type(variable_dot) is ad_util.Zero:
      raise Exception(
        "can't differentiate Variable._set operation, "
        'did you forget jax.lax.stop_gradient?'
      )
    set_mutable_hijax_variable_p.bind(
      variable, *vals, treedef=treedef, var_type=type(variable)
    )
    set_mutable_hijax_variable_p.bind(
      variable_dot, *val_dots, treedef=treedef, var_type=type(variable_dot)
    )
    return [], []

  def transpose(_, *args, treedef):
    raise NotImplementedError(
      'transpose not implemented for SetMutableHijaxVariable'
    )


set_mutable_hijax_variable_p = SetMutableHijaxVariable(f'set_variable')


class GetMutableHijaxVariable(hijax.HiPrimitive):
  multiple_results = True

  def abstract_eval(self, variable_ty, *, avals):
    return avals, {mutable_hijax_variable_effect}

  def to_lojax(_, mutable_hijax_var: MutableHijaxVariable, *, avals):
    return jax.tree.leaves(mutable_hijax_var._raw_value)

  def jvp(_, primals, tangents, *, avals):
    (box,), (variable_dot,) = primals, tangents
    return (
      get_mutable_hijax_variable_p.bind(box, avals=avals),
      get_mutable_hijax_variable_p.bind(
        variable_dot, avals=tuple(a.to_tangent_aval() for a in avals)
      ),
    )

  def transpose(_, *args):
    raise NotImplementedError(
      'transpose not implemented for GetMutableHijaxVariable'
    )


get_mutable_hijax_variable_p = GetMutableHijaxVariable(f'get_variable')


# ---------------------------------
# HijaxVariable implementations
# ---------------------------------
def _variable_has_changed(old: Variable, new: Variable) -> bool:
  old_structure = jax.tree.structure(old)
  new_structure = jax.tree.structure(new)
  if old_structure != new_structure:
    return True
  old_leaves = jax.tree.leaves(old)
  new_leaves = jax.tree.leaves(new)
  return any(o is not n for o, n in zip(old_leaves, new_leaves))


def _as_hijax_property(name: str, *, get: bool, set: bool) -> property:
  """Creates a property that operates on the hijax type."""

  def _getter_wrapper(mutable_hijax_var):
    variable = _get_mutable_hijax_state(mutable_hijax_var)
    old_state = jax.tree.map(lambda x: x, variable)
    out = getattr(variable, name)
    if _variable_has_changed(old_state, variable):
      _set_mutable_hijax_state(mutable_hijax_var, variable)
    return out

  def _setter_wrapper(mutable_hijax_var, value):
    variable = _get_mutable_hijax_state(mutable_hijax_var)
    setattr(variable, name, value)
    _set_mutable_hijax_state(mutable_hijax_var, variable)

  _hijax_property = property(
    fget=_getter_wrapper if get else None,
    fset=_setter_wrapper if set else None,
  )
  return _hijax_property  # type: ignore[return]


def _as_aval_property(p: property) -> jax_core.aval_property:
  """Wraps a property `p` operate on the aval type."""

  _aval_property = jax_core.aval_property(
    fget=p.fget,
    fset=p.fset,
  )
  return _aval_property  # type: ignore[return]


def _as_hijax_attribute(name: str) -> property:
  """Creates a property that operates on the hijax type."""

  def _getter_wrapper(mutable_hijax_var):
    variable = _get_mutable_hijax_state(mutable_hijax_var)
    old_state = jax.tree.map(lambda x: x, variable)
    out = getattr(variable, name)
    if _variable_has_changed(old_state, variable):
      _set_mutable_hijax_state(mutable_hijax_var, variable)
    return out

  def _setter_wrapper(mutable_hijax_var, value):
    variable = _get_mutable_hijax_state(mutable_hijax_var)
    setattr(variable, name, value)
    _set_mutable_hijax_state(mutable_hijax_var, variable)

  _getter_wrapper.__name__ = name
  _setter_wrapper.__name__ = name

  _hijax_property = property(
    fget=_getter_wrapper,
    fset=_setter_wrapper,
  )
  return _hijax_property  # type: ignore[return]


def _as_hijax_method(name: str) -> tp.Any:
  """Creates a method that operates on the hijax type."""

  def hijax_method_wrapper(mutable_hijax_var, *args, **kwargs):
    variable = _get_mutable_hijax_state(mutable_hijax_var)
    old_state = jax.tree.map(lambda x: x, variable)
    method = getattr(variable, name)
    out = method(*args, **kwargs)
    if _variable_has_changed(old_state, variable):
      _set_mutable_hijax_state(mutable_hijax_var, variable)
    return out

  hijax_method_wrapper.__name__ = name

  return hijax_method_wrapper


def _as_tracer_method(name: str):
  def op(self, mutable_hijax_var, *args, **kwargs):
    variable = _get_mutable_hijax_state(mutable_hijax_var)
    old_state = jax.tree.map(lambda x: x, variable)
    out = getattr(variable, name)(*args, **kwargs)
    if _variable_has_changed(old_state, variable):
      _set_mutable_hijax_state(mutable_hijax_var, variable)
    return out

  op.__name__ = name
  return op


class MutableHijaxVariableMeta(type):
  def __instancecheck__(self, instance):
    if super().__instancecheck__(instance):
      return True

    if isinstance(instance, jax_core.Tracer):
      ty = jax_core.typeof(instance)
      if isinstance(ty, AbstractMutableHijaxVariable):
        return issubclass(ty._var_type, self._var_type)  # type: ignore
    return False

  def __isubclasscheck__(self, subclass):
    if super().__subclasscheck__(subclass):
      return True
    if issubclass(subclass, AbstractMutableHijaxVariable):
      return issubclass(subclass._var_type, self._var_type)  # type: ignore
    return False


class MutableHijaxVariable(tp.Generic[A], metaclass=MutableHijaxVariableMeta):  # type: ignore
  __slots__ = ('_raw_value', '_metadata')
  _raw_value: A
  _metadata: dict[str, tp.Any]
  _var_type: type[Variable[tp.Any]]
  is_hijax: bool | tp.Literal['mutable'] = 'mutable'

  raw_value = _as_hijax_attribute('raw_value')

  def __new__(cls, *args, **kwargs):
    return _new_mutable_hijax_variable(cls._var_type)

  __init__ = _as_hijax_method('__init__')
  trace_state = _as_hijax_property('trace_state', get=True, set=False)
  __getattr__ = _as_hijax_method('__getattr__')
  __setattr__ = _as_hijax_method('__setattr__')
  __delattr__ = _as_hijax_method('__delattr__')
  type = _as_hijax_property('type', get=True, set=False)
  has_ref = property(lambda self: False)
  get_metadata = _as_hijax_method('get_metadata')
  set_metadata = _as_hijax_method('set_metadata')

  def copy_from(self, other: Variable[A] | MutableHijaxVariable[A]) -> None:
    if isinstance(other, MutableHijaxVariable):
      other = _get_mutable_hijax_state(other)
    variable = _get_mutable_hijax_state(self)
    variable.copy_from(other)
    _set_mutable_hijax_state(self, variable)

  def update_from_state(
    self, variable_state: Variable[A] | MutableHijaxVariable[A]
  ):
    if isinstance(variable_state, MutableHijaxVariable):
      variable_state = _get_mutable_hijax_state(variable_state)
    variable = _get_mutable_hijax_state(self)
    variable.update_from_state(variable_state)
    _set_mutable_hijax_state(self, variable)

  value = _as_hijax_property('value', get=True, set=True)
  add_axis = _as_hijax_method('add_axis')
  remove_axis = _as_hijax_method('remove_axis')

  def replace(self, *args, **kwargs) -> MutableHijaxVariable:
    variable = _get_mutable_hijax_state(self)
    variable = variable.replace(*args, **kwargs)
    mutable_hijax_var = _new_mutable_hijax_from_variable(variable)
    return mutable_hijax_var

  def from_metadata(
    self, value: A, metadata: dict[str, tp.Any]
  ) -> MutableHijaxVariable:
    mutable_hijax_var = _new_mutable_hijax_variable(self._var_type)
    variable = self._var_type._new(value, metadata)
    _set_mutable_hijax_state(mutable_hijax_var, variable)
    return mutable_hijax_var

  def copy(self) -> MutableHijaxVariable:
    variable = _get_mutable_hijax_state(self)
    mutable_hijax_var = _new_mutable_hijax_from_variable(variable)
    return mutable_hijax_var

  to_state = copy

  def __str__(self):
    variable = _get_mutable_hijax_state(self)
    variable.set_metadata(is_hijax='mutable')
    return str(variable)

  def __repr__(self):
    variable = _get_mutable_hijax_state(self)
    variable.set_metadata(is_hijax='mutable')
    return repr(variable)

  def __treescope_repr__(self, path, subtree_renderer):
    variable = _get_mutable_hijax_state(self)
    variable.set_metadata(is_hijax='mutable')
    return subtree_renderer(variable, path)

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

  @classmethod
  def _new(
    cls,
    value,
    metadata: dict[str, tp.Any],
    var_type: type[Variable[A]],
  ):
    mutable_hijax_var = object.__new__(cls)
    object.__setattr__(mutable_hijax_var, '_raw_value', value)
    object.__setattr__(mutable_hijax_var, '_metadata', metadata)
    return mutable_hijax_var

  def cur_qdd(self):
    return self.type_state()

  @property
  def ty(self):
    return self._var_type._abstract_hijax_type()

  def type_state(self):
    variable = self._var_type._new(self._raw_value, self._metadata)
    leaves, treedef = jax.tree.flatten(variable)
    leaf_avals = tuple(map(jax_core.typeof, leaves))
    return MutableHijaxVariableQDD(leaf_avals, treedef, self._var_type)


class AbstractMutableHijaxVariable(tp.Generic[A], hijax.MutableHiType):
  _var_type: type[Variable[tp.Any]]
  # forwarded to value
  is_hijax = jax_core.aval_property(lambda self: 'mutable')
  raw_value = _as_aval_property(MutableHijaxVariable.raw_value)
  trace_state = _as_aval_property(MutableHijaxVariable.trace_state)

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
    if hasattr(AbstractMutableHijaxVariable, name):
      raise AttributeError
    if name.startswith('_'):
      raise AttributeError
    return _as_aval_property(_as_hijax_attribute(name))

  # __setattr__ supported via __getattr__
  # __delattr__ CURRENTLY NOT SUPPORTED
  type = _as_aval_property(MutableHijaxVariable.type)
  has_ref = jax_core.aval_property(lambda self: False)
  get_metadata = jax_core.aval_method(MutableHijaxVariable.get_metadata)
  set_metadata = jax_core.aval_method(MutableHijaxVariable.set_metadata)
  copy_from = jax_core.aval_method(MutableHijaxVariable.copy_from)
  update_from_state = jax_core.aval_method(
    MutableHijaxVariable.update_from_state
  )
  value = _as_aval_property(MutableHijaxVariable.value)
  add_axis = jax_core.aval_method(MutableHijaxVariable.add_axis)
  remove_axis = jax_core.aval_method(MutableHijaxVariable.remove_axis)
  replace = jax_core.aval_method(MutableHijaxVariable.replace)

  @classmethod
  def from_metadata(cls, value, metadata: dict[str, tp.Any]):
    variable = cls._var_type._new(value, metadata)
    mutable_hijax_var = _new_mutable_hijax_from_variable(variable)
    return mutable_hijax_var

  copy = jax_core.aval_method(MutableHijaxVariable.copy)
  to_state = copy

  def __str__(self):
    return f'{self._var_type.__name__}()'

  def __repr__(self):
    return f'{self._var_type.__name__}()'

  @jax_core.aval_method
  def __treescope_repr__(self, path, subtree_renderer):
    raise NotImplementedError

  # ---------------------------------
  # proxy methods
  # ---------------------------------
  __jax_array__ = jax_core.aval_method(MutableHijaxVariable.__jax_array__)
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
    return hash(AbstractMutableHijaxVariable)

  def __eq__(self, other):
    return isinstance(other, AbstractMutableHijaxVariable)

  def str_short(self, short_dtypes=False, **_) -> str:  # type: ignore
    return f'MutableHijaxVariable({self._var_type.__name__}())'

  # mutable interface
  def lo_ty_qdd(self, variable_state: MutableHijaxVariableQDD) -> list:  # type: ignore
    return [lo_ty for t in variable_state.leaf_avals for lo_ty in t.lo_ty()]

  def new_from_loval(
    self, variable_state: MutableHijaxVariableQDD, *lo_vals
  ) -> MutableHijaxVariable:  # type: ignore
    lo_vals_ = iter(lo_vals)
    hi_vals = [
      hi_ty.raise_val(*it.islice(lo_vals_, len(hi_ty.lo_ty())))  # type: ignore
      for hi_ty in variable_state.leaf_avals
    ]
    assert next(lo_vals_, None) is None
    variable: Variable = jax.tree.unflatten(variable_state.treedef, hi_vals)
    return self._var_type._mutable_hijax_type._new(
      variable.raw_value, variable._var_metadata, type(variable)
    )  # will be mutated

  def read_loval(
    self, variable_state: MutableHijaxVariableQDD, variable
  ) -> list:  # type: ignore
    leaf_vals, treedef = jax.tree.flatten(_get_mutable_hijax_state(variable))
    assert treedef == variable_state.treedef
    return [
      lo_val
      for hi_ty, hi_val in zip(variable_state.leaf_avals, leaf_vals)
      for lo_val in hi_ty.lower_val(hi_val)
    ]  # type: ignore

  def update_from_loval(
    self, box_state: MutableHijaxVariableQDD, variable, *lo_vals
  ) -> None:  # type: ignore
    lo_vals_ = iter(lo_vals)
    hi_vals = [
      hi_ty.raise_val(*it.islice(lo_vals_, len(hi_ty.lo_ty())))  # type: ignore
      for hi_ty in box_state.leaf_avals
    ]
    assert next(lo_vals_, None) is None
    _set_mutable_hijax_state(
      variable, jax.tree.unflatten(box_state.treedef, hi_vals)
    )

  def to_tangent_aval(self):
    return type(self)()


hijax.register_hitype(MutableHijaxVariable, lambda b: b.ty)


class VariableMeta(type):
  def __new__(cls, cls_name, bases, attrs):
    if '__slots__' not in attrs:
      attrs['__slots__'] = ()
    return super().__new__(cls, cls_name, bases, attrs)

  def __instancecheck__(self, instance):
    if super().__instancecheck__(instance):
      return True

    if isinstance(instance, jax_core.Tracer):
      ty = jax_core.typeof(instance)
      if isinstance(ty, AbstractMutableHijaxVariable):
        var_type = ty._var_type
        return issubclass(var_type, self)
    if isinstance(instance, MutableHijaxVariable):
      var_type = instance._var_type
      return issubclass(var_type, self)
    return False

  def __subclasscheck__(self, subclass):
    if super().__subclasscheck__(subclass):
      return True
    if issubclass(subclass, AbstractMutableHijaxVariable):
      var_type = subclass._var_type
      return issubclass(var_type, self)
    if issubclass(subclass, MutableHijaxVariable):
      var_type = subclass._var_type
      return issubclass(var_type, self)
    return False

  if not tp.TYPE_CHECKING:

    def __call__(cls, *args, **kwargs):
      return cls._variable_meta_call(*args, **kwargs)

  def _variable_meta_call(
    cls, *args, use_hijax: bool | tp.Literal['mutable'] | None = None, **kwargs
  ):
    if use_hijax is None:
      use_hijax = using_hijax()
    variable = super().__call__(*args, use_hijax=use_hijax, **kwargs)
    if use_hijax == 'mutable':
      variable = _new_mutable_hijax_from_variable(variable)
    elif use_hijax is True:
      raise ValueError('`use_hijax=True` is currently not supported.')
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

  __slots__ = ('raw_value', '_trace_state', '_var_metadata')
  _var_type: type[Variable[A]]
  _mutable_hijax_type: type[MutableHijaxVariable[tp.Any]]
  _abstract_hijax_type: type[AbstractMutableHijaxVariable[tp.Any]]
  raw_value: A
  _trace_state: tracers.TraceState
  _var_metadata: dict[str, tp.Any]
  is_hijax: bool | tp.Literal['mutable'] = False
  is_stateful = False

  def __init__(
    self,
    value: A | VariableMetadata[A],
    *,
    use_ref: bool | None = None,
    use_hijax: bool | tp.Literal['mutable'] | None = None,
    **metadata: tp.Any,
  ):
    if use_hijax and use_ref:
      raise ValueError('Cannot use both `use_hijax` and `use_ref`.')

    if use_ref is None:
      use_ref = using_refs()

    var_t = type(self)
    object.__setattr__(self, '_trace_state', tracers.TraceState())

    if isinstance(value, VariableMetadata):
      metadata.update(value.metadata)
      value = tp.cast(A, value.raw_value)

    if is_array_ref(value):
      raise ValueError('Cannot pass a Ref directly into Variable constructor.')

    object.__setattr__(self, 'raw_value', value)

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

    # shard the _value if applicable
    do_eager_sharding = config.flax_always_shard_variable
    if 'eager_sharding' in metadata:
      do_eager_sharding = metadata['eager_sharding']
    if do_eager_sharding and 'sharding_names' in metadata:
      value = core_spmd.shard_value(
        value,
        metadata['sharding_names'],
        metadata.get('sharding_rules', None),
        metadata.get('mesh', None),
      )

    # Create the ref out of the array value
    if use_ref:
      value = array_ref(jnp.asarray(value))  # type: ignore[assignment]  # type: ignore[assignment]

    object.__setattr__(self, '_var_metadata', metadata)
    object.__setattr__(self, 'raw_value', value)

  def _to_mutable_hijax(self: V) -> V:
    mutable_hijax_var = _new_mutable_hijax_from_variable(self)
    return mutable_hijax_var  # type: ignore[return-value]

  @property
  def trace_state(self) -> tracers.TraceState:
    return self._trace_state

  def __getattr__(self, name: str) -> tp.Any:
    if name in object.__getattribute__(self, '_var_metadata'):
      return self._var_metadata[name]
    return getattr(self.raw_value, name)

  def __setattr__(self, name: str, value: tp.Any):
    if not self._trace_state.is_valid() and (
      name != 'value' or not self.has_ref
    ):
      raise errors.TraceContextError(
        f'Cannot mutate {type(self).__name__} from a different trace level'
      )
    try:
      object.__setattr__(self, name, value)
    except AttributeError:
      raise AttributeError(
        f'Cannot set attribute {name}.\n'
        f"To set Variable metadata use: `variable.set_metadata('{name}', value)`."
      )

  def __delattr__(self, name: str):
    if not self._trace_state.is_valid():
      raise errors.TraceContextError(
        f'Cannot mutate {type(self).__name__} from a different trace level'
      )

    if (
      name == 'value'
      or name == 'raw_value'
      or name == '_var_metadata'
      or name == '_trace_state'
    ):
      raise AttributeError(f'Cannot delete attribute {name}')
    else:
      del self._var_metadata[name]

  # NOTE(cgarciae): adding this for backward compatibility with VariableState
  @property
  def type(self):
    """The type of the variable."""
    import warnings

    warnings.warn(
      "'.type' is deprecated, use 'type(variable)' instead.",
      DeprecationWarning,
      stacklevel=2,
    )
    return type(self)

  @property
  def has_ref(self) -> bool:
    return is_array_ref(self.raw_value)

  @tp.overload
  def get_metadata(self) -> dict[str, tp.Any]: ...
  @tp.overload
  def get_metadata(self, name: str) -> tp.Any: ...
  def get_metadata(self, name: str | None = None):
    """Get metadata for the Variable.

    Args:
      name: The key of the metadata element to get. If not provided, returns
        the full metadata dictionary.
    """
    if name is None:
      return self._var_metadata
    return self._var_metadata[name]

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
      self._var_metadata = dict(args[0])
    elif len(args) == 2:
      name, value = args
      self._var_metadata[name] = value
    elif kwargs:
      self._var_metadata.update(kwargs)
    else:
      raise TypeError(
        f'set_metadata takes either 1 or 2 arguments, or at least 1 keyword argument, '
        f'got args={args}, kwargs={kwargs}'
      )

  def copy_from(self, other: Variable[A]) -> None:
    if type(self) is not type(other):
      raise ValueError(
        f'Cannot copy from incompatible container, '
        f'expected {type(self).__name__}, got {type(other).__name__}'
      )
    if self is other:
      return
    self.raw_value = other.raw_value
    self._var_metadata.clear()
    self._var_metadata.update(other.get_metadata())

  def update_from_state(self, variable_state: Variable[A]):
    if self.has_ref and (
      variable_state.has_ref or isinstance(variable_state.raw_value, jax.Array)
    ):
      self.raw_value[...] = variable_state.raw_value[...]  # type: ignore
    else:
      object.__setattr__(self, 'raw_value', variable_state.raw_value)

    if self._var_metadata != variable_state._var_metadata:
      object.__setattr__(
        self, '_var_metadata', variable_state._var_metadata.copy()
      )

  @property
  def value(self) -> A:
    value = self.raw_value
    if is_array_ref(value):
      value = value[...]

    if 'on_get_value' in self._var_metadata:
      value = self._var_metadata['on_get_value'](self, value)
    return value

  @value.setter
  def value(self, value: A):
    if isinstance(value, Variable):
      raise ValueError(
        'Cannot set value to a Variable, use `copy_from` method instead'
      )
    if 'on_set_value' in self._var_metadata:
      value = self._var_metadata['on_set_value'](self, value)
    if self.has_ref:
      self.raw_value[...] = value  # type: ignore
    else:
      object.__setattr__(self, 'raw_value', value)

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
      kwargs['raw_value'] = value

    # rename `value` to `raw_value`
    if 'value' in kwargs:
      kwargs['raw_value'] = kwargs.pop('value')

    # return `value` if it is a Variable
    if 'raw_value' in kwargs and isinstance(
      value := kwargs['raw_value'], Variable
    ):
      # remove value from kwargs
      kwargs.pop('raw_value')
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
    object.__setattr__(obj, 'raw_value', kwargs.pop('raw_value'))
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
    object.__setattr__(obj, 'raw_value', value)
    object.__setattr__(obj, '_var_metadata', metadata)
    return obj

  @classmethod
  def from_metadata(
    cls,
    value: A,
    attributes: dict[str, tp.Any],
    use_hijax: bool | tp.Literal['mutable'] | None = None,
  ) -> Variable[A]:
    obj = cls._new(value, attributes)
    if use_hijax is None:
      use_hijax = using_hijax()
    if use_hijax == 'mutable':
      obj = _new_mutable_hijax_from_variable(obj)
    elif use_hijax is True:
      raise ValueError('`use_hijax=True` is currently not supported.')
    return obj  # type: ignore[return-value]

  def copy(self: Variable[A]) -> Variable[A]:
    obj = object.__new__(type(self))
    object.__setattr__(obj, '_trace_state', tracers.TraceState())
    object.__setattr__(obj, 'raw_value', self.raw_value)
    object.__setattr__(obj, '_var_metadata', self.get_metadata().copy())
    return obj

  to_state = copy

  def __nnx_repr__(self):
    stats = SizeBytes.from_any(self.raw_value)
    if stats:
      comment = f' # {stats}'
    else:
      comment = ''

    yield reprlib.Object(type=type(self).__name__, comment=comment)
    yield reprlib.Attr('value', self.raw_value)
    for name, value in self._var_metadata.items():
      yield reprlib.Attr(name, value)

  def __treescope_repr__(self, path, subtree_renderer):
    size_bytes = SizeBytes.from_any(self.value)
    if size_bytes:
      stats_repr = f' # {size_bytes}'
      first_line_annotation = treescope.rendering_parts.comment_color(
        treescope.rendering_parts.text(f'{stats_repr}')
      )
    else:
      first_line_annotation = None

    children = {'value': self.raw_value, **self._var_metadata}
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
    return self.value

  # pickle support
  def __getstate__(self):
    return {
      'raw_value': self.raw_value,
      '_trace_state': self._trace_state,
      '_var_metadata': self._var_metadata,
    }

  def __setstate__(self, state):
    object.__setattr__(self, 'raw_value', state['raw_value'])
    object.__setattr__(self, '_trace_state', state['_trace_state'])
    object.__setattr__(self, '_var_metadata', state['_var_metadata'])

  # --------------------------------------------
  # proxy methods
  # --------------------------------------------

  def __getitem__(self, key) -> jax.Array:
    return self.value[key]  # type: ignore

  def __setitem__(self, key, value) -> None:
    if not self.has_ref and not self._trace_state.is_valid():
      raise errors.TraceContextError(
        f'Cannot mutate {type(self).__name__} from a different trace level'
      )
    if self.has_ref:
      self.raw_value[key] = value  # type: ignore
    elif key == ...:
      self.value = value
    elif isinstance(self.value, jax.Array):
      self.value = self.value.at[key].set(value)  # type: ignore
    else:
      self.raw_value[key] = value  # type: ignore

  def __delitem__(self, key) -> None:
    if not self._trace_state.is_valid():
      raise errors.TraceContextError(
        f'Cannot mutate {type(self).__name__} from a different trace level'
      )
    del self.raw_value[key]  # type: ignore

  def __call__(self, *args, **kwargs) -> tp.Any:
    return self.value(*args, **kwargs)  # type: ignore

  def __len__(self) -> int:
    return len(self.value)  # type: ignore

  def __iter__(self) -> tp.Iterator:
    return iter(self.value)  # type: ignore

  def __contains__(self, item) -> bool:
    return item in self.value  # type: ignore

  def __add__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__add__(other)  # type: ignore

  def __sub__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__sub__(other)  # type: ignore

  def __mul__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__mul__(other)  # type: ignore

  def __matmul__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__matmul__(other)  # type: ignore

  def __truediv__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__truediv__(other)  # type: ignore

  def __floordiv__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__floordiv__(other)  # type: ignore

  def __mod__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__mod__(other)  # type: ignore

  def __divmod__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__divmod__(other)  # type: ignore

  def __pow__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__pow__(other)  # type: ignore

  def __lshift__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__lshift__(other)  # type: ignore

  def __rshift__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__rshift__(other)  # type: ignore

  def __and__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__and__(other)  # type: ignore

  def __xor__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__xor__(other)  # type: ignore

  def __or__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__or__(other)  # type: ignore

  def __radd__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__radd__(other)  # type: ignore

  def __rsub__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__rsub__(other)  # type: ignore

  def __rmul__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__rmul__(other)  # type: ignore

  def __rmatmul__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__rmatmul__(other)  # type: ignore

  def __rtruediv__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__rtruediv__(other)  # type: ignore

  def __rfloordiv__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__rfloordiv__(other)  # type: ignore

  def __rmod__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__rmod__(other)  # type: ignore

  def __rdivmod__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__rdivmod__(other)  # type: ignore

  def __rpow__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__rpow__(other)  # type: ignore

  def __rlshift__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__rlshift__(other)  # type: ignore

  def __rrshift__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__rrshift__(other)  # type: ignore

  def __rand__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__rand__(other)  # type: ignore

  def __rxor__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__rxor__(other)  # type: ignore

  def __ror__(self, other) -> A:
    if isinstance(other, Variable):
      other = other.value
    return self.value.__ror__(other)  # type: ignore

  def __iadd__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable.value += x` instead.'
    )

  def __isub__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable.value -= x` instead.'
    )

  def __imul__(self: V, other) -> V:
    raise NotImplementedError(
      'In-place operations are no longer supported for Variable.\n'
      'Use `variable.value *= x` instead.'
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

  def __neg__(self) -> A:
    return self.value.__neg__()  # type: ignore

  def __pos__(self) -> A:
    return self.value.__pos__()  # type: ignore

  def __abs__(self) -> A:
    return self.value.__abs__()  # type: ignore

  def __invert__(self) -> A:
    return self.value.__invert__()  # type: ignore

  def __complex__(self) -> A:
    return self.value.__complex__()  # type: ignore

  def __int__(self) -> A:
    return self.value.__int__()  # type: ignore

  def __float__(self) -> A:
    return self.value.__float__()  # type: ignore

  def __index__(self) -> A:
    return self.value.__index__()  # type: ignore

  def __round__(self, ndigits: int) -> A:
    return self.value.__round__(ndigits)  # type: ignore

  def __trunc__(self) -> A:
    return self.value.__trunc__()  # type: ignore

  def __floor__(self) -> A:
    return self.value.__floor__()  # type: ignore

  def __ceil__(self) -> A:
    return self.value.__ceil__()  # type: ignore

  # --------------------------------------------

  def __init_subclass__(cls) -> None:
    if '__slots__' not in vars(cls):
      cls.__slots__ = ()
    super().__init_subclass__()
    jax.tree_util.register_pytree_with_keys(
      cls,
      flatten_with_keys=_variable_flatten_with_keys,
      unflatten_func=partial(_variable_unflatten, cls),  # type: ignore
      flatten_func=_variable_flatten,
    )
    cls._var_type = cls

    class MutableHijaxVar(MutableHijaxVariable):
      _var_type = cls

    MutableHijaxVar.__name__ = cls.__name__

    class AbstractMutableHijaxVar(AbstractMutableHijaxVariable):
      _var_type = cls

    AbstractMutableHijaxVar.__name__ = cls.__name__

    cls._mutable_hijax_type = MutableHijaxVar
    cls._abstract_hijax_type = AbstractMutableHijaxVar


def _variable_flatten_with_keys(x: Variable[tp.Any]):
  metadata = tuple(sorted(x._var_metadata.items()))
  node = (jtu.GetAttrKey('value'), x.raw_value)
  return (node,), metadata


def _variable_flatten(x: Variable[tp.Any]):
  metadata = tuple(sorted(x._var_metadata.items()))
  return (x.raw_value,), metadata


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
MutableHijaxVariable._var_type = Variable
AbstractMutableHijaxVariable._var_type = Variable
Variable._mutable_hijax_type = MutableHijaxVariable
Variable._abstract_hijax_type = AbstractMutableHijaxVariable

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


class _Missing:
  pass


_MISSING = _Missing()


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
  typ: type[Variable[A]] | _Missing = _MISSING,
  *,
  overwrite=False,
) -> type[Variable[A]] | tp.Callable[[type[Variable[A]]], type[Variable[A]]]:
  """Register a pair of Linen collection name and its NNX type."""
  if typ is _MISSING:
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
