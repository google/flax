# Lint as: python3
"""TODO(jheek): DO NOT SUBMIT without one-line documentation for scope.

TODO(jheek): DO NOT SUBMIT without a detailed description of scope.
"""

import enum
import functools
import hashlib
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, TypeVar, Union

from . import tracers
from .frozen_dict import freeze
from .frozen_dict import FrozenDict
from .frozen_dict import unfreeze

import jax
from jax import lax
from jax import random


T = TypeVar('T')


PRNGKey = Any
Array = Any


KindFilter = Union[bool, str, Sequence[str]]

MaybeFrozenKind = Union[Dict[str, Any], FrozenDict[str, Any]]

Variables = Dict[str, MaybeFrozenKind]

def _fold_in_str(rng: PRNGKey, data: str) -> PRNGKey:
  """Fold a string into a jax.random.PRNGKey using its SHA-1 hash."""
  m = hashlib.sha1()
  m.update(data.encode('utf-8'))
  d = m.digest()
  hash_int = int.from_bytes(d[:4], byteorder='big')
  return random.fold_in(rng, hash_int)


def in_kind_filter(kind_filter: KindFilter, kind: str) -> bool:
  if isinstance(kind_filter, str):
    return kind == kind_filter
  if isinstance(kind_filter, Sequence) and not isinstance(kind_filter, str):
    return kind in kind_filter
  if isinstance(kind_filter, bool):
    return kind_filter
  raise TypeError('Invalid KindFilter')


class ScanVariableMode(enum.Enum):
  CARRY = ('carry', 'carry')
  BROADCAST = ('broadcast', None)
  ONCE = ('broadcast', 'broadcast')
  SCAN = ('scan', None)
  YIELD = (None, 'scan')
  MAP = ('scan', 'scan')


ScanVariableModes = Sequence[Tuple[KindFilter, Union[ScanVariableMode, str]]]


def group_kinds(xs: Variables,
                kind_filters: Sequence[KindFilter]) -> Sequence[Variables]:
  """Group variables by kind filters."""
  kinds = xs.keys()
  groups = []
  for kind_filter in kind_filters:
    remaining_kinds = []
    group = {}
    for kind in kinds:
      if in_kind_filter(kind_filter, kind):
        group[kind] = jax.tree_map(lambda x: x, xs[kind])
      else:
        remaining_kinds.append(kind)
    kinds = remaining_kinds
    groups.append(group)
  return tuple(groups)


class Scope:
  """Scope."""

  def __init__(self,
               variables: Variables,
               rngs: Optional[Dict[str, PRNGKey]] = None,
               name: Optional[str] = None,
               parent: Optional['Scope'] = None):
    self.parent = parent
    self.name = name
    self.variables = variables
    self.rngs = rngs if rngs else {}

    self.root = parent.root if parent else self
    self.trace_level = tracers.trace_level(tracers.current_trace())

    self.rng_counters = {key: 0 for key in self.rngs}
    self.rewind()

  def _validate_trace_level(self):
    tracers.check_trace_level(self.trace_level)

  def rewind(self):
    self.reservations = set()

  def default_name(self, name_prefix: str) -> str:
    i = 0
    while True:
      name = f'{name_prefix}_{i}'
      if name not in self.reservations:
        return name
      i += 1

  def push(self, name: Optional[str] = None, name_prefix: str = '') -> 'Scope':
    self._validate_trace_level()
    if name is None:
      name = self.default_name(name_prefix)
    assert name not in self.reservations
    self.reservations.add(name)
    rngs = {key: _fold_in_str(rng, name) for key, rng in self.rngs.items()}
    scope = Scope({}, name=name, rngs=rngs, parent=self)
    return scope

  def child(self,
            fn: Callable[..., Any],
            name: Optional[str] = None,
            **partial_kwargs) -> Callable[..., Any]:
    """Partially applies a child scope to fn."""
    prefix = fn.__name__ + '_' if hasattr(fn, '__name__') else ''
    scope = self.push(name, name_prefix=prefix)
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
      scope.rewind()
      kwargs = dict(partial_kwargs, **kwargs)
      return fn(scope, *args, **kwargs)
    return wrapper

  def get_kind(self, kind: str, mutable: bool = False) -> MaybeFrozenKind:
    """Returns all variable of a given kind."""
    if kind not in self.variables:
      if self.parent:
        parent_kind = self.parent.get_kind(kind, mutable)
        if self.name not in parent_kind:
          if isinstance(parent_kind, FrozenDict) or not mutable:
            return FrozenDict()
          parent_kind[self.name] = {}
        self.variables[kind] = parent_kind[self.name]
      elif mutable:
        self.variables[kind] = {}
      else:
        return FrozenDict()
    return self.variables[kind]

  def has_rng(self, kind: str) -> bool:
    return kind in self.rngs

  def make_rng(self, kind: str) -> PRNGKey:
    assert self.has_rng(kind)
    self._validate_trace_level()
    self.rng_counters[kind] += 1
    return random.fold_in(self.rngs[kind], self.rng_counters[kind])

  def get_variable(self, kind: str, name: str, default: T = None) -> T:
    variables = self.get_kind(kind)
    if name in variables:
      return variables[name]
    else:
      return default

  def has_variable(self, kind: str, name: str) -> bool:
    variables = self.get_kind(kind)
    return name in variables

  def put_variable(self, kind: str, name: str, value: Any):
    self._validate_trace_level()
    variables = self.get_kind(kind, mutable=True)
    variables[name] = value

  def param(self, name: str, init_fn: Callable[..., T], *init_args) -> T:
    if not self.has_variable('param', name):
      init_value = init_fn(self.make_rng('param'), *init_args)
      self.put_variable('param', name, init_value)
    return self.get_variable('param', name)

  def _populate_kinds(self):
    kinds = self.root.variables.keys()
    for kind in kinds:
      self.get_kind(kind)

  @staticmethod
  def pack(fn: Callable[..., Any],
           in_variable_filters: Sequence[KindFilter],
           out_variable_filters: Sequence[KindFilter],
           rng_filters: Sequence[KindFilter]) -> Callable[..., Any]:
    """Pack variables and rngs for functional transformations."""
    @functools.wraps(fn)
    def wrapper(self, *args):
      # pylint: disable=protected-access
      self._validate_trace_level()
      self._populate_kinds()
      variable_groups = group_kinds(self.variables, in_variable_filters)
      # Make sure in only variable kinds are frozen
      for variable_group in variable_groups:
        for kind, kind_variables in variable_group.items():
          kind_in_out = any(
              in_kind_filter(kind_filter, kind)
              for kind_filter in out_variable_filters)
          if not kind_in_out:
            variable_group[kind] = freeze(kind_variables)

      rng_groups = group_kinds(self.rngs, rng_filters)
      for rng_group in rng_groups:
        for kind in rng_group:
          rng_group[kind] = self.make_rng(kind)

      def scope_fn(variable_groups, rng_groups):
        variables = {}
        rngs = {}
        for variable_group in variable_groups:
          variables.update(variable_group)
        for rng_group in rng_groups:
          rngs.update(rng_group)
        scope = Scope(variables, name=self.name, rngs=rngs, parent=None)
        return scope

      def repack(scope):
        scope._validate_trace_level()
        mutable_variables = {key: val for key, val
                             in scope.variables.items()
                             if not isinstance(val, FrozenDict)}
        out_variable_groups = group_kinds(
            mutable_variables, tuple(out_variable_filters) + (True,))
        remainder = tuple(out_variable_groups[-1].keys())
        if remainder:
          raise ValueError(f'unmapped output variables: {remainder}')
        return out_variable_groups[:-1]

      y, out_variable_groups = fn(
          scope_fn, repack, variable_groups, rng_groups, *args)
      for out_variable_group in out_variable_groups:
        for kind, kind_variables in out_variable_group.items():
          for name, value in kind_variables.items():
            self.put_variable(kind, name, value)
      return y
    return wrapper

  @staticmethod
  def vmap(fn: Callable[..., Any],
           in_axes=0, out_axes=0,
           variable_in_axes=((True, 0),),
           variable_out_axes=((True, 0),),
           split_rngs=((True, True),)) -> Callable[..., Any]:
    """Wraps jax.vmap."""
    variable_in_groups, variable_in_axes = _unzip2(variable_in_axes)
    variable_out_groups, variable_out_axes = _unzip2(variable_out_axes)
    rng_groups, rng_splits = _unzip2(split_rngs)
    rng_axes = tuple(0 if rng_split else None for rng_split in rng_splits)

    def inner(scope_fn, repack_fn, variable_groups, rng_groups, *args):
      def find_axis_size(axis, x):
        if axis is not None:
          leaves = jax.tree_leaves(x)
          if leaves:
            return leaves[0].shape[axis]
        return ()
      # split rngs
      axis_sizes = jax.tree_multimap(find_axis_size, in_axes, args)
      axis_size, = set(jax.tree_leaves(axis_sizes))
      split_fn = lambda rng: random.split(rng, axis_size)
      rng_groups = tuple(
          jax.tree_map(split_fn, rng_group) if split else rng_group
          for rng_group, split in zip(rng_groups, rng_splits))

      @functools.partial(jax.vmap,
                         in_axes=(variable_in_axes, rng_axes, in_axes),
                         out_axes=(out_axes, variable_out_axes))
      @functools.wraps(fn)
      def mapped(variable_groups, rng_groups, args):
        scope = scope_fn(variable_groups, rng_groups)
        y = fn(scope, *args)
        return y, repack_fn(scope)

      return mapped(variable_groups, rng_groups, args)

    return Scope.pack(
        inner, variable_in_groups, variable_out_groups, rng_groups)

  @staticmethod
  def scan(
      fn: Callable[..., Any], scope: 'Scope', init_carry: Any, xs: Any,
      length: Optional[int] = None, reverse: bool = False,
      variable_modes: ScanVariableModes = ((True, 'carry'),),
      split_rngs=((True, True),)) -> Callable[..., Any]:
    """Wraps jax.lax.scan."""
    def parse_mode(mode):
      if isinstance(mode, str):
        mode = ScanVariableMode[mode.upper()]
      return mode.value
    variable_modes = tuple(
        (group, parse_mode(mode)) for group, mode in variable_modes)
    if length is None:
      length, = set(x.shape[0] for x in jax.tree_leaves(xs))
    variable_groups, variable_modes = _unzip2(variable_modes)
    rng_groups, rng_splits = _unzip2(split_rngs)
    variable_in_groups = tuple(
        False if mode[0] is None else group
        for group, mode in zip(variable_groups, variable_modes))
    variable_out_groups = tuple(
        False if mode[1] is None else group
        for group, mode in zip(variable_groups, variable_modes))

    def split(variable_groups, i):
      scan_vars = tuple(
          group if mode[i] == 'scan' else {}
          for group, mode in zip(variable_groups, variable_modes))
      carry_vars = tuple(
          group if mode[i] == 'carry' else {}
          for group, mode in zip(variable_groups, variable_modes))
      broadcast_vars = tuple(
          group if mode[i] == 'broadcast' else {}
          for group, mode in zip(variable_groups, variable_modes))
      return scan_vars, carry_vars, broadcast_vars

    def combine(*variable_groups):
      combined_groups = []
      for groups in zip(*variable_groups):
        result = {}
        for group in groups:
          result.update(group)
        combined_groups.append(result)
      return combined_groups

    def inner(scope_fn, repack_fn, variable_groups, rng_groups):
      # split rngs
      split_fn = lambda rng: random.split(rng, length)
      broadcast_rngs = tuple(
          rng_group for rng_group, split
          in zip(rng_groups, rng_splits) if not split)
      scan_rngs = tuple(
          jax.tree_map(split_fn, rng_group)
          for rng_group, split in zip(rng_groups, rng_splits) if split)

      def body(carry, xs):
        nonlocal broadcast_vars
        carry_vars, c = carry
        scan_vars, scan_rngs, x = xs
        variable_groups = combine(scan_vars, carry_vars, broadcast_vars)
        scope = scope_fn(variable_groups, broadcast_rngs + scan_rngs)
        carry, y = fn(scope, c, x)
        out_vars = repack_fn(scope)
        scan_vars, carry_vars, broadcast_vars = split(out_vars, 1)
        return (carry_vars, carry), (scan_vars, y)

      scan_vars, carry_vars, broadcast_vars = split(variable_groups, 0)
      carry0 = (carry_vars, init_carry)
      xxs = (scan_vars, scan_rngs, xs)
      (carry_vars, carry), (scan_vars, ys) = lax.scan(
          body, carry0, xxs, length=length, reverse=reverse)
      out_vars = combine(carry_vars, scan_vars, broadcast_vars)
      return (carry, ys), out_vars

    return Scope.pack(
        inner, variable_in_groups, variable_out_groups, rng_groups)(scope)

  @staticmethod
  def jit(fn: Callable[..., Any],
          static_argnums: Union[int, Iterable[int]] = (),
          device=None,
          backend: Union[str, None] = None,
          in_variables: KindFilter = True,
          out_variables: KindFilter = True,
          rngs: KindFilter = True) -> Callable[..., Any]:
    """Wraps jax.jit."""
    if not isinstance(static_argnums, Iterable):
      static_argnums = (static_argnums,)
    static_argnums = tuple(i + 1 for i in static_argnums if i > 0)
    def inner(scope_fn, repack_fn, variable_groups, rng_groups, *args):
      @functools.partial(jax.jit,
                         static_argnums=static_argnums,
                         device=device, backend=backend)
      @functools.wraps(fn)
      def jitted(variable_groups, rng_groups, *args):
        scope = scope_fn(variable_groups, rng_groups)
        y = fn(scope, *args)
        return y, repack_fn(scope)

      return jitted(variable_groups, rng_groups, *args)

    return Scope.pack(inner, (in_variables,), (out_variables,), (rngs,))


def _unzip2(xs):
  ys = tuple(zip(*xs))
  return ys if ys else ((), ())


def _unfreeze_variables(variables, mutable):
  new_variables = {}
  for key, value in variables.items():
    if in_kind_filter(mutable, key):
      new_variables[key] = unfreeze(value)
    else:
      new_variables[key] = value
  return new_variables


def apply(fn: Callable[..., Any],
          mutable: KindFilter = False) -> Callable[..., Any]:
  """Functionalize a module."""
  @functools.wraps(fn)
  def wrapper(variables, *args, rngs=None, **kwargs):
    new_variables = _unfreeze_variables(variables, mutable)
    root = Scope(new_variables, rngs=rngs)
    y = fn(root, *args, **kwargs)
    if mutable:
      return y, freeze(new_variables)
    else:
      return y
  return wrapper


def init(fn: Callable[..., Any], mutable: bool = True) -> Callable[..., Any]:
  @functools.wraps(fn)
  def wrapper(rngs, *args, **kwargs):
    if not isinstance(rngs, dict):
      rngs = {'param': rngs}
    return apply(fn, mutable=mutable)({}, *args, rngs=rngs, **kwargs)
  return wrapper
