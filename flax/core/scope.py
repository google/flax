# Copyright 2020 The Flax Authors.
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

"""Flax functional core: Scopes.

The goal of the Flax functional core is to provide a purely functional 
abstraction that takes care of various types of bookkeeping such as parameter
and RNG management. The main abstraction is Scope, which contains parameters
and RNGs, and which can be nested.

TODO: elaborate.

"""

import contextlib
import enum
import functools
import hashlib
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, TypeVar, Union, Generic

from . import tracers
from .frozen_dict import freeze
from .frozen_dict import FrozenDict
from .frozen_dict import unfreeze

import jax
from jax import lax
from jax import random
from jax import numpy as jnp

T = TypeVar('T')


PRNGKey = Any
Array = Any

Filter = Union[bool, str, Sequence[str]]
CollectionFilter = Filter
PRNGSequenceFilter = Filter

FrozenCollection = FrozenDict[str, Any]
MaybeFrozenCollection = Union[Dict[str, Any], FrozenCollection]

Variables = Dict[str, MaybeFrozenCollection]


def _fold_in_str(rng: PRNGKey, data: str) -> PRNGKey:
  """Folds a string into a jax.random.PRNGKey using its SHA-1 hash.
  
  This is faster than splitting an PRNGKey because it allows generating new PRNG
  keys in parellel that are independent of each other.

  Args:
   rng: The rng to fold the string into.
   data: The string to be folded in.

  Returns:
   The newly generated PRNG key.
  """
  m = hashlib.sha1()
  m.update(data.encode('utf-8'))
  d = m.digest()
  hash_int = int.from_bytes(d[:4], byteorder='big')
  return random.fold_in(rng, hash_int)


def in_filter(filter_like: Filter, col: str) -> bool:
  """Checks whether a filter can be applied to a collection.
  
  Used for both collections and rng sequence filters.

  Args:
    filter_like: A filter (either a boolean, a string, or a list of strings)
      for a collection.
    col: A collection, which is a string identifying a dictionary of data, for
      instance "params" or "batch_stats".

  Returns:
    True if either `filter_like` is True, equal to `col`, or a sequence 
    containing `col`.
  """
  if isinstance(filter_like, str):
    return col == filter_like
  if isinstance(filter_like, Sequence) and not isinstance(filter, str):
    return col in filter_like
  if isinstance(filter_like, bool):
    return filter_like
  raise TypeError('Invalid Filter')


def group_collections(xs: Variables,
                col_filters: Sequence[CollectionFilter]) -> Sequence[Variables]:
  """Groups variables by collection filters.

  Iteratively applies the filters in `col_filters` to `xs`, and adds the result
  of applying each filter to the output sequence. Each key in `xs` is only added
  to the output once.
  
  Args:
    xs: A dictionary of variables, keyed by collections (strings).
    col_filters: A list of collection filters.
    
  Returns:
    A sequence S with `len(S) == len(col_filters)`. Each `S[i]` is the result of
    applying filter `col_filters[i]` to the remaining keys in `xs`.
    """
  cols = xs.keys()
  groups = []
  for col_filter in col_filters:
    remaining_cols = []
    group = {}
    for col in cols:
      if in_filter(col_filter, col):
        group[col] = jax.tree_map(lambda x: x, xs[col])
      else:
        remaining_cols.append(col)
    cols = remaining_cols
    groups.append(group)
  return tuple(groups)


class Variable(Generic[T]):
  """Scope Variable.

  A Variable is a dictionary of data. Variables are stored in Scopes, and 
  identified by a collection (e.g., "params") and a name (e.g., "dense"). New
  variable instances are obtained from a scope by applying `scope.variable()`.
  The value property gives access to the variable's content and can be assigned
  to for mutation.
  """

  def __init__(self, scope: 'Scope', collection: str, name: str):
    """Initializes a variable.

    Args:
      scope: The scope in which the variable is stored.
      collection: The collection of the variable (e.g., "params").
      name: The name of the variable (e.g., "dense").
    """
    self.scope = scope
    self.collection = collection
    self.name = name

  @property
  def value(self) -> T:
    """Returns the value of this Variable."""
    return self.scope.get_variable(self.collection, self.name)

  @value.setter
  def value(self, value: T):
    """Updates the value of this Variable."""
    self.scope.put_variable(self.collection, self.name, value)


class Scope:
  """Scope."""

  def __init__(self,
               variables: Variables,
               rngs: Optional[Dict[str, PRNGKey]] = None,
               name: Optional[str] = None,
               parent: Optional['Scope'] = None):
    """Initializes a Scope.

    Args:
      variables: Variables to initialize the Scope with.
      rngs: RNGs used in this scope or one of the child scopes.
      name: Name of this scope.
      parent: Parent scope.
    """
    self._variables = variables
    self.parent = parent
    self.name = name
    self.rngs = rngs if rngs else {}

    self.root = parent.root if parent else self
    self.trace_level = tracers.trace_level(tracers.current_trace())

    self.rng_counters = {key: 0 for key in self.rngs}
    self.reservations = set()

    self._children = {}

    self._invalid = False

  @property
  def invalid(self) -> bool:
    """Returns true if this scope is invalidated as a result of `Scope.temporary`."""
    return self._invalid

  def _check_valid(self):
    if self._invalid:
      raise ValueError('This scope is no longer valid.')

  @contextlib.contextmanager
  def temporary(self):
    """Returns a context manager that will invalidate this Scope when leaving the context."""
    try:
      yield self
    finally:
      self.invalidate()

  def invalidate(self):
    """Invalidates the Scope."""
    self._invalid = True

  def variables(self) -> FrozenCollection:
    """Returns an immutable copy of the variables belonging to this Scope."""
    self._populate_collections()
    return freeze(self._variables)

  def _validate_trace_level(self):
    tracers.check_trace_level(self.trace_level)

  def rewound(self, rewind_rngs: bool = False) -> 'Scope':
    """Returns a rewound version of this Scope.

    Args:
      rewind_rngs: If true, reset the RNG counter of this scope.
    Returns:
      A rewound version of this scope, which means reservations and children are 
      emptied, and the rng counter is optionally rewound.
    """
    self._check_valid()
    scope = Scope(self._variables, self.rngs, self.name, self.parent)
    if not rewind_rngs:
      scope.rng_counters = self.rng_counters
    return scope

  def reserve(self, name: str):
    """Reserves a name for a child Scope or Variable.
    
    Args:
      name: The name to reserve.
    """
    if name in self.reservations:
      raise ValueError(f'Duplicate use of name: "{name}"')
    self.reservations.add(name)

  def default_name(self, prefix: str) -> str:
    """Generates an unreserved name with the given prefix.
    
    Args:
      prefix: Prefix to use for generating an unreserved name.
    """
    i = 0
    while True:
      name = f'{prefix}{i}'
      if name not in self.reservations:
        return name
      i += 1

  def push(self, name: Optional[str] = None, prefix: str = '', reuse=False) -> 'Scope':
    """Creates a child Scope.
    
    Args:
      name: Optional name of the child.
      prefix: prefix used for generating the name if `name` is `None`.
      reuse: If True will return a pre-existing child scope with the given name
        instead of throwing an error.
    Returns:
      The child scope.
    """
    self._check_valid()
    self._validate_trace_level()
    if name is None:
      name = self.default_name(prefix)
    if reuse and name in self._children:
      return self._children[name]
    self.reserve(name)
    rngs = {key: _fold_in_str(rng, name) for key, rng in self.rngs.items()}
    scope = Scope({}, name=name, rngs=rngs, parent=self)
    self._children[name] = scope
    return scope

  def child(self,
            fn: Callable[..., Any],
            name: Optional[str] = None,
            prefix: Optional[str] = None,
            named_call: bool = True,
            **partial_kwargs) -> Callable[..., Any]:
    """Partially applies a child scope to fn.

    When calling the returned function multiple times variables will be reused.
    
    Args:
      fn: the function to partially apply the child Scope to.
      name: Optional name of the child.
      prefix: prefix used for generating name if it is `None`.
      named_call: If true, `fn` will be wrapped with `lift.named_call`.
        The XLA profiler will use this to name tag the computation.
      **partial_kwargs: additional kwargs partially applied to `fn`.
    Returns:
      The function with a partially applied scope.
    """
    if name is None:
      if prefix is None:
        prefix = fn.__name__ + '_' if hasattr(fn, '__name__') else ''
      name = self.default_name(prefix)
    scope = self.push(name)
    if named_call:
      # We import named_call at runtime to avoid a circular import issue.
      from . import lift
      fn = lift.named_call(fn, name)
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
      kwargs = dict(partial_kwargs, **kwargs)
      return fn(scope.rewound(), *args, **kwargs)
    return wrapper

  def collection(self, col: str, mutable: bool = False) -> MaybeFrozenCollection:
    """Returns a collection of variables."""
    if col not in self._variables:
      if self.parent:
        parent_col = self.parent.collection(col, mutable)
        if self.name not in parent_col:
          if isinstance(parent_col, FrozenDict) or not mutable:
            return FrozenDict()
          parent_col[self.name] = {}
        self._variables[col] = parent_col[self.name]
      elif mutable:
        self._variables[col] = {}
      else:
        return FrozenDict()
    return self._variables[col]

  def has_rng(self, name: str) -> bool:
    """Check whether a PRNGSequence exists."""
    return name in self.rngs

  def make_rng(self, name: str) -> PRNGKey:
    """Generate A PRNGKey from a PRNGSequence."""
    assert self.has_rng(name), f'Need PRNG for "{name}"'
    self._check_valid()
    self._validate_trace_level()
    self.rng_counters[name] += 1
    return random.fold_in(self.rngs[name], self.rng_counters[name])

  def get_variable(self, col: str, name: str, default: T = None) -> T:
    """Retrieve the value of a Variable."""
    variables = self.collection(col)
    if name in variables:
      return variables[name]
    else:
      return default

  def has_variable(self, col: str, name: str) -> bool:
    """Check whether a variable exists."""
    variables = self.collection(col)
    return name in variables

  def put_variable(self, col: str, name: str, value: Any):
    """Update the value of a Variable."""
    self._check_valid()
    self._validate_trace_level()
    variables = self.collection(col, mutable=True)
    variables[name] = value

  def variable(self, col: str, name: str, init_fn: Callable[..., T],
               *init_args) -> Variable[T]:
    """Create a Variable."""
    self.reserve(name)
    if not self.has_variable(col, name):
      init_value = init_fn(*init_args)
      self.put_variable(col, name, init_value)
    return Variable(self, col, name)

  def param(self, name: str, init_fn: Callable[..., T], *init_args) -> T:
    """Create a parameter."""
    self.reserve(name)
    if self.has_variable('params', name):
      abs_rng = jax.ShapeDtypeStruct((2,), jnp.uint32)
      value = self.get_variable('params', name)
      # validate shape of init_fn output is the same as the shape of the existing
      # parameter.
      abs_value = jax.eval_shape(lambda rng: init_fn(rng, *init_args), abs_rng)
      abs_value_flat = jax.tree_leaves(abs_value)
      value_flat = jax.tree_leaves(value)
      for val, abs_val in zip(value_flat, abs_value_flat):
        # NOTE: we could check dtype consistency here as well but it's usefuleness is less obvious.
        # we might intentionally change the dtype for inference to a half float type for example.
        if jnp.shape(val) != jnp.shape(abs_val):
          raise ValueError('Inconsistent shapes between value and initializer '
                           f'for parameter "{name}": {jnp.shape(val)}, {jnp.shape(abs_val)}')
      return value
    else:
      value = init_fn(self.make_rng('params'), *init_args)
      self.put_variable('params', name, value)
      return value

  def _populate_collections(self):
    collections = self.root._variables.keys()
    for col in collections:
      self.collection(col)

def _unfreeze_variables(variables, mutable):
  new_variables = {}
  for key, value in variables.items():
    if in_filter(mutable, key):
      new_variables[key] = unfreeze(value)
    else:
      new_variables[key] = value
  return new_variables


def apply(fn: Callable[..., Any],
          mutable: CollectionFilter = False) -> Callable[..., Any]:
  """Functionalize a `Scope` function.

  Args:
    fn: a function taking a `Scope` as its first argument.
    mutable: The filter determining which variable collections are mutable.
  Returns:
    `fn` with the scope partially applied.
  """
  @functools.wraps(fn)
  def wrapper(variables, *args, rngs=None, **kwargs):
    new_variables = _unfreeze_variables(variables, mutable)
    with Scope(new_variables, rngs=rngs).temporary() as root:
      y = fn(root, *args, **kwargs)
    if mutable:
      mutated_variables = {k: v
                           for k, v in new_variables.items()
                           if in_filter(mutable, k)}
      return y, freeze(mutated_variables)
    else:
      return y
  return wrapper


def init(fn: Callable[..., Any], mutable: CollectionFilter = True) -> Callable[..., Any]:
  """Functionalize a `Scope` function for initialization.

  Args:
    fn: a function taking a `Scope` as its first argument.
    mutable: The filter determining which variable collections are mutable.
  Returns:
    `fn` with the scope partially applied.
  """
  @functools.wraps(fn)
  def wrapper(rngs, *args, **kwargs):
    if not isinstance(rngs, dict):
      assert rngs.shape == (2,)
      rngs = {'params': rngs}
    return apply(fn, mutable=mutable)({}, *args, rngs=rngs, **kwargs)
  return wrapper
