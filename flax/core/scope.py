# Copyright 2021 The Flax Authors.
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

"""Flax functional core: Scopes."""

import contextlib
import functools
import hashlib
import dataclasses
from typing import Any, Callable, Container, Dict, Generic, Iterable, Mapping, Optional, Sequence, Set, Tuple, TypeVar, Union

from . import tracers
from flax import errors
from .frozen_dict import freeze
from .frozen_dict import FrozenDict
from .frozen_dict import unfreeze
import jax
from jax import numpy as jnp
from jax import random

T = TypeVar('T')

PRNGKey = Any
Array = Any

RNGSequences = Dict[str, PRNGKey]


Filter = Union[bool, str, Container[str], 'DenyList']

@dataclasses.dataclass(frozen=True, eq=True)
class DenyList:
  """DenyList represents an opt-out based mutability filter.
  
  DenyList can be used to make every collection mutable except the ones
  defined in the given filter.
  To for example make everything but the params collection mutable::

    nn.apply(fn, mutable=nn.DenyList(["params"]))

  Attributes:
    deny: The filter representing the collections that are not mutable.

  """
  deny: Filter


CollectionFilter = Filter
PRNGSequenceFilter = Filter

Collection = Mapping[str, Any]
MutableCollection = Dict[str, Any]

VariableDict = Mapping[str, Collection]
FrozenVariableDict = FrozenDict[str, Collection]
MutableVariableDict = Dict[str, MutableCollection]


def _fold_in_str(rng: PRNGKey, data: str) -> PRNGKey:
  """Folds a string into a jax.random.PRNGKey using its SHA-1 hash.

  This is faster than splitting an PRNGKey because it allows generating new PRNG
  keys in parallel that are independent of each other.

  Args:
   rng: the rng to fold the string into.
   data: the string to be folded in.

  Returns:
   The newly generated PRNG key.
  """
  m = hashlib.sha1()
  m.update(data.encode('utf-8'))
  d = m.digest()
  hash_int = int.from_bytes(d[:4], byteorder='big')
  return random.fold_in(rng, jnp.uint32(hash_int))


def in_filter(filter_like: Filter, col: str) -> bool:
  """Checks whether a filter can be applied to a collection.

  Used for both collections and rng sequence filters.

  Args:
    filter_like: a filter (either a boolean, a string, or a list of strings) for
      a collection.
    col: a collection, which is a string identifying a dictionary of data, for
      instance "params" or "batch_stats".

  Returns:
    True if either `filter_like` is True, equal to `col`, or a sequence
    containing `col`.
  """
  if isinstance(filter_like, str):
    return col == filter_like
  if isinstance(filter_like, Container):
    return col in filter_like
  if isinstance(filter_like, bool):
    return filter_like
  if isinstance(filter_like, DenyList):
    return not in_filter(filter_like.deny, col)
  raise errors.InvalidFilterError(filter_like)


def filter_to_set(x: Filter) -> Set[str]:
  """Converts a Filter into a set of collections, fails on the infinite set.

  Args:
    x: a filter (boolean, string, or list of strings).

  Returns:
    The input filter represented as a set of strings.
  """
  assert x is not True and not isinstance(x, DenyList), 'Infinite set'
  if x is False:
    return set()
  if isinstance(x, str):
    return set([x])
  if isinstance(x, Iterable):
    return set(x)
  raise errors.InvalidFilterError(x)


def union_filters(a: Filter, b: Filter) -> Filter:
  """Takes the union of two filters (similar to a logical or).

  Args:
    a: a filter.
    b: a filter.

  Returns:
    The union of the two input filters. For instance,
    `union_filters('f1', ['f2']) = {'f1', 'f2'}`.
  """
  if a is True or b is True:
    return True
  if isinstance(a, DenyList) and isinstance(b, DenyList):
    return DenyList(intersect_filters(a.deny, b.deny))
  if isinstance(b, DenyList):
    a, b = b, a
  if isinstance(a, DenyList):
    return DenyList(subtract_filters(a.deny, b))
  
  a = filter_to_set(a)
  b = filter_to_set(b)
  return a.union(b)


def subtract_filters(a: Filter, b: Filter) -> Filter:
  """Returns the subtraction of b from a.

  Args:
    a: a filter.
    b: a filter.

  Returns:
    A filter matching with values in a that are not in b.
  """
  if b is True:
    return False
  if a is True:
    return DenyList(b)
  if isinstance(a, DenyList) and isinstance(b, DenyList):
    return subtract_filters(b.deny, a.deny)
  if isinstance(a, DenyList):
    return DenyList(union_filters(a.deny, b))
  if isinstance(b, DenyList):
    return intersect_filters(a, b.deny)
  a = filter_to_set(a)
  b = filter_to_set(b)
  return a - b


def intersect_filters(a: Filter, b: Filter) -> Filter:
  """Take the intersection of two filters (similar to a logical and).

  Args:
    a: a filter.
    b: a filter.

  Returns:
    The intersection of the two input filters. For instance,
    `intersect_filters('f1', ['f1', 'f2']) = {'f1'}`.
  """
  if a is True:
    return b
  if b is True:
    return a
  if isinstance(a, DenyList) and isinstance(b, DenyList):
    return DenyList(union_filters(b.deny, a.deny))
  if isinstance(b, DenyList):
    b, a = a, b
  if isinstance(a, DenyList):
    return subtract_filters(b, a.deny)
  a = filter_to_set(a)
  b = filter_to_set(b)
  return a.intersection(b)


def group_collections(
    xs: VariableDict,
    col_filters: Sequence[CollectionFilter]) -> Sequence[MutableVariableDict]:
  """Groups variables by collection filters.

  Iteratively applies the filters in `col_filters` to `xs`, and adds the result
  of applying each filter to the output sequence. Each key in `xs` is only added
  to the output once.

  Args:
    xs: a dictionary of variables, keyed by collections (strings).
    col_filters: a list of collection filters.

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
  """A Variable object allows mutable access to a variable in a VariableDict.

  Variables are identified by a collection (e.g., "batch_stats") and a name
  (e.g., "moving_mean"). The value property gives access to the variable's
  content and can be assigned to for mutation.
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

  def is_mutable(self) -> bool:
    """Checks if this Variable is mutable."""
    return self.scope.is_mutable_collection(self.collection)


class Scope:
  """A Scope allows easy access to variables and manages RNGS of a neural network layer.

  Scopes are purely functional and encapsulated in
  :class:`flax.linen.module.Module`, so users writing neural network code
  usually generally do not interact with ``Scopes`` directly.

  See `core design tests
  <https://github.com/google/flax/tree/master/tests/core/design>`_
  for a number of examples using ``Scopes``.
  """

  def __init__(self,
               variables: MutableVariableDict,
               rngs: Optional[Dict[str, PRNGKey]] = None,
               name: Optional[str] = None,
               mutable: CollectionFilter = False,
               parent: Optional['Scope'] = None,
               path: Iterable[str] = ()):
    """Initializes a Scope.

    Args:
      variables: VariableDict to initialize the Scope with.
      rngs: RNGs used in this scope or one of the child scopes.
      name: name of this scope.
      mutable: A CollectionFilter determining which variables are mutable.
      parent: The parent scope.
      path: The path in the variable tree from the root scope to this scope. 
    """
    self._variables = variables
    self.parent = parent
    self.name = name
    self.path = tuple(path)
    self.rngs = rngs if rngs else {}
    self.mutable = mutable

    self.root = parent.root if parent else self
    self.trace_level = tracers.trace_level(tracers.current_trace())

    self.rng_counters = {key: 0 for key in self.rngs}
    self.reservations = set()

    self._children = {}

    self._invalid = False

  @property
  def path_text(self) -> str:
    """Returns the path as a human readable string with slashes between parts."""
    return '/' + '/'.join(self.path)

  @property
  def invalid(self) -> bool:
    """Returns true if this scope is invalidated as a result of `Scope.temporary`."""
    return self._invalid

  def _check_valid(self):
    if self._invalid:
      raise errors.InvalidScopeError(self.name)

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

  def mutable_variables(self) -> VariableDict:
    """Returns an immutable copy of the mutable variables belonging to this Scope."""
    self._populate_collections()
    xs = {k: v for k, v in self._variables.items()
          if in_filter(self.mutable, k)}
    return freeze(xs)

  def variables(self) -> VariableDict:
    """Returns an immutable copy of the variables belonging to this Scope."""
    self._populate_collections()
    return freeze(self._variables)

  def _validate_trace_level(self):
    tracers.check_trace_level(self.trace_level)

  def rewound(self, rewind_rngs: bool = False) -> 'Scope':
    """Returns a rewound version of this Scope.

    Args:
      rewind_rngs: if true, reset the RNG counter of this scope.

    Returns:
      A rewound version of this scope, which means reservations and children are
      emptied, and the rng counter is optionally rewound.
    """
    self._check_valid()
    scope = Scope(self._variables, self.rngs, self.name, self.mutable,
                  self.parent)
    if not rewind_rngs:
      scope.rng_counters = self.rng_counters
    return scope

  def reserve(self, name: str):
    """Reserves a name for a child Scope or Variable.

    Args:
      name: the name to reserve.
    """
    if not isinstance(name, str):
      raise TypeError('The type of scope "{name}" should be string but '
                     f'it is {type(name)}')
    if name in self.reservations:
      raise ValueError(f'Duplicate use of scope name: "{name}"')
    self.reservations.add(name)

  def default_name(self, prefix: str) -> str:
    """Generates an unreserved name with the given prefix.

    Args:
      prefix: prefix to use for generating an unreserved name.

    Returns:
      The generated name.
    """
    i = 0
    while True:
      name = f'{prefix}{i}'
      if name not in self.reservations:
        return name
      i += 1

  def push(self,
           name: Optional[str] = None,
           prefix: str = '',
           reuse=False) -> 'Scope':
    """Creates a child Scope.

    Args:
      name: optional name of the child.
      prefix: prefix used for generating the name if `name` is `None`.
      reuse: if True will return a pre-existing child scope with the given name
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
    scope = Scope({},
                  name=name,
                  rngs=rngs,
                  parent=self,
                  path=self.path + (name,))
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
      name: optional name of the child.
      prefix: prefix used for generating name if it is `None`.
      named_call: if true, `fn` will be wrapped with `lift.named_call`. The XLA
        profiler will use this to name tag the computation.
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
      from . import lift  # type: ignore
      fn = lift.named_call(fn, name)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
      kwargs = dict(partial_kwargs, **kwargs)
      return fn(scope.rewound(), *args, **kwargs)

    return wrapper

  def is_mutable_collection(self, col: str) -> bool:
    """Returns true if the collection `col` is mutable."""
    return in_filter(self.root.mutable, col)

  def _mutable_collection(self, col: str) -> MutableCollection:
    """Returns the collection `col` as a mutable object."""
    assert self.is_mutable_collection(col), f'Collection {col} is not mutable'
    if col not in self._variables:
      if self.parent:
        parent_col = self.parent._mutable_collection(col)
        if self.name not in parent_col:
          parent_col[self.name] = {}
        self._variables[col] = parent_col[self.name]
      else:
        self._variables[col] = {}
    return self._variables[col]

  def _collection(self, col: str) -> Collection:
    """Returns a collection of variables of collection `col`."""
    if col not in self._variables:
      if self.parent:
        parent_col = self.parent._collection(col)
        if self.name not in parent_col:
          return FrozenDict()
        self._variables[col] = parent_col[self.name]
      else:
        return FrozenDict()
    return self._variables[col]

  def has_rng(self, name: str) -> bool:
    """Returns true if a PRNGSequence with name `name` exists."""
    return name in self.rngs

  def make_rng(self, name: str) -> PRNGKey:
    """Generates A PRNGKey from a PRNGSequence with name `name`."""
    if not self.has_rng(name):
      raise errors.InvalidRngError(f'{self.name} needs PRNG for "{name}"')
    self._check_valid()
    self._validate_trace_level()
    self.rng_counters[name] += 1
    return random.fold_in(self.rngs[name], self.rng_counters[name])

  def get_variable(self, col: str, name: str, default: T = None) -> T:
    """Retrieves the value of a Variable.

    Args:
      col: the variable collection.
      name: the name of the variable.
      default: the default value to return if the variable does not exist in
        this scope.

    Returns:
      The value of the input variable, of the default value if the variable
      doesn't exist in this scope.
    """
    variables = self._collection(col)
    if name in variables:
      return variables[name]
    else:
      return default

  def has_variable(self, col: str, name: str) -> bool:
    """Returns true if the given variable exists in this scope.

    Args:
      col: the collection of the variable.
      name: the name of the variable.
    """
    variables = self._collection(col)
    return name in variables

  def put_variable(self, col: str, name: str, value: Any):
    """Updates the value of the given variable if it is mutable, or an error otherwise.

    Args:
      col: the collection of the variable.
      name: the name of the variable.
      value: the new value of the given variable.
    """
    self._check_valid()
    self._validate_trace_level()
    if not self.is_mutable_collection(col):
      raise errors.ModifyScopeVariableError(col, name, self.path_text)
    variables = self._mutable_collection(col)
    variables[name] = value

  def variable(self, col: str, name: str, init_fn: Callable[..., T],
               *init_args) -> Variable[T]:
    """Creates a variable if it doesn't exist yet in this scope and returns it.

    Args:
      col: the collection of the variable.
      name: the name of the variable.
      init_fn: a function taking a PRNGKey plus any other number of positional
        arguments.
      *init_args: the arguments to evaluate init_fn on lazily.

    Returns:
      The variable.
    """
    self.reserve(name)
    if not self.has_variable(col, name):
      if not self.is_mutable_collection(col):
        raise errors.ScopeVariableNotFoundError(name, col, self.path_text)
      init_value = init_fn(*init_args)
      self.put_variable(col, name, init_value)
    return Variable(self, col, name)

  def param(self, name: str, init_fn: Callable[..., T], *init_args) -> T:
    """Creates a parameter if it doesn't exist yet in this scope and returns it.

    If the parameter exists already, the existing value is simply returned.

    Args:
      name: the name of the parameter.
      init_fn: a function taking a PRNGKey plus any other number of positional
        arguments.
      *init_args: the arguments to evaluate init_fn on lazily.

    Returns:
      The parameters.
    """
    self.reserve(name)
    if self.has_variable('params', name):
      abs_rng = jax.ShapeDtypeStruct((2,), jnp.uint32)
      value = self.get_variable('params', name)
      # Validate that the shape of the init_fn output is the same as the shape
      # of the existing parameter. This is to make sure that the hparams set up
      # in a Flax Module match the shapes coming in during apply, and if not,
      # catch it with an error message.
      # NOTE: We could consider moving this to `self.`
      abs_value = jax.eval_shape(lambda rng: init_fn(rng, *init_args), abs_rng)
      abs_value_flat = jax.tree_leaves(abs_value)
      value_flat = jax.tree_leaves(value)
      for val, abs_val in zip(value_flat, abs_value_flat):
        # NOTE: We could check dtype consistency here as well but it's
        # usefuleness is less obvious. We might intentionally change the dtype
        # for inference to a half float type for example.
        if jnp.shape(val) != jnp.shape(abs_val):
          raise errors.ScopeParamShapeError(name, self.path_text, 
              jnp.shape(val), jnp.shape(abs_val))
    else:
      if not self.is_mutable_collection('params'):
        raise errors.ScopeParamNotFoundError(name, self.path_text)
      value = init_fn(self.make_rng('params'), *init_args)
      self.put_variable('params', name, value)

    return value

  def _populate_collections(self):
    collections = self.root._variables.keys()
    for col in collections:
      self._collection(col)


def _unfreeze_variables(variables, mutable):
  new_variables = {}
  for key, value in variables.items():
    if in_filter(mutable, key):
      new_variables[key] = unfreeze(value)
    else:
      new_variables[key] = freeze(value)
  return new_variables


def bind(variables: VariableDict,
         rngs: Optional[RNGSequences] = None,
         mutable: CollectionFilter = False):
  """Bind variables and rngs to a new ``Scope``.
  
  bind provides a ``Scope`` instance without transforming a function
  with ``apply``. This is particalary useful for debugging and
  interactive use cases like notebooks where a function would limit
  the ability split up code into different cells.

  a ``Scope`` instance is a stateful object. Note that idiomatic JAX is functional
  and therefore a ``Scope` does not mix well well with vanilla JAX APIs. Therefore,
  we recommend using ``apply`` when code should be reusable and compatible
  across the JAX software ecosystem.
  """
  if not _is_valid_variables(variables):
    raise errors.ApplyScopeInvalidVariablesError()
  if rngs is not None and not _is_valid_rngs(rngs):
    raise errors.InvalidRngError(
      'rngs should be a dictionary mapping strings to `jax.PRNGKey`.')
  new_variables = _unfreeze_variables(variables, mutable)
  return Scope(new_variables, rngs=rngs, mutable=mutable)


def apply(fn: Callable[..., Any],
          mutable: CollectionFilter = False) -> Callable[..., Any]:
  """Functionalize a `Scope` function.

  Args:
    fn: a function taking a `Scope` as its first argument.
    mutable: the filter determining which variable collections are mutable.

  Returns:
    `fn` with the scope partially applied.
  """

  @functools.wraps(fn)
  def wrapper(variables: VariableDict,
              *args,
              rngs: Optional[RNGSequences] = None,
              **kwargs) -> Union[Any, Tuple[Any, VariableDict]]:
    with bind(variables, rngs=rngs, mutable=mutable).temporary() as root:
      y = fn(root, *args, **kwargs)
    if mutable is not False:
      return y, root.mutable_variables()
    else:
      return y

  return wrapper


def init(fn: Callable[..., Any],
         mutable: CollectionFilter = True) -> Callable[..., Any]:
  """Functionalize a `Scope` function for initialization.

  Args:
    fn: a function taking a `Scope` as its first argument.
    mutable: the filter determining which variable collections are mutable.

  Returns:
    `fn` with the scope partially applied.
  """

  @functools.wraps(fn)
  def wrapper(rngs, *args, **kwargs) -> Tuple[Any, VariableDict]:
    if not _is_valid_rng(rngs) and not _is_valid_rngs(rngs):
      raise ValueError('First argument passed to an init function should be a '
                       '`jax.PRNGKey` or a dictionary mapping strings to '
                       '`jax.PRNGKey`.')
    if not isinstance(rngs, dict):
      rngs = {'params': rngs}
    return apply(fn, mutable=mutable)({}, *args, rngs=rngs, **kwargs)

  return wrapper


def _is_valid_collection(col: VariableDict):
  if not isinstance(col, (FrozenDict, dict)):
    return False
  for name in col.keys():
    # Any value can be stored in a collection so only keys can be verified.
    if not isinstance(name, str):
      return False
  return True


def _is_valid_variables(variables: VariableDict) -> bool:
  """Checks whether the given variable dict is valid.

  Args:
    variables: A variable dict.

  Returns:
    True if `variables` is a valid variable dict.
  """
  for name, col in variables.items():
    if not isinstance(name, str):
      return False
    if not _is_valid_collection(col):
      return False
  return True


def _is_valid_rng(rng: Array):
  if not isinstance(rng, jnp.ndarray):
    return False
  if rng.shape != (2,) or rng.dtype != jnp.uint32:
    return False
  return True


def _is_valid_rngs(rngs: RNGSequences):
  if not isinstance(rngs, dict):
    return False
  for key, val in rngs.items():
    if not isinstance(key, str):
      return False
    if not _is_valid_rng(val):
      return False
  return True
