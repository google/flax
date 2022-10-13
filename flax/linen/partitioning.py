# Copyright 2022 The Flax Authors.
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

"""Utilities for working with pjit and partitioned models.

**Experimental: please give feedback, and expect changes.**

This module introduces `axis_rules`, `logical_to_mesh_axes`,
`with_sharding_constraint` for appyling pjit sharding constraints in terms of
"logical named axes" rather than pjit's default mesh axes.

Additionally, flax linen methods `param_with_axes` and `variable_with_axes`
are introduced alongside `get_axis_names` for defining variables and parameters
and variables with logical axis name annotations that are managed as metadata.

Lastly, `*_with_axes` versions of `nn.scan` and `nn.vmap` are introduced to add
logical axis metadata to the underlying Lifted transformations.
"""

import collections
import contextlib
import enum
import functools
import re
import threading
from typing import (Any, Callable, List, Mapping, Optional, Sequence, Tuple,
                    Union)
import flax
from flax import linen as nn
from flax.core.frozen_dict import freeze
from flax.core.frozen_dict import unfreeze
from flax.core.lift import In as ScanIn  # pylint: disable=unused-import
from flax.core.lift import Out as ScanOut  # pylint: disable=unused-import
import flax.struct
from flax.traverse_util import flatten_dict
from flax.traverse_util import unflatten_dict
import jax
from jax.experimental import maps
from jax.experimental import pjit

# Real types and dummy aliases for documentation
LogicalRules = Sequence[Tuple[str, Union[str, Tuple[str], None]]]
Array = Any  # pylint: disable=invalid-name
ArrayPytree = Any  # pylint: disable=invalid-name
LogicalPartitionSpec = Any  # pylint: disable=invalid-name
LogicalPartitionSpecPytree = Any  # pylint: disable=invalid-name
PartitionSpecPytree = Any  # pylint: disable=invalid-name

# Dynamic Axis Mapping Context
# ------------------------------------------------------------------------------


class _AxisRules:
  """Dynamic logical axis to mesh axis binding context."""

  def __init__(self):
    self._thread_data = threading.local()

  @property
  def rules(self) -> LogicalRules:
    if not hasattr(self._thread_data, 'rules'):
      self._thread_data.rules = ()
    return self._thread_data.rules

  @rules.setter
  def rules(self, value: LogicalRules):
    self._thread_data.rules = value


# Global axis binding context.
_axis_rules = _AxisRules()


def set_axis_rules(rules: LogicalRules):
  """Sets the global logical axis to mesh axis binding."""
  _axis_rules.rules = rules


def get_axis_rules() -> LogicalRules:
  """Returns the global logical axis to mesh axis binding."""
  return _axis_rules.rules


@contextlib.contextmanager
def axis_rules(rules: LogicalRules):
  """Context manager for setting the logical to mesh axis bindings."""
  old_rules = _axis_rules.rules
  try:
    _axis_rules.rules = rules
    yield
  finally:
    _axis_rules.rules = old_rules


class _UnassignedAxis:
  """Sentinel class for unassigned logical axis name."""

  def __repr__(self):
    return 'UnassignedAxis'

  def __bool__(self):
    return False


_unassigned_axis = _UnassignedAxis()


def _mesh_assignment_free(new_assignment, existing_assignments):
  """Determines if a given mesh axis has already been assigned."""
  new = set(jax.tree_util.tree_leaves(new_assignment))
  existing = set(jax.tree_util.tree_leaves(existing_assignments))
  if existing.intersection(new):
    return False
  return True


def _logical_to_mesh_axes(
    array_dim_names: Optional[Sequence[Optional[str]]],
    rules: Optional[LogicalRules] = None,
) -> Optional[List[Union[_UnassignedAxis, None, str, Tuple[str]]]]:
  """Same as logical_to_mesh_axes, but doesn't fill in _unassigned_axis."""
  if array_dim_names is None:
    return None
  if rules is None:
    rules = _axis_rules.rules
  axis_name_counts = collections.Counter(array_dim_names)
  dups = tuple(
      k for k, v in axis_name_counts.items() if v > 1 and k is not None)
  if dups:
    raise ValueError(
        f'Unsupported: Dimensions {dups} occur more than once in array names.')
  if not isinstance(rules, (tuple, list)):
    raise ValueError('Unknown axis rule specification type.')
  # We assign mesh axes using a priority based ruleset over logical axis names.
  result: List[Union[_UnassignedAxis, None, str, Tuple[str]]]
  result = [_unassigned_axis] * len(array_dim_names)
  for rule_model_name, rule_mesh_names in rules:
    if rule_model_name in array_dim_names:
      pos = array_dim_names.index(rule_model_name)
      if (_mesh_assignment_free(rule_mesh_names, result) and
          result[pos] == _unassigned_axis):
        result[pos] = rule_mesh_names
  return result


def logical_to_mesh_axes(
    array_dim_names: Optional[Sequence[Optional[str]]],
    rules: Optional[LogicalRules] = None,
) -> pjit.PartitionSpec:
  """Compute layout for an array.

  The rules are in order of precedence, and consist of pairs:
  (ArrayDimensionName, MeshDimensionName), meaning that the given array
  dimension (if present and unused) should be sharded across the given
  mesh dimension (if present and unused).

  A Layout of an Array is expressed as a tuple with one element for each
  dimension in the Array. The element is either None, or is the name of a
  mesh-dimension, meaning that this dimension of the array is sharded across
  this dimension of the mesh.

  For example, given an array with
    array_dim_names = ('batch', 'length', 'heads', 'features')
  and the layout rules are:
    rules = (('batch', 'X'),
             ('features', 'X'),
             ('heads', 'Y'),
             ('batch', 'Z'))

  then this function will return

    PartitionSpec('X', None, 'Y', None)

  Args:
    array_dim_names: Tuple of array dimension names or None.
    rules: Optional logical to mesh rules override.  Defaults to using the
      rules defined in the dynamic context set from the `axis_rules` function.

  Returns:
    PartitionSpec for the parameter.
  """
  result = _logical_to_mesh_axes(array_dim_names, rules)
  if result is None:
    return None
  # We default to None - ie unsharded along the dimension.
  result = [None if x is _unassigned_axis else x for x in result]
  return pjit.PartitionSpec(*result)


def _global_mesh_defined() -> bool:
  """Checks if global xmap/pjit mesh resource environment is defined."""
  maps_env = maps.thread_resources.env
  return maps_env.physical_mesh.devices.shape != ()  # pylint: disable=g-explicit-bool-comparison


class RulesFallback(enum.Enum):
  """How a sharding constraint should behave when no matching rule is found."""
  AXIS_IS_UNSHARDED = 'axis_is_unsharded'
  RAISE_ERROR = 'raise_error'
  NO_CONSTRAINT = 'no_constraint'


def _with_sharding_constraint(x: Array, axis_resources: pjit.PartitionSpec):
  """Wrapper for pjit with_sharding_constraint, no-op on cpu or outside pjit."""
  if jax.devices()[0].platform == 'cpu' or not _global_mesh_defined():
    return x
  else:
    return pjit.with_sharding_constraint(x, axis_resources)


def _with_sharding_constraint_one_fallback(
    axis_resources: LogicalPartitionSpec,
    x: Array,
    fallback: RulesFallback = RulesFallback.AXIS_IS_UNSHARDED):
  """Either imposes a sharding constraint or applies fallback."""
  mesh_axes = _logical_to_mesh_axes(axis_resources)
  if mesh_axes is None:
    return _with_sharding_constraint(x, None)

  if fallback == RulesFallback.AXIS_IS_UNSHARDED:
    mesh_axes = [None if x is _unassigned_axis else x for x in mesh_axes]
  else:
    if any(x is _unassigned_axis for x in mesh_axes):
      if fallback == RulesFallback.RAISE_ERROR:
        raise ValueError(f'Axis names {axis_resources} did not match a rule')
      else:
        return x
  return _with_sharding_constraint(x, pjit.PartitionSpec(*mesh_axes))


def _is_logical_spec(x):
  return x is None or (
      isinstance(x, tuple) and all(isinstance(e, str) or e is None for e in x))


def with_sharding_constraint(
    x: ArrayPytree,
    logical_axis_resources: LogicalPartitionSpecPytree,
    fallback: RulesFallback = RulesFallback.AXIS_IS_UNSHARDED):
  """Version of pjit's with_sharding_constraint that uses logical axis names."""
  # If no axis binding is set, this is a no-op.
  if not _axis_rules.rules or logical_axis_resources is None:
    return x
  # Translate logical names to mesh assignments.
  return jax.tree_util.tree_map(
      functools.partial(
          _with_sharding_constraint_one_fallback, fallback=fallback),
      logical_axis_resources,
      x,
      is_leaf=_is_logical_spec)


# Annotated parameters and Module axis metadata handling.
# ------------------------------------------------------------------------------


@flax.struct.dataclass
class AxisMetadata:
  """Contains a tuple of axis names, which is passed through FLAX."""
  names: LogicalPartitionSpecPytree = flax.struct.field(pytree_node=False)


def _param_with_axes_sow_reduce_fn(x, y):
  """Reduction function for sow() calls.

  Args:
    x: Existing value, or () if there was none.
    y: New axis names sown.

  Returns:
    New axis names.

  Raises:
    TypeError: If the newly sown value is not an AxisMetadata.
    ValueError: If the newly sown axis names don't match previously sown axis
      names.
    AssertionError: If a previously sown value was truthy and not an
      AxisMetadata.
  """
  if not isinstance(y, AxisMetadata):
    raise TypeError('Expected newly sown value to be an AxisMetadata')

  if isinstance(x, AxisMetadata):
    if x != y:
      raise ValueError('If axis names are sown twice, expected them to match. '
                       f'Got {x} and {y}.')
  elif x:
    # Shouldn't happen, so raise a fairly internal error.
    raise AssertionError(f'Non-initial-or-AxisMetadata value encountered: {x}')
  return y


def param_with_axes(
    name: str,
    init_fn,
    *init_args,
    axes: Optional[Tuple[str, ...]] = None,
    module: Optional[nn.Module] = None):
  """Declares and returns a parameter with logical axes in the current Module.

  See :mod:`flax.linen.module.param` for original docstring.

  Args:
    name: The parameter name.
    init_fn: The function that will be called to compute the initial value
      of this variable. This function will only be called the first time
      this parameter is used in this module.
    *init_args: The arguments to pass to init_fn.
    axes: A tuple of axis names, must match the rank of the param array.
    module: Use an explicit module instead of deriving the most recent from
      dynamic module context.

  Returns:
    The value of the initialized parameter.

  Raises:
    TypeError: if axes specification is mal-formed.
    ValueError: if specified logical axes don't match parameter rank.
  """
  # get current module if not explicitly provided
  if module is None:
    module = nn.module._context.module_stack[-1]  # pylint: disable=protected-access
    assert module is not None
  # define/fetch parameter on that module
  module_param = module.param(name, init_fn, *init_args)
  if axes is not None:
    # apply logical axis constraint immediately
    module_param = with_sharding_constraint(module_param,
                                            pjit.PartitionSpec(*axes))
    # record logical axis constraint for global axis metadata
    module.sow(
        'params_axes', f'{name}_axes', AxisMetadata(axes),
        reduce_fn=_param_with_axes_sow_reduce_fn)
  return module_param


class PartitionedVariable(flax.core.scope.Variable):
  """A PartitionedVariable object allows mutable access to a variable.

  PartitionedVariables are identified by a collection (e.g., "batch_stats") and
  a name (e.g., "moving_mean"). The value property gives access to the
  variable's content and can be assigned to for mutation.  Additionally,
  PartitionedVariables enforce logical sharding constraints on both retrieval
  and assignment.
  """

  def __init__(self,
               scope,
               collection: str,
               name: str,
               axes: Optional[Tuple[str, ...]] = None,
               fallback: RulesFallback = RulesFallback.AXIS_IS_UNSHARDED):
    """Initializes a partitioned variable.

    Args:
      scope: The scope in which the variable is stored.
      collection: The collection of the variable (e.g., "params").
      name: The name of the variable (e.g., "dense").
      axes: logical axes name of variable.
      fallback: Fallback behavior if no matching rule is found.
    """
    self.scope = scope
    self.collection = collection
    self.name = name
    self.axes = axes
    self.fallback = fallback

  @property
  def value(self):
    """Returns the value of this Variable."""
    value = self.scope.get_variable(self.collection, self.name)
    if self.axes is not None:
      value = with_sharding_constraint(value, self.axes, fallback=self.fallback)
    return value

  @value.setter
  def value(self, value):
    """Updates the value of this Variable."""
    if self.axes is not None:
      value = with_sharding_constraint(value, self.axes, fallback=self.fallback)
    self.scope.put_variable(self.collection, self.name, value)


def _core_variable_with_axes(
    scope,
    col: str,
    name: str,
    init_fn: Callable[..., Any],
    *init_args,
    axes: Optional[Tuple[str, ...]] = None,
    fallback: RulesFallback = RulesFallback.AXIS_IS_UNSHARDED):
  """Variant of flax core variable scope call with sharding constraints."""
  scope.reserve(name)
  if not scope.has_variable(col, name):
    if not scope.is_mutable_collection(col):
      raise flax.errors.ScopeVariableNotFoundError(name, col, scope.path_text)
    init_value = init_fn(*init_args)
    if axes is not None:
      init_value = with_sharding_constraint(init_value, axes, fallback=fallback)
    scope.put_variable(col, name, init_value)
  return PartitionedVariable(scope, col, name, axes, fallback)


def variable_with_axes(
    collection: str,
    name: str,
    init_fn,
    *init_args,
    axes: Optional[Tuple[str, ...]] = None,
    module: Optional[nn.Module] = None,
    fallback: RulesFallback = RulesFallback.AXIS_IS_UNSHARDED):
  """Declares and returns a variable with logical axes in the current Module.

  See :mod:`flax.linen.module.variable` for original docstring.

  Args:
    collection: The name of the variable collection.
    name: The variable name.
    init_fn: The function that will be called to compute the initial value
      of this variable. This function will only be called the first time
      this parameter is used in this module.
    *init_args: The arguments to pass to init_fn.
    axes: A tuple of axis names, must match the rank of the variable array.
    module: Use an explicit module instead of deriving the most recent from
      dynamic module context.
    fallback: How sharding should behave if there is no rule covering some axis.

  Returns:
    A flax `PartitionedVariable` object referencing the initialized variable
    array.

  Raises:
    TypeError: if axes specification is mal-formed.
    ValueError: if specified logical axes don't match parameter rank.
  """
  # get current module if not explicitly provided
  if module is None:
    module = nn.module._context.module_stack[-1]  # pylint: disable=protected-access
    assert module is not None
  module_var = _core_variable_with_axes(
      module.scope,
      collection,
      name,
      init_fn,
      *init_args,
      axes=axes,
      fallback=fallback)
  if axes is not None:
    # record logical axis constraint for global axis metadata
    module.sow(
        f'{collection}_axes', f'{name}_axes', AxisMetadata(axes),
        reduce_fn=_param_with_axes_sow_reduce_fn)
  return module_var


def get_axis_names(axes_metadata):
  """Gets axis names for variables as logical PartitionSpecs.

  Args:
    axes_metadata: a single axes-metadata collection from a flax-initialized
      set of collections.

  Returns:
    Collection of Partitionspecs with logical axis names, with the "_axes"
    suffix on variable names removed to match original variable collection for
    annotations.
  """
  def leaf_rewrite(x):
    return None if x is None else pjit.PartitionSpec(*x)
  def rewrite(tree):
    return jax.tree_util.tree_map(leaf_rewrite, tree, is_leaf=_is_logical_spec)
  axes_metadata = unfreeze(axes_metadata)  # pytype: disable=wrong-arg-types
  flat_dict = {
      re.sub(r'_axes$', '', '/'.join(k)): rewrite(v.names)
      for k, v in flatten_dict(axes_metadata).items()
  }
  return freeze(unflatten_dict(
      {tuple(k.split('/')): v for k, v in flat_dict.items()}))


# Metadata Aware Scan
# -----------------------------------------------------------------------------


def _tree_map_axes(fn, tree):
  """Only map over AxisMetadata leaves in pytree - identity for other leaves."""
  safe_fn = lambda x: fn(x) if isinstance(x, AxisMetadata) else x
  return jax.tree_util.tree_map(
      safe_fn, tree, is_leaf=lambda x: isinstance(x, AxisMetadata))


def _is_mutable(axis_col: str) -> bool:
  """Determines whether a collection is mutable.

  For example, when a module is called with `module.apply(..., mutable=['z'])`,
  this function will return True for `axis_col='z'` and False otherwise.

  If there is no module in scope, this function will return True.

  Args:
    axis_col: Name of the collection in question.

  Returns:
    Whether it is currently mutable.
  """
  last = nn.module._context.module_stack[-1]  # pylint: disable=protected-access
  if last:
    return last.is_mutable_collection(axis_col)
  else:
    return True


# uses this variable_transform to change 'params_axes' pytree as it bubbles
# up / out from scan.
def _add_axis_to_metadata(fn, axis_pos, axis_name, axis_col='params_axes'):
  """Insert a named axis to axes metadata."""
  # Handle In() / Out() scan axis marker types.
  if hasattr(axis_pos, 'axis'):
    axis_pos = axis_pos.axis

  def insert_fn_leaf(names):
    if names is None:
      return names
    names = list(names)
    names.insert(axis_pos, axis_name)
    return tuple(names)

  def insert_fn(x):
    new_names = jax.tree_util.tree_map(insert_fn_leaf, x.names,
                                       is_leaf=_is_logical_spec)
    return x.replace(names=new_names)

  def remove_fn_leaf(names):
    if names is None:
      return names
    names = list(names)
    if names[axis_pos] != axis_name:
      raise ValueError(f'Expected axis {axis_name} at position {axis_pos} in '
                       f'axis metadata {names}.')
    names.pop(axis_pos)
    return tuple(names)

  def remove_fn(x):
    new_names = jax.tree_util.tree_map(remove_fn_leaf, x.names,
                                       is_leaf=_is_logical_spec)
    return x.replace(names=new_names)

  return nn.transforms.map_variables(
      fn,
      axis_col,
      mutable=_is_mutable(axis_col),
      trans_in_fn=lambda tree: _tree_map_axes(remove_fn, tree),
      trans_out_fn=lambda tree: _tree_map_axes(insert_fn, tree)
      )


# pylint: disable=dangerous-default-value
def scan_with_axes(
    target: flax.linen.transforms.Target,
    variable_axes: Mapping[flax.core.lift.CollectionFilter,
                           flax.core.lift.InOutScanAxis] = {},
    variable_broadcast: flax.core.lift.CollectionFilter = False,
    variable_carry: flax.core.lift.CollectionFilter = False,
    split_rngs: Mapping[flax.core.lift.PRNGSequenceFilter, bool] = {},
    in_axes=0,
    out_axes=0,
    length: Optional[int] = None,
    reverse: bool = False,
    unroll: int = 1,
    axis_name: str = 'layers',
    axes_collections: Tuple[str, ...] = ('params',),
    data_transform: Optional[Callable[..., Any]] = None,
    methods=None) -> flax.linen.transforms.Target:
  """Wrapped version of nn.scan that handles logical axis metadata."""

  # we broadcast the static metadata collections.
  axes_filters = tuple(f'{col}_axes' for col in axes_collections)
  variable_broadcast = flax.core.scope.union_filters(
      variable_broadcast, axes_filters)

  # perform usual lifted scan
  scanned = flax.linen.transforms.lift_transform(
      flax.core.lift.scan,
      target,
      variable_axes=variable_axes,
      variable_broadcast=variable_broadcast,
      variable_carry=variable_carry,
      split_rngs=split_rngs,
      in_axes=in_axes,
      out_axes=out_axes,
      length=length,
      reverse=reverse,
      unroll=unroll,
      data_transform=data_transform,
      methods=methods)

  # add scan axis to logical axes metadata
  for col in axes_collections:
    if col in variable_axes:
      scanned = _add_axis_to_metadata(scanned,
                                      axis_pos=variable_axes[col],
                                      axis_name=axis_name,
                                      axis_col=f'{col}_axes')
  return scanned


# pylint: disable=dangerous-default-value
def vmap_with_axes(target: flax.linen.transforms.Target,
                   variable_axes: Mapping[flax.core.lift.CollectionFilter,
                                          flax.core.lift.InOutAxis],
                   split_rngs: Mapping[flax.core.lift.PRNGSequenceFilter,
                                       bool] = {},
                   in_axes=0,
                   out_axes=0,
                   axis_size: Optional[int] = None,
                   axis_name: Optional[str] = None,
                   partitioning_axis_names: Mapping[str, str] = {},
                   spmd_axis_name: Optional[str] = None,
                   methods=None) -> flax.linen.transforms.Target:
  """Wrapped version of nn.vmap that handles logical axis metadata."""

  # tell normal vmap to broadcast axis metadata.
  variable_axes = dict(variable_axes)  # shallow copy
  for name in partitioning_axis_names:
    variable_axes[f'{name}_axes'] = None

  # perform usual lifted vmap
  vmapped = flax.linen.transforms.lift_transform(
      flax.core.lift.vmap,
      target,
      variable_axes=variable_axes,
      split_rngs=split_rngs,
      in_axes=in_axes,
      out_axes=out_axes,
      axis_size=axis_size,
      axis_name=axis_name,
      spmd_axis_name=spmd_axis_name,
      methods=methods)

  for collection_name, axis in variable_axes.items():
    if collection_name in partitioning_axis_names:
      vmapped = _add_axis_to_metadata(  # pylint: disable=protected-access
          vmapped,
          axis_pos=axis,
          axis_name=partitioning_axis_names[collection_name],
          axis_col=f'{collection_name}_axes')

  return vmapped


# Remat abstraction bug hotfix
# ------------------------------------------------------------------------------
# TODO(levskaya): upstream this fix into main flax.core.lift.remat.
# Workaround a scan(remat(...)) abstraction bug by manually implementing a
# static_argnums behavior for flax remat via closure before applying jax remat.


def core_remat_static(fn,
                      variables=True,
                      rngs=True,
                      concrete=False,
                      prevent_cse=True,
                      static_argnums=(),
                      policy=None):
  """Flax functional core remat version with static_argnums."""

  static_argnums = tuple(sorted(static_argnums))

  def _repack_remat_args(dyn_args, static_args):
    """Remake arg list from static and dynamic args given static_argnums."""
    args = []
    s_cnt, d_cnt = 0, 0
    for i in range(len(dyn_args) + len(static_args)):
      if i in static_argnums:
        args.append(static_args[s_cnt])
        s_cnt += 1
      else:
        args.append(dyn_args[d_cnt])
        d_cnt += 1
    return tuple(args)

  def inner(scope_fn, repack_fn, variable_groups, rng_groups, *args):
    static_args = tuple(x for i, x in enumerate(args) if i in static_argnums)
    dyn_args = tuple(x for i, x in enumerate(args) if i not in static_argnums)

    @functools.partial(
        jax.remat, concrete=concrete, prevent_cse=prevent_cse, policy=policy)
    @functools.wraps(fn)
    def rematted(variable_groups, rng_groups, *dyn_args):
      args = _repack_remat_args(dyn_args, static_args)
      scope = scope_fn(variable_groups, rng_groups)
      y = fn(scope, *args)
      return y, repack_fn(scope)

    return rematted(variable_groups, rng_groups, *dyn_args)

  return flax.core.lift.pack(
      inner, (variables,), (variables,), (rngs,), name='remat')


def remat(target,
          variables=True,
          rngs=True,
          concrete=False,
          prevent_cse=True,
          static_argnums=(),
          policy=None,
          methods=None):
  """Flax lifted remat that supports static_argnums."""
  return flax.linen.transforms.lift_transform(
      core_remat_static,
      target,
      variables=variables,
      rngs=rngs,
      concrete=concrete,
      prevent_cse=prevent_cse,
      static_argnums=static_argnums,
      policy=policy,
      methods=methods)
