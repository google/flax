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

"""Jax transform lifting."""

import collections
from dataclasses import dataclass
import functools


import jax
from jax import random

from typing import Any, Callable, Sequence, Union, Iterable, Optional, Mapping, TypeVar, Generic

from .frozen_dict import freeze
from .frozen_dict import FrozenDict
from .frozen_dict import unfreeze

from .scope import Scope, CollectionFilter, PRNGSequenceFilter, in_filter, union_filters, intersect_filters, group_collections

from . import axes_scan


T = TypeVar('T')


def _dedup_scopes(scopes):
  paths = []
  # must preseve insertion order for duplication to work correctly
  minimal_set = collections.OrderedDict((s, ()) for s in scopes)
  for leaf in scopes:
    scope = leaf.parent
    max_parent = leaf
    max_parent_path = ()
    path = [leaf.name]
    while scope is not None:
      if scope in minimal_set:
        max_parent = scope
        max_parent_path = tuple(reversed(path))
      path.append(scope.name)
      scope = scope.parent
    if max_parent is not leaf and leaf in minimal_set:
      del minimal_set[leaf]
    paths.append((max_parent, max_parent_path))
  return tuple(minimal_set), tuple(paths)


def _dup_scopes(orig_scopes, scopes, paths):
  mapping = dict(zip(orig_scopes, scopes))
  scopes = []
  for root, path in paths:
    scope = mapping[root]
    for name in path:
      scope = scope.push(name, reuse=True)
    scopes.append(scope)
  return scopes

def _transpose(xs):
  return tuple(zip(*xs))

def pack(fn: Callable[..., Any],
         in_variable_filters: Sequence[CollectionFilter],
         out_variable_filters: Sequence[CollectionFilter],
         rng_filters: Sequence[PRNGSequenceFilter],
         name=None) -> Callable[..., Any]:
  """Pack variables and rngs for functional transformations.

  The pack function is the building block for all other lifted transformations.
  """
  @functools.wraps(fn)
  def wrapper(scope_tree: Scope, *args):
    # pylint: disable=protected-access
    scopes, treedef = jax.tree_flatten(scope_tree)
    scopes, paths = _dedup_scopes(scopes)

    variable_groups_xs = []

    for scope in scopes:
      scope._validate_trace_level()
      scope._populate_collections()
      variable_groups_xs.append(group_collections(
          scope._variables, in_variable_filters))
    variable_groups_xs_t = _transpose(variable_groups_xs)

    # Make sure that in-only variable collections are frozen
    for variable_group_xs in variable_groups_xs_t:
      for variable_group in variable_group_xs:
        for col_name, collection in variable_group.items():
          col_in_out = any(
              in_filter(col_filter, col_name)
              for col_filter in out_variable_filters)
          if not col_in_out:
            variable_group[col_name] = freeze(collection)
    rng_groups_xs = []
    for scope in scopes:
      rng_groups = group_collections(scope.rngs, rng_filters)
      for rng_group in rng_groups:
        for kind in rng_group:
          rng_group[kind] = scope.make_rng(kind)
      rng_groups_xs.append(rng_groups)
    rng_groups_xs_t = _transpose(rng_groups_xs)

    inner_scopes = []
    def scope_fn(variable_groups_xs_t, rng_groups_xs_t):
      nonlocal inner_scopes
      for inner_scope in inner_scopes:
        inner_scope.invalidate()
      inner_scopes = []
      mutable = False
      for out_filter in out_variable_filters:
        mutable = union_filters(mutable, out_filter)
      # could be () in the edge case where no rngs or variable_groups are lifted
      # in this case fallback to ((),) * len(scopes) to make sure the zip has something
      # to iterate over for each scope.
      variable_groups_xs = _transpose(variable_groups_xs_t) or ((),) * len(scopes)
      rng_groups_xs = _transpose(rng_groups_xs_t) or ((),) * len(scopes)
      assert len(variable_groups_xs) == len(scopes)
      assert len(rng_groups_xs) == len(scopes)
      for variable_groups, rng_groups, scope in zip(variable_groups_xs, rng_groups_xs, scopes):
        variables = {}
        rngs = {}
        for variable_group in variable_groups:
          variables.update(variable_group)
        for rng_group in rng_groups:
          rngs.update(rng_group)
        # make sure variable dicts are cloned and can't be manipulated by ref sharing.
        variables = jax.tree_map(lambda x: x, variables)
        scope_mutable = intersect_filters(scope.root.mutable, mutable)
        new_path = scope.path
        if name:
          if new_path:
            new_path = new_path[:-1] + (f'{name}({new_path[-1]})',)
          else:
            new_path = (f'{name}()',)
        inner_scope = Scope(
            variables, name=scope.name, rngs=rngs,
            mutable=scope_mutable, parent=None,
            path=new_path)
        inner_scopes.append(inner_scope)
      inner_scopes = _dup_scopes(scopes, inner_scopes, paths)
      return treedef.unflatten(inner_scopes)

    def repack(inner_scope_tree):
      inner_scopes = treedef.flatten_up_to(inner_scope_tree)
      inner_scopes, inner_paths = _dedup_scopes(inner_scopes)
      inner_scopes = list(inner_scopes)
      assert [p for _, p in paths] == [p for _, p in inner_paths]
      out_variable_groups_xs = []
      for inner_scope in inner_scopes:
        inner_scope.invalidate()
        inner_scope._validate_trace_level()
        mutable_variables = {key: val for key, val
                             in inner_scope._variables.items()
                             if not isinstance(val, FrozenDict)}
        out_variable_groups = group_collections(
            mutable_variables, tuple(out_variable_filters) + (True,))
        remainder = tuple(out_variable_groups[-1].keys())
        if remainder:
          raise ValueError(f'unmapped output variables: {remainder}')
        out_variable_groups_xs.append(out_variable_groups[:-1])

      return _transpose(out_variable_groups_xs)

    try:
      y, out_variable_groups_xs_t = fn(
          scope_fn, repack,
          variable_groups_xs_t, rng_groups_xs_t,
          *args)
    finally:
      for inner_scope in inner_scopes:
        inner_scope.invalidate()
    out_variable_groups_xs = _transpose(out_variable_groups_xs_t)
    for scope, out_variable_groups in zip(scopes, out_variable_groups_xs):
      for out_variable_group in out_variable_groups:
        for col_name, collection in out_variable_group.items():
          for var_name, value in collection.items():
            scope.put_variable(col_name, var_name, value)
    return y
  return wrapper


id_fn = lambda x: x


def transform(
    fn: Callable[..., Any],
    target: CollectionFilter,
    trans_in_fn: Callable[..., Any] = id_fn,
    trans_out_fn: Callable[..., Any] = id_fn,
    init: bool = False, mutable: bool = False,
    rngs: PRNGSequenceFilter = True, variables: CollectionFilter = True):
  """Locally transform Variables inside a scope.

  Args:
    fn: the function to be transformed.
    target: the collection(s) to be transformed.
    trans_in_fn: creates a view of the target variables.
    trans_out_fn: transforms the updated variables in the view after mutation.
    init: If True, variables are initialized before transformation.
    rngs: PRNGSequences added to the transformed scope (default: all).
    variables: Addtional Variable collections added to the transformed scope.
      Besides those specified by `target` (default: all).
  """
  def wrapper(scope_fn, repack, variable_groups, rng_groups, treedef, *args):
    target, variables = variable_groups
    if init:
      scope = scope_fn((target, variables), rng_groups)
      fn(scope, *args)
      target, _ = repack(scope)
      target_tree = trans_out_fn(treedef.unflatten(target))
      target = treedef.flatten_up_to(target_tree)
    target_tree = treedef.unflatten(map(unfreeze, target))
    target_tree = trans_in_fn(target_tree)
    target = treedef.flatten_up_to(target_tree)
    if not is_target_out:
      target = tuple(map(freeze, target))
    scope = scope_fn((target, variables), rng_groups)
    y = fn(scope, *args)
    out_target, out_vars = repack(scope)
    if is_target_out:
      out_target_tree = trans_out_fn(treedef.unflatten(out_target))
      out_target = treedef.flatten_up_to(out_target_tree)
    return y, (out_target, out_vars)

  is_target_out = mutable or init
  in_vars = (target, variables)
  out_vars = (target, variables) if is_target_out else ((), variables)
  wrapper = pack(wrapper, in_vars, out_vars, (rngs,), name='transform')
  @functools.wraps(wrapper)
  def catch_treedef(scopes, *args):
    treedef = jax.tree_structure(scopes)
    return wrapper(scopes, treedef, *args)
  return catch_treedef


def transform_module(fn: Callable[..., Any],
                     target: CollectionFilter = 'params',
                     trans_in_fn: Callable[..., Any] = id_fn,
                     trans_out_fn: Callable[..., Any] = id_fn,
                     mutable: bool = False,
                     rngs: PRNGSequenceFilter = True,
                     variables: CollectionFilter = True):
  """"Wrapper around `transform` for automatic init detection.

  This function will detect if the target collection exists.
  If it doesn't `init=True` is will be passed to `transform`.

  See `transform` for more details.
  """
  def wrapper(scope, *args, **kwargs):
    vs = scope.variables()
    is_init = target not in vs or not vs[target]
    fn_p = functools.partial(fn, **kwargs)
    lift_trans = transform(
        fn_p,
        target,
        trans_in_fn=trans_in_fn,
        trans_out_fn=trans_out_fn,
        init=is_init, mutable=mutable,
        rngs=rngs, variables=variables)
    return lift_trans(scope, *args)
  return wrapper


def swap_collection(fn: Callable[..., Any], col_a: str, col_b: str):
  """Swap two collections."""
  def swap(target):
    a = target[col_a] if col_a in target else {}
    b = target[col_b] if col_b in target else {}
    target[col_b], target[col_a] = a, b
    return target

  return transform(fn, (col_a, col_b), swap, swap, mutable=True)


@dataclass(frozen=True)
class In(Generic[T]):
  """Specifies a variable collection should only be lifted as input."""
  axis: Any  # pytype does not support generic variable annotation

@dataclass(frozen=True)
class Out(Generic[T]):
  """Specifies a variable collection should only be lifted as output."""
  axis: Any  # pytype does not support generic variable annotation


def _split_in_out_axes(xs: Mapping[CollectionFilter, Any]):
  unpack = lambda v: v.axis if isinstance(v, (In, Out)) else v
  in_axes = {k: unpack(v) for k, v in xs.items() if not isinstance(v, Out)}
  out_axes = {k: unpack(v) for k, v in xs.items() if not isinstance(v, In)}
  return in_axes, out_axes


Axis = Optional[int]
InOutAxis = Union[Axis, In[Axis], Out[Axis]]


def vmap(fn: Callable[..., Any],
         variable_axes: Mapping[CollectionFilter, InOutAxis],
         split_rngs: Mapping[PRNGSequenceFilter, bool],
         in_axes=0, out_axes=0, axis_size=None, axis_name=None) -> Callable[..., Any]:
  """A lifted version of `jax.vmap`.

  See `jax.vmap` for the unlifted batch transform in Jax.

  Example::

    # a dense mapping with separate parameters for axis 0.
    batch_dense = lift.vmap(
        nn.dense,
        in_axes=(0, None),
        variable_axes={'params': 0},
        split_rngs={'params': True})

  Args:
    fn: the function to be transformed.
    variable_axes: the variable collections that are lifted into the
      batching transformation. Use `None` to indicate a broadcasted
      collection or an integer to map over an axis.
    split_rngs: Split PRNG sequences will be different for each index
      of the batch dimension. Unsplit PRNGs will be broadcasted.
    in_axes: Specifies the mapping of the input arguments (see `jax.vmap).
    out_axes: Specifies the mapping of the return value (see `jax.vmap).
    axis_size: Specifies the size of the batch axis. This only needs
      to be specified if it cannot be derived from the input arguments.
    axis_name: Specifies a name for the batch axis. Can be used together
      with parallel reduction primitives (e.g. `jax.lax.pmean`,
      `jax.lax.ppermute`, etc.)
  """
  variable_in_axes, variable_out_axes = _split_in_out_axes(variable_axes)
  variable_in_groups, variable_in_axes = _unzip2(variable_in_axes.items())
  variable_out_groups, variable_out_axes = _unzip2(variable_out_axes.items())
  rng_groups, rng_splits = _unzip2(split_rngs.items())
  rng_axes = tuple(0 if rng_split else None for rng_split in rng_splits)

  def inner(scope_fn, repack_fn, variable_groups, rng_groups, *args):
    def find_axis_size(axis, x):
      if axis is not None:
        leaves = jax.tree_leaves(x)
        if leaves:
          return leaves[0].shape[axis]
      return ()

    # split rngs
    axis_sizes = jax.tree_multimap(find_axis_size, (variable_in_axes, in_axes), (variable_groups, args))
    axis_sizes = set(jax.tree_leaves(axis_sizes))
    if axis_size is None and len(axis_sizes) == 1:
      d_axis_size, = axis_sizes
    elif len(axis_sizes) > 1:
      raise ValueError(f'Inconsistent batch axis sizes: {axis_sizes}')
    elif axis_size is None:
      raise ValueError('axis_size should be specified manually.')
    else:
      d_axis_size = axis_size
    split_fn = lambda rng: random.split(rng, d_axis_size)

    rng_groups = tuple(
        jax.tree_map(split_fn, rng_group) if split else rng_group
        for rng_group, split in zip(rng_groups, rng_splits))

    @functools.partial(jax.vmap,
                       in_axes=(variable_in_axes, rng_axes, in_axes),
                       out_axes=(out_axes, variable_out_axes),
                       axis_name=axis_name)
    @functools.wraps(fn)
    def mapped(variable_groups, rng_groups, args):
      scope = scope_fn(variable_groups, rng_groups)
      y = fn(scope, *args)
      return y, repack_fn(scope)

    return mapped(variable_groups, rng_groups, args)

  return pack(
      inner, variable_in_groups, variable_out_groups, rng_groups,
      name='vmap')


ScanAxis = int
InOutScanAxis = Union[ScanAxis, In[ScanAxis], Out[ScanAxis]]


def scan(fn: Callable[..., Any],
         variable_axes: Mapping[CollectionFilter, InOutScanAxis] = {},
         variable_broadcast: CollectionFilter = False,
         variable_carry: CollectionFilter = False,
         split_rngs: Mapping[PRNGSequenceFilter, bool] = {},
         in_axes=0, out_axes=0,
         length: Optional[int] = None,
         reverse: bool = False) -> Callable[..., Any]:
  """A lifted version of `jax.lax.scan`.

  See `jax.lax.scan` for the unlifted scan in Jax.

  Example::

    scope.variable('counter', 'i', jnp.zeros, ())
    def body_fn(scope, c, x):
      counter = scope.variable('counter', 'i', jnp.zeros, ())
      counter.value += 1
      x = scope.child(nn.dense)(x, 1)
      return c, x

    _, ys = lift.scan(
        body_fn,
        variable_carry='counter',
        variable_broadcast='params',
        split_rngs={'params': False})(scope, (), xs)

  Args:
    fn: the function to be transformed.
    variable_axes: the variable collections that are scanned over.
    variable_broadcast: Specifies the broadcasted variable collections.
      A broadcasted variable should not depend on any computation that cannot be lifted out of the loop.
      This is typically used to define shared parameters inside the fn.
    variable_carry: Specifies the variable collections that are carried through the loop.
      Mutations to these variables are carried to the next iteration and will be preserved
      when the scan finishes.
    split_rngs: Split PRNG sequences will be different for each loop iterations.
      If split is False the PRNGs will be the same across iterations.
    in_axes: Specifies the axis to scan over for the arguments. Should be a prefix
      tree of the arguments. Use `flax.core.broadcast` to feed an entire input
      to each iteration of the scan body.
    out_axes: Specifies the axis to scan over for the return value. Should be a prefix
      tree of the return value.
    length: Specifies the number of loop iterations. This only needs
      to be specified if it cannot be derivied from the scan arguments.
    reverse: If true, scan from end to start in reverse order.
  """
  variable_in_axes, variable_out_axes = _split_in_out_axes(variable_axes)
  variable_in_groups, variable_in_axes = _unzip2(variable_in_axes.items())
  variable_out_groups, variable_out_axes = _unzip2(variable_out_axes.items())
  assert all(isinstance(ax, int) for ax in variable_in_axes)
  assert all(isinstance(ax, int) for ax in variable_out_axes)
  rng_groups, rng_splits = _unzip2(split_rngs.items())
  rng_axes = tuple(0 if rng_split else axes_scan.broadcast
                   for rng_split in rng_splits)

  def inner(scope_fn, repack_fn,
            variable_groups, rng_groups,
            init, *args):
    def find_length(axis, x):
      if axis is not axes_scan.broadcast:
        leaves = jax.tree_leaves(x)
        if leaves:
          return leaves[0].shape[axis]
      return ()
    # split rngs
    lengths = jax.tree_multimap(find_length, in_axes, args)
    lengths = set(jax.tree_leaves(lengths))
    if length is None and len(lengths) == 1:
      d_length, = lengths
    elif len(lengths) > 1:
      raise ValueError(f'Inconsistent scan lengths: {lengths}')
    elif length is None:
      raise ValueError('length should be specified manually.')
    else:
      d_length = length
    split_fn = lambda rng: random.split(rng, d_length)

    rng_groups = tuple(
        jax.tree_map(split_fn, rng_group) if split else rng_group
        for rng_group, split in zip(rng_groups, rng_splits))

    @functools.partial(axes_scan.scan,
                       in_axes=(variable_in_axes, rng_axes, in_axes),
                       out_axes=(out_axes, variable_out_axes),
                       length=length, reverse=reverse)
    def scanned(broadcast_vars, carry, scan_variable_groups, rng_groups, args):
      carry_vars, c = carry
      variable_groups = (broadcast_vars, carry_vars) + scan_variable_groups
      scope = scope_fn(variable_groups, rng_groups)
      c, y = fn(scope, c, *args)
      out_vars = repack_fn(scope)
      broadcast_vars_out = out_vars[0]
      carry_vars = out_vars[1]
      scan_vars = out_vars[2:]
      # add immutable broadcast vars back to broadcast output
      # otherwise they won't be fed to the actual scan body
      for in_group, out_group in zip(broadcast_vars, broadcast_vars_out):
        for col in in_group:
          if col not in out_group:
            out_group[col] = in_group[col]
      return broadcast_vars_out, (carry_vars, c), (y, scan_vars)

    broadcast_vars = variable_groups[0]
    carry_vars = variable_groups[1]
    scan_vars = variable_groups[2:]
    broadcast_vars, (carry_vars, c), (ys, scan_vars) = scanned(
        broadcast_vars, (carry_vars, init), scan_vars, rng_groups, args)
    # remove immutable broadcast vars otherwise they will be updated
    # with their own value which will cause an error
    for out_group in broadcast_vars:
      for name, col in tuple(out_group.items()):
        if isinstance(col, FrozenDict):
          del out_group[name]
    out_vars = (broadcast_vars, carry_vars,) + scan_vars
    return (c, ys), out_vars

  return pack(
      inner,
      (variable_broadcast, variable_carry) + variable_in_groups,
      (variable_broadcast, variable_carry) + variable_out_groups,
      rng_groups,
      name='scan')


def custom_vjp(fn: Callable[..., Any], backward_fn: Callable[..., Any],
               grad_kind: CollectionFilter = 'params',
               nondiff_argnums=()):
  """"Lifted version of `jax.custom_vjp`.

  `backward_fn` defines a custom vjp (backward gradient) for `fn`.

  Example::

    def fwd(scope, x, features):
      y = nn.dense(scope, x, features)
      return y, x

    def bwd(features, scope_fn, params, res, g):
      x = res
      fn = lambda params, x: nn.dense(scope_fn(params), x, features)
      _, pullback = jax.vjp(fn, params, x)
      g_param, g_x = pullback(g)
      g_param = jax.tree_map(jnp.sign, g_param)
      return g_param, g_x

    dense_sign_grad = lift.custom_vjp(fwd, backward_fn=bwd, nondiff_argnums=(2,))

  Args:
    fn: should return a tuple of output and auxilliary data for the backward pass.
    backward_fn: arguments are passed as (*nondiff_args, scope_fn, grad_variables, aux, g_y)
      where scope_fn takes grad_variables to create the scope,
      aux is the auxilliary data returend by `fn`,
      and g_y is the tangent of y.
  """
  # TODO(jheek) is this transform general/flexible enough?
  def inner(scope_fn, repack_fn, variable_groups, rng_groups, *args):
    grad_variables, other_variables = variable_groups

    def simple_scope_fn(grad_variables):
      grad_variables = tuple(freeze(x) for x in grad_variables)
      return scope_fn((grad_variables, other_variables), rng_groups)

    def f(grad_variables, *args):
      scope = scope_fn((grad_variables, other_variables), rng_groups)
      y, _ = fn(scope, *args)
      vars_out = repack_fn(scope)
      return y, vars_out
    f = jax.custom_vjp(f, nondiff_argnums=nondiff_argnums)

    def f_fwd(grad_variables, *args):
      scope = simple_scope_fn(grad_variables)
      y, res = fn(scope, *args)
      vars_out = repack_fn(scope)
      return (y, vars_out), (res, grad_variables)

    def f_bwd(*args):
      nondiff_args = args[:-2]
      res, g = args[-2:]
      g_y, _ = g
      user_res, grad_variables = res
      return backward_fn(*nondiff_args, simple_scope_fn, grad_variables, user_res, g_y)

    f.defvjp(f_fwd, f_bwd)

    return f(grad_variables, *args)

  variable_in_groups = (grad_kind, True,)
  variable_out_groups = (grad_kind, True,)
  rng_groups = (True,)
  return pack(
      inner, variable_in_groups, variable_out_groups, rng_groups,
      name='custom_vjp')


def remat(fn: Callable[..., Any],
          variables: CollectionFilter = True,
          rngs: PRNGSequenceFilter = True) -> Callable[..., Any]:
  """Lifted version of jax.remat."""
  def inner(scope_fn, repack_fn, variable_groups, rng_groups, *args):
    @jax.remat
    @functools.wraps(fn)
    def rematted(variable_groups, rng_groups, *args):
      scope = scope_fn(variable_groups, rng_groups)
      y = fn(scope, *args)
      return y, repack_fn(scope)

    return rematted(variable_groups, rng_groups, *args)
  return pack(inner, (variables,), (variables,), (rngs,), name='remat')


def jit(fn: Callable[..., Any],
        static_argnums: Union[int, Iterable[int]] = (),
        device=None,
        backend: Union[str, None] = None,
        variables: CollectionFilter = True,
        rngs: PRNGSequenceFilter = True) -> Callable[..., Any]:
  """Lifted version of jax.jit."""
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

  return pack(inner, (variables,), (variables,), (rngs,), name='jit')


def remat_scan(body_fn: Callable[..., Any], scope: Scope, carry: Any,
               lengths: Sequence[int],
               variable_broadcast: CollectionFilter = False,
               variable_carry: CollectionFilter = False,
               variable_axes: Mapping[CollectionFilter, InOutScanAxis] = {},
               split_rngs: Mapping[PRNGSequenceFilter, bool] = {}):
  """Combines `lift.remat` and `lift.scan` for memory efficient scans.

  Example::
    def body_fn(scope, x):
      return nn.dense(scope, x, features=x.shape[-1])

    # 100x dense with O(sqrt(N)) memory for gradient computation
    y = lift.remat_scan(
        body_fn, scope, x, lengths=(10, 10),
        variable_axes={'params': 0},
        split_rngs={'params': True})
  """
  # TODO(jheek) should remat scan have scan inputs/outputs?
  scan_fn = functools.partial(
      scan,
      variable_broadcast=variable_broadcast,
      variable_carry=variable_carry,
      variable_axes=variable_axes,
      split_rngs=split_rngs)
  if len(lengths) == 1:
    def wrapper(scope, carry):
      return body_fn(scope, carry), ()
    carry, _ = scan_fn(wrapper, length=lengths[0])(scope, carry)
  else:
    @remat
    def inner_loop(scope, carry):
      carry = remat_scan(body_fn, scope, carry, lengths[1:],
                         variable_broadcast, variable_carry,
                         variable_axes, split_rngs)
      return carry, ()
    carry, _ = scan_fn(inner_loop, length=lengths[0])(scope, carry)
  return carry


def named_call(fn: Callable[..., Any], name: str) -> Callable[..., Any]:
  """Adds a name scope to `fn` during profiling."""
  def inner(scope_fn, repack_fn, variable_groups, rng_groups, args, kwargs):
    @functools.wraps(fn)
    def named(variable_groups, rng_groups):
      scope = scope_fn(variable_groups, rng_groups)
      y = fn(scope, *args, **kwargs)
      return y, repack_fn(scope)
    named = jax.named_call(named, name=name)
    return named(variable_groups, rng_groups)
  lifted = pack(inner, (True,), (True,), (True,))
  def wrapper(scope, *args, **kwargs):
    return lifted(scope, args, kwargs)
  return wrapper


def _unzip2(xs):
  ys = tuple(zip(*xs))
  return ys if ys else ((), ())
