
import enum
import functools


import jax
from jax import random
from jax import lax
from jax import numpy as jnp

from jax.interpreters import partial_eval as pe
from jax import linear_util as lu

from typing import Any, Callable, Sequence, Union, Iterable, Tuple, Optional, Mapping

from .frozen_dict import freeze
from .frozen_dict import FrozenDict
from .frozen_dict import unfreeze

from .scope import Scope, KindFilter, in_kind_filter, group_kinds

scan_variable_modes = set(['carry', 'broadcast', 'scan', None])

ScanVariableMode = Union[str, Tuple[str, str]]

def pack(fn: Callable[..., Any],
         in_variable_filters: Sequence[KindFilter],
         out_variable_filters: Sequence[KindFilter],
         rng_filters: Sequence[KindFilter]) -> Callable[..., Any]:
  """Pack variables and rngs for functional transformations."""
  @functools.wraps(fn)
  def wrapper(scope: Scope, *args):
    # pylint: disable=protected-access
    scopes, treedef = jax.tree_flatten(scope)
    # TODO(jheek) check aliasing between scopes!!!

    variable_groups_xs = []

    for scope in scopes:
      scope._validate_trace_level()
      scope._populate_kinds()
      variable_groups_xs.append(group_kinds(scope._variables, in_variable_filters))
    # Make sure in only variable kinds are frozen
    for variable_groups in variable_groups_xs:
      for variable_group in variable_groups:
        for kind, kind_variables in variable_group.items():
          kind_in_out = any(
              in_kind_filter(kind_filter, kind)
              for kind_filter in out_variable_filters)
          if not kind_in_out:
            variable_group[kind] = freeze(kind_variables)
    rng_groups_xs = []
    for scope in scopes:
      rng_groups = group_kinds(scope.rngs, rng_filters)
      for rng_group in rng_groups:
        for kind in rng_group:
          rng_group[kind] = scope.make_rng(kind)
      rng_groups_xs.append(rng_groups)

    inner_scopes = []
    def scope_fn(variable_groups_xs, rng_groups_xs):
      nonlocal inner_scopes
      for inner_scope in inner_scopes:
        inner_scope.invalidate()
      inner_scopes = []
      for variable_groups, rng_groups in zip(variable_groups_xs, rng_groups_xs):
        variables = {}
        rngs = {}
        for variable_group in variable_groups:
          variables.update(variable_group)
        for rng_group in rng_groups:
          rngs.update(rng_group)
        # make sure variable dicts are cloned and can't be manipulated by ref sharing.
        variables = jax.tree_map(lambda x: x, variables)
        inner_scope = Scope(variables, name=scope.name, rngs=rngs, parent=None)
        inner_scopes.append(inner_scope)
      return treedef.unflatten(inner_scopes)

    def repack(inner_scope_tree):
      inner_scopes = treedef.flatten_up_to(inner_scope_tree)
      out_variable_groups_xs = []
      for inner_scope in inner_scopes:
        inner_scope.invalidate()
        inner_scope._validate_trace_level()
        mutable_variables = {key: val for key, val
                              in inner_scope._variables.items()
                              if not isinstance(val, FrozenDict)}
        out_variable_groups = group_kinds(
            mutable_variables, tuple(out_variable_filters) + (True,))
        remainder = tuple(out_variable_groups[-1].keys())
        if remainder:
          raise ValueError(f'unmapped output variables: {remainder}')
        out_variable_groups_xs.append(out_variable_groups[:-1])
      return tuple(out_variable_groups_xs)
    try:
      y, out_variable_groups_xs = fn(
          scope_fn, repack, tuple(variable_groups_xs), tuple(rng_groups_xs), *args)
    finally:
      for inner_scope in inner_scopes:
        inner_scope.invalidate()
    for scope, out_variable_groups in zip(scopes, out_variable_groups_xs):
      for out_variable_group in out_variable_groups:
        for kind, kind_variables in out_variable_group.items():
          for name, value in kind_variables.items():
            scope.put_variable(kind, name, value)
    return y
  return wrapper

id_fn = lambda x: x

def transform_module(fn: Callable[..., Any],
                     target: KindFilter = 'param',
                     trans_in_fn: Callable[..., Any] = id_fn,
                     trans_out_fn: Callable[..., Any] = id_fn,
                     init: bool = True, mutable: bool = False,
                     rngs: KindFilter = True,
                     variables: KindFilter = True):
  def wrapper(scope, *args, **kwargs):
    if init:
      vs = scope.variables()
      is_init = target not in vs or not vs[target]
    else:
      is_init = False
    lift_trans = transform(
        target,
        trans_in_fn=trans_in_fn,
        trans_out_fn=trans_out_fn,
        init=is_init, mutable=mutable,
        rngs=rngs, variables=variables)
    fn_p = functools.partial(fn, **kwargs)
    return lift_trans(scope, fn_p, *args)
  return wrapper


def transform(
    target: KindFilter,
    trans_in_fn: Callable[..., Any] = id_fn,
    trans_out_fn: Callable[..., Any] = id_fn,
    init: bool = False, mutable: bool = False,
    rngs: KindFilter = True, variables: KindFilter = True):
  def wrapper(scope_fn, repack, variable_groups_xs, rng_groups_xs, fn, *args):
    assert len(variable_groups_xs) == 1, 'transform does not support multi-scope lifting.'
    target, variables = variable_groups_xs[0]
    if init:
      scope = scope_fn(((target, variables),), rng_groups_xs)
      fn(scope, *args)
      target, _ = repack(scope)[0]
      target = trans_out_fn(target)
    target = trans_in_fn(unfreeze(target))
    if not is_target_out:
      target = freeze(target)
    scope = scope_fn(((target, variables),), rng_groups_xs)
    y = fn(scope, *args)
    out_target, out_vars = repack(scope)[0]
    if is_target_out:
      out_target = trans_out_fn(out_target)
    return y, ((out_target, out_vars),)

  is_target_out = mutable or init
  in_vars = (target, variables)
  out_vars = (target, variables) if is_target_out else ((), variables)
  wrapper = pack(wrapper, in_vars, out_vars, (rngs,))
  return wrapper


def swapkind(from_kind: str, to_kind: str):
  def swap(target):
    a = target[from_kind] if from_kind in target else {}
    b = target[to_kind] if to_kind in target else {}
    target[to_kind], target[from_kind] = a, b
    return target

  return transform((from_kind, to_kind), swap, swap, mutable=True)


def vmap(fn: Callable[..., Any],
         variable_in_axes: Mapping[KindFilter, Optional[int]],
         variable_out_axes: Mapping[KindFilter, Optional[int]],
         split_rngs: Mapping[KindFilter, bool],
         in_axes=0, out_axes=0, axis_size=None) -> Callable[..., Any]:
  """Wraps jax.vmap."""
  variable_in_groups, variable_in_axes = _unzip2(variable_in_axes.items())
  variable_out_groups, variable_out_axes = _unzip2(variable_out_axes.items())
  rng_groups, rng_splits = _unzip2(split_rngs.items())
  rng_axes = tuple(0 if rng_split else None for rng_split in rng_splits)

  def inner(scope_fn, repack_fn, variable_groups_xs, rng_groups_xs, *args):
    def find_axis_size(axis, x):
      if axis is not None:
        leaves = jax.tree_leaves(x)
        if leaves:
          return leaves[0].shape[axis]
      return ()
    # split rngs
    axis_sizes = jax.tree_multimap(find_axis_size, in_axes, args)
    if axis_size is None:
      d_axis_size, = set(jax.tree_leaves(axis_sizes))
    else:
      d_axis_size = axis_size
    split_fn = lambda rng: random.split(rng, d_axis_size)

    def split_rngs(rng_groups):
      return tuple(
        jax.tree_map(split_fn, rng_group) if split else rng_group
        for rng_group, split in zip(rng_groups, rng_splits))

    rng_groups_xs = tuple(map(split_rngs, rng_groups_xs))

    n = len(variable_groups_xs)
    variable_in_axes_xs = (variable_in_axes,) * n
    variable_out_axes_xs = (variable_out_axes,) * n
    rng_axes_xs = (rng_axes,) * n

    @functools.partial(jax.vmap,
                       in_axes=(variable_in_axes_xs, rng_axes_xs, in_axes),
                       out_axes=(out_axes, variable_out_axes_xs))
    @functools.wraps(fn)
    def mapped(variable_groups_xs, rng_groups_xs, args):
      scope = scope_fn(variable_groups_xs, rng_groups_xs)
      y = fn(scope, *args)
      return y, repack_fn(scope)

    return mapped(variable_groups_xs, rng_groups_xs, args)

  return pack(
      inner, variable_in_groups, variable_out_groups, rng_groups)


def scan(
    fn: Callable[..., Any], scope: 'Scope', init_carry: Any, xs: Any,
    variable_modes: Mapping[KindFilter, ScanVariableMode],
    split_rngs: Mapping[KindFilter, bool],
    length: Optional[int] = None, reverse: bool = False) -> Callable[..., Any]:
  """Wraps jax.lax.scan."""
  # TODO(jheek) scan can be simplified dramatically be making a version of lax.scan
  # that follows an in_axis, out_axis api.
  if length is None:
    length, = set(x.shape[0] for x in jax.tree_leaves(xs))
  variable_groups, variable_modes = _unzip2(variable_modes.items())

  def parse_mode(mode):
    if isinstance(mode, str):
      mode = (mode, mode)
    mode_in, mode_out = mode
    if mode_in not in scan_variable_modes or mode_out not in scan_variable_modes:
      raise ValueError(f'illegal scan variable mode: {mode}')
    return mode
  variable_modes = tuple(parse_mode(m) for m in variable_modes)

  rng_groups, rng_splits = _unzip2(split_rngs.items())
  variable_in_groups = tuple(
      False if mode[0] is None else group
      for group, mode in zip(variable_groups, variable_modes))
  variable_out_groups = tuple(
      False if mode[1] is None else group
      for group, mode in zip(variable_groups, variable_modes))

  def split(variable_groups_xs, i):
    scan_vars_xs = []
    carry_vars_xs = []
    broadcast_vars_xs = []
    for variable_groups in variable_groups_xs:
      scan_vars = tuple(
          group if mode[i] == 'scan' else {}
          for group, mode in zip(variable_groups, variable_modes))
      carry_vars = tuple(
          group if mode[i] == 'carry' else {}
          for group, mode in zip(variable_groups, variable_modes))
      broadcast_vars = tuple(
          group if mode[i] == 'broadcast' else {}
          for group, mode in zip(variable_groups, variable_modes))
      scan_vars_xs.append(scan_vars)
      carry_vars_xs.append(carry_vars)
      broadcast_vars_xs.append(broadcast_vars)
    return tuple(scan_vars_xs), tuple(carry_vars_xs), tuple(broadcast_vars_xs)

  def combine(*variable_groups_xs):
    combined_groups_xs = []
    for groups_xs in zip(*variable_groups_xs):
      combined_groups = []
      for groups in zip(*groups_xs):
        result = {}
        for group in groups:
          result.update(group)
        combined_groups.append(result)
      combined_groups_xs.append(tuple(combined_groups))
    return tuple(combined_groups_xs)

  def inner(scope_fn, repack_fn, variable_groups_xs, rng_groups_xs):
    # split rngs
    split_fn = lambda rng: random.split(rng, length)
    broadcast_rngs_xs = []
    scan_rngs_xs = []
    for rng_groups in rng_groups_xs:
      broadcast_rngs_xs.append(tuple(
          rng_group for rng_group, split
          in zip(rng_groups, rng_splits) if not split))
      scan_rngs_xs.append(tuple(
          jax.tree_map(split_fn, rng_group)
          for rng_group, split in zip(rng_groups, rng_splits) if split))

    def body(carry, xs, init_mode=False):
      carry_vars_xs, c = carry
      scan_vars_xs, scan_rngs_xs, x = xs
      variable_groups_xs = combine(scan_vars_xs, carry_vars_xs, broadcast_vars_xs)
      rng_groups_xs = []
      for broadcast_rngs, scan_rngs in zip(broadcast_rngs_xs, scan_rngs_xs):
        rng_groups_xs.append(broadcast_rngs + scan_rngs)
      scope = scope_fn(variable_groups_xs, rng_groups_xs)
      carry, y = fn(scope, c, x)
      out_vars = repack_fn(scope)
      scan_vars_xs, carry_vars_out_xs, broadcast_vars_out_xs = split(out_vars, 1)

      # TODO(jheek) more informative error check
      def check_shapes(c_in, c_out):
        if not isinstance(c_in, jnp.ndarray) or not isinstance(c_out, jnp.ndarray):
          return
        if jnp.shape(c_in) != jnp.shape(c_out) or jnp.dtype(c_in) != jnp.dtype(c_out):
          raise ValueError()
      try:
        jax.tree_multimap(check_shapes, carry_vars_xs, carry_vars_out_xs)
      except ValueError:
        raise ValueError('carry variables must have the same shape and dtype before and after scan.')

      if init_mode:
        return broadcast_vars_out_xs
      else:
        return (carry_vars_out_xs, carry), (scan_vars_xs, y)
    broadcast_body = functools.partial(body, init_mode=True)

    scan_vars_xs, carry_vars_xs, broadcast_vars_xs = split(variable_groups_xs, 0)
    carry0 = (carry_vars_xs, init_carry)
    xxs = (scan_vars_xs, scan_rngs_xs, xs)

    # use partial evaluation to find the variables that are broadcasted out
    # an error is thrown if a broadcasted output has a dependency on any scan variables
    carry_pvals = jax.tree_map(
        lambda x: pe.PartialVal.unknown(jax.ShapedArray(x.shape, x.dtype)),
        carry0)
    scan_pvals = jax.tree_map(
        lambda x: pe.PartialVal.unknown(jax.ShapedArray(x.shape[1:], x.dtype)),
        xxs)
    input_pvals = (carry_pvals, scan_pvals)
    in_pvals, in_tree = jax.tree_flatten(input_pvals)
    f_flat, out_tree = jax.api_util.flatten_fun_nokwargs(lu.wrap_init(broadcast_body), in_tree)

    _, out_pvals, _ = pe.trace_to_jaxpr(f_flat, in_pvals)
    # _, out_pvals, _ = pe.trace_to_jaxpr(f_flat, in_pvals, stage_out=True)
    
    out_flat = []
    for pv, const in out_pvals:
      if pv is not None:
        raise ValueError('broadcasted variable has a data dependency on the scan body.')
      out_flat.append(const)

    (carry_vars_xs, carry), (scan_vars_xs, ys) = lax.scan(
        body, carry0, xxs, length=length, reverse=reverse)
    
    broadcast_vars_xs = jax.tree_unflatten(out_tree(), out_flat)
        
    out_vars_xs = combine(carry_vars_xs, scan_vars_xs, broadcast_vars_xs)
    return (carry, ys), out_vars_xs

  return pack(
      inner, variable_in_groups, variable_out_groups, rng_groups)(scope)


def custom_vjp(module_fn: Callable[..., Any], backward_fn: Callable[..., Any],
               grad_kind: KindFilter='param',
               nondiff_argnums=()):
  def inner(scope_fn, repack_fn, variable_groups_xs, rng_groups_xs, *args):
    assert len(variable_groups_xs) == 1, 'transform does not support multi-scope lifting.'
    grad_variables, other_variables = variable_groups_xs[0]

    def simple_scope_fn(grad_variables):
      return scope_fn(((freeze(grad_variables), other_variables),), rng_groups_xs)

    def f(grad_variables, *args):
      scope = scope_fn(((grad_variables, other_variables),), rng_groups_xs)
      y, _ = module_fn(scope, *args)
      vars_out = repack_fn(scope)
      return y, vars_out
    f = jax.custom_vjp(f, nondiff_argnums=nondiff_argnums)

    def f_fwd(grad_variables, *args):
      scope = simple_scope_fn(grad_variables)
      y, res = module_fn(scope, *args)
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
      inner, variable_in_groups, variable_out_groups, rng_groups)


def remat(fn: Callable[..., Any],
          variables: KindFilter = True,
          rngs: KindFilter = True) -> Callable[..., Any]:
  """Wraps jax.jit."""
  def inner(scope_fn, repack_fn, variable_groups, rng_groups, *args):
    @jax.remat
    @functools.wraps(fn)
    def rematted(variable_groups_xs, rng_groups_xs, *args):
      scope = scope_fn(variable_groups_xs, rng_groups_xs)
      y = fn(scope, *args)
      return y, repack_fn(scope)

    return rematted(variable_groups, rng_groups, *args)
  return pack(inner, (variables,), (variables,), (rngs,))


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
  def inner(scope_fn, repack_fn, variable_groups_xs, rng_groups_xs, *args):
    @functools.partial(jax.jit,
                       static_argnums=static_argnums,
                       device=device, backend=backend)
    @functools.wraps(fn)
    def jitted(variable_groups_xs, rng_groups_xs, *args):
      scope = scope_fn(variable_groups_xs, rng_groups_xs)
      y = fn(scope, *args)
      return y, repack_fn(scope)

    return jitted(variable_groups_xs, rng_groups_xs, *args)

  return pack(inner, (in_variables,), (out_variables,), (rngs,))


def remat_scan(body_fn: Callable[..., Any], scope: Scope, carry: Any,
               lengths: Sequence[int],
               variable_modes: Mapping[KindFilter, ScanVariableMode],
               split_rngs: Mapping[KindFilter, bool]):
  # TODO(jheek) should remat scan have scan inputs/outputs?
  if len(lengths) == 1:
    def wrapper(scope, carry, _):
      return body_fn(scope, carry), ()
    carry, _ = scan(
        wrapper, scope, carry, (),
        length=lengths[0],
        variable_modes=variable_modes,
        split_rngs=split_rngs)
  else:
    @remat
    def inner_loop(scope, carry, _):
      carry = remat_scan(body_fn, scope, carry, lengths[1:], variable_modes, split_rngs)
      return carry, ()
    carry, _ = scan(
        inner_loop, scope, carry, (),
        length=lengths[0],
        variable_modes=variable_modes,
        split_rngs=split_rngs)
  return carry


def named_call(fn: Callable[..., Any], name: str) -> Callable[..., Any]:
  """Wraps jax.jit."""
  def inner(scope_fn, repack_fn, variable_groups, rng_groups, args, kwargs):
    @functools.wraps(fn)
    def named(variable_groups, rng_groups):
      scope = scope_fn(variable_groups, rng_groups)
      y = fn(scope, *args, **kwargs)
      return y, repack_fn(scope)
    named = _named_call(named, name)
    return named(variable_groups, rng_groups)
  lifted = pack(inner, (True,), (True,), (True,))
  def wrapper(scope, *args, **kwargs):
    return lifted(scope, args, kwargs)
  return wrapper


def _named_call(f, name):
  _, in_tree = jax.tree_flatten(())
  def named_f(*args, **kwargs):
    lu_f = jax.linear_util.wrap_init(lambda: f(*args, **kwargs))
    flat_f, out_tree = jax.api_util.flatten_fun_nokwargs(lu_f, in_tree)
    out_flat = jax.core.call_p.bind(flat_f, name=name)
    return jax.tree_unflatten(out_tree(), out_flat)
  return named_f

def _unzip2(xs):
  ys = tuple(zip(*xs))
  return ys if ys else ((), ())
