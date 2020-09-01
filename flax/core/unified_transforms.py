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

from dataclasses import dataclass
import functools

import jax
import jax.numpy as jnp
from jax import lax

from jax.interpreters import partial_eval as pe
from jax import linear_util as lu

from typing import Union, Optional, Callable, Any

import numpy as np


@dataclass(frozen=True)
class Scan:
  axis: int

ScanAxis = Optional[int]

class _Broadcast:
  pass

broadcast = _Broadcast()


def scan(
    fn: Callable[..., Any],
    in_axes: Any,
    out_axes: Any,
    length: Optional[int] = None,
    reverse: bool = False):

  def transpose_to_front(axis, xs):
    if axis is broadcast:
      return ()
    def trans(x):
      perm = tuple(range(x.ndim))
      perm = (axis,) + tuple(np.delete(perm, axis))
      return jnp.transpose(x, perm)
    return jax.tree_map(trans, xs)

  def transpose_from_front(axis, xs):
    if axis is broadcast:
      return ()
    def trans(x):
      if axis < 0:
        ax = x.ndim - axis
      else:
        ax = axis
      assert ax < x.ndim
      perm = tuple(range(1, ax + 1)) + (0,) + tuple(range(ax + 1, x.ndim))
      return jnp.transpose(x, perm)
    return jax.tree_map(trans, xs)

  def scan_fn(init, *args):
    xs = jax.tree_multimap(transpose_to_front, in_axes, args)

    def body_fn(c, xs, init_mode=False):
      # inject constants
      xs = jax.tree_multimap(lambda ax, arg, x: (arg if ax is broadcast else x),
                             in_axes, args, xs)
      c, ys = fn(c, *xs)
      
      if init_mode:
        ys = jax.tree_multimap(lambda ax, y: (y if ax is broadcast else ()),
                               out_axes, ys)
        return ys
      else:
        ys = jax.tree_multimap(lambda ax, y: (() if ax is broadcast else y),
                               out_axes, ys)
        return c, ys
      return c, ys
    broadcast_body = functools.partial(body_fn, init_mode=True)

    carry_pvals = jax.tree_map(
        lambda x: pe.PartialVal.unknown(jax.ShapedArray(jnp.shape(x), jnp.result_type(x))),
        init)
    scan_pvals = jax.tree_map(
        lambda x: pe.PartialVal.unknown(jax.ShapedArray(jnp.shape(x)[1:], jnp.result_type(x))),
        xs)
    input_pvals = (carry_pvals, scan_pvals)
    in_pvals, in_tree = jax.tree_flatten(input_pvals)
    f_flat, out_tree = jax.api_util.flatten_fun_nokwargs(lu.wrap_init(broadcast_body), in_tree)
    _, out_pvals, _ = pe.trace_to_jaxpr(f_flat, in_pvals)

    out_flat = []
    for pv, const in out_pvals:
      if pv is not None:
        raise ValueError('broadcasted variable has a data dependency on the scan body.')
      out_flat.append(const)
    constants_out = jax.tree_unflatten(out_tree(), out_flat)

    c, ys = lax.scan(body_fn, init, xs, length=length, reverse=reverse)
    ys = jax.tree_multimap(transpose_from_front, out_axes, ys)
    ys = jax.tree_multimap(lambda ax, const, y: (const if ax is broadcast else y),
                           out_axes, constants_out, ys)
    return c, ys

  return scan_fn


# def loop(c, x, y):
#   print(c, x, y)
#   return c + 1, (x * 2, y * 2)


# f = scan(loop, in_axes=(broadcast, 1), out_axes=(broadcast, 1))
# c, (xs, ys) = f(0, 1., jnp.arange(3)[None])
# print(c, xs, ys)
