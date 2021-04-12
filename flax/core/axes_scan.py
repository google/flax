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

import functools

import jax
import jax.numpy as jnp
from jax import lax

from jax.interpreters import partial_eval as pe
from jax import linear_util as lu

from typing import Union, Optional, Callable, Any

import numpy as np


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
  """A wrapper around `jax.lax.scan` with in_axes/out_axes api.

  Example::
    def body_fn(b, c, x):
      return b + 2, c + 1, 2 * x

    loop = scan(body_fn, in_axes=0, out_axes=0)
    broadcast_in = 1
    carry = 2
    xs = jnp.arange(3)
    broadcast_out, carry, ys = loop(broadcast_in, carry, xs)
    print(broadcast_out)  # prints: 3
    print(carry)  # prints: 5
    print(ys)  # prints: [0, 2, 4]


  Args:
    fn: the body function of the scan loop of the form
      `(broadcast_in, carry, *args) -> (broadcast_out, carry, scan_out)`.
      the broadcast argument allows for loop independent inputs/outputs to
      be computed inside `fn`. `fn` will be called once to compute `broadcast_out`
      The actual loop will receive `broadcast_out` as the new `broadcast_in`.
      This is useful for initializing values inside the loop.
    in_axes: specifies the axis along which arguments are scanned.
      Use `broadcast` to use the same value across iterations.
    out_axes: specifies the axis along which outputs are concatenated.
      Use `broadcast` if a return value should not be concatenated and
      is independent of the loop body.
    length: number of iterations. Only needs to be specified if there
      is no scan axis from which it can be derived.
    reverse: scan in reverse order from end to start.
   Returns:
     the function that performs the scan of the form:
     (broadcast_in, carry_in, *args) -> (broadcast_out, carry_out, scan_out).
  """

  def transpose_to_front(ax, xs):
    if ax is broadcast:
      return ()
    if ax == 0:
      return xs
    def trans(x):
      perm = tuple(range(x.ndim))
      perm = (ax,) + tuple(np.delete(perm, ax))
      return jnp.transpose(x, perm)
    return jax.tree_map(trans, xs)

  def transpose_from_front(ax, xs):
    if ax is broadcast:
      return ()
    if ax == 0:
      return xs
    def trans(x):
      if ax < 0:
        pax = x.ndim - ax
      else:
        pax = ax
      assert pax < x.ndim
      perm = tuple(range(1, pax + 1)) + (0,) + tuple(range(pax + 1, x.ndim))
      return jnp.transpose(x, perm)
    return jax.tree_map(trans, xs)

  def scan_fn(broadcast_in, init, *args):
    xs = jax.tree_multimap(transpose_to_front, in_axes, args)

    def body_fn(c, xs, init_mode=False):
      # inject constants
      xs = jax.tree_multimap(lambda ax, arg, x: (arg if ax is broadcast else x),
                             in_axes, args, xs)
      broadcast_out, c, ys = fn(broadcast_in, c, *xs)
      
      if init_mode:
        ys = jax.tree_multimap(lambda ax, y: (y if ax is broadcast else ()),
                               out_axes, ys)
        return broadcast_out, ys
      else:
        ys = jax.tree_multimap(lambda ax, y: (() if ax is broadcast else y),
                               out_axes, ys)
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
    broadcast_in, constants_out = jax.tree_unflatten(out_tree(), out_flat)
    
    c, ys = lax.scan(body_fn, init, xs, length=length, reverse=reverse)
    ys = jax.tree_multimap(transpose_from_front, out_axes, ys)
    ys = jax.tree_multimap(lambda ax, const, y: (const if ax is broadcast else y),
                           out_axes, constants_out, ys)
    return broadcast_in, c, ys

  return scan_fn
