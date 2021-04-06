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
"""Utilities we could consider upstreaming to Jax.
"""

import collections
from collections.abc import Iterable  # pylint: disable=g-importing-member
import warnings

import jax
from jax import lax
from jax import linear_util as lu
from jax.config import config
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
import jax.lib.xla_bridge as xb
import jax.numpy as jnp
import numpy as np


def _replicate(x, devices=None):
  x = jax.numpy.asarray(x)
  if devices is None:
    # match the default device assignments used in pmap:
    # for single-host, that's the XLA default device assignment
    # for multi-host, it's the order of jax.local_devices()
    if jax.host_count() == 1:
      devices = [d for d in xb.get_backend().get_default_device_assignment(
          jax.device_count()) if d.host_id == jax.host_id()]
    else:
      devices = jax.local_devices()
  if hasattr(jax.api, "device_put_sharded"):  # jax >= 0.2.0
    return jax.api.device_put_sharded(len(devices) * [x], devices)
  else:
    aval = jax.ShapedArray((len(devices),) + x.shape, x.dtype)
    buffers = [xla.device_put(x, device=d) for d in devices]
    return jax.pxla.ShardedDeviceArray(aval, buffers)


def replicate(tree, devices=None):
  """Replicates arrays to multiple devices.

  Args:
    tree: a pytree containing the arrays that should be replicated.
    devices: the devices the data is replicated to
      (default: `jax.local_devices()`).
  Returns:
    A new pytree containing the replicated arrays.
  """
  return jax.tree_map(lambda x: _replicate(x, devices), tree)


def unreplicate(tree):
  """Returns a single instance of a replicated array."""
  return jax.tree_map(lambda x: x[0], tree)


def pmean(xs, axis_name):
  warnings.warn('use jax.lax.pmean instead',
                DeprecationWarning)
  return lax.pmean(xs, axis_name)


def partial_eval_by_shape(fn, input_spec, *args, **kwargs):
  """Lazily evaluate a function by using the shapes of the inputs.

  This function is similar to `jax.eval_shape` with the key difference that
  function outputs that can be computed without a concrete value of the
  inputs are returned as is instead of only the shape. See for example
  `module.init_by_shape` where this functionality is used to initialize a
  model without using input data lr computation.

  Args:
    fn: the function to be lazily evaluated.
    input_spec: an iterable of shapes or (shape, dtype) tuples specifying the
      shape and type of the inputs. If unspecified the dtype is float32.
    *args: other arguments passed to the module's apply function
    **kwargs: keyword arguments passed to the module's apply function
  Returns:
    A pair consisting of the model output and an instance of Model
  """
  # output cannot be returned in lazy_create because jax.eval_shape will only
  # return the shape and dtype.
  # TODO(mattjj,jheek): use a public JAX API
  f = lambda *inputs: fn(*inputs, *args, **kwargs)
  input_structs = [_parse_spec(spec) for spec in input_spec]
  inputs_flat, in_tree = jax.tree_flatten(input_structs)
  f_flat, out_tree = jax.api_util.flatten_fun_nokwargs(lu.wrap_init(f), in_tree)
  in_pvals = [pe.PartialVal.unknown(jax.ShapedArray(x.shape, x.dtype))
              for x in inputs_flat]
  _, out_pvals, _ = pe.trace_to_jaxpr(f_flat, in_pvals)
  out_flat = [const if pv is None else jax.ShapeDtypeStruct(pv.shape, pv.dtype)
              for pv, const in out_pvals]
  return jax.tree_unflatten(out_tree(), out_flat)


def _parse_spec(spec):
  """Parse an input spec of the form (shape, dtype) or shape into a jax.ShapeDtypeStruct."""
  spec = tuple(spec)
  if len(spec) == 2 and isinstance(spec[0], Iterable):
    return jax.ShapeDtypeStruct(tuple(spec[0]), spec[1])
  else:
    return jax.ShapeDtypeStruct(spec, jnp.float32)


def prefetch_to_device(iterator, size, devices=None):
  """"Shard and prefetch batches on device.

  This utility takes an iterator and returns a new iterator which fills an on
  device prefetch buffer. Eager prefetching can improve the performance of
  training loops significantly by overlapping compute and data transfer.
  
  This utility is mostly useful for GPUs, for TPUs it should not be necessary.

  Args:
    iterator: an iterator that yields a pytree of ndarrays where the first
      dimension is sharded across devices.
    size: the size of the prefetch buffer.
    devices: the list of devices to which the arrays should be prefetched.
  Yields:
    The original items from the iterator where each ndarray is now a sharded to
    the specified devices.
  """
  queue = collections.deque()
  if devices is None:
    devices = jax.local_devices()
  def _prefetch(xs):
    if hasattr(jax.api, "device_put_sharded"):  # jax>=0.2.0
      return jax.api.device_put_sharded(list(xs), devices)
    else:
      aval = jax.xla.abstractify(xs)
      assert xs.shape[0] == len(devices), (
          "The first dimension of the iterator's ndarrays is not "
          "equal to the number of devices.")
      buffers = [xla.device_put(x, devices[i])
                 for i, x in enumerate(xs)]
      return jax.pxla.ShardedDeviceArray(aval, buffers)
  try:
    while len(queue) < size:
      queue.append(jax.tree_map(_prefetch, next(iterator)))
  except StopIteration:
    pass

  while True:
    try:
      xs = queue.popleft()
    except IndexError:
      return
    try:
      queue.append(jax.tree_map(_prefetch, next(iterator)))
    except StopIteration:
      pass
    yield xs


def _scan_nd(body_fn, init, xs, n=1):
  """Utility for performing an n-dimensional `lax.scan`.

  The n-d scan is simply recursive call of 1-d scan.
  Args:
    body_fn: the body of the loop of type (c, x) -> (c, y).
    init: initial value for the carry.
    xs: a pytree of tensors to scan over.
    n: number of dimensions to scan over (default: 1)
  Returns:
    A tuple of the final carry and the values returned by the body.
  """
  if n == 1:
    return lax.scan(body_fn, init, xs)
  else:
    def scan_body(c, x):
      return _scan_nd(body_fn, c, x, n=n-1)
    return lax.scan(scan_body, init, xs)


def _invert_perm(perm):
  perm_inv = [0] * len(perm)
  for i, j in enumerate(perm):
    perm_inv[j] = i
  return tuple(perm_inv)


def scan_in_dim(body_fn, init, xs, axis=(0,), keepdims=False):
  """utility for doing a scan along arbitrary dimensions.

  see `lax.scan` for details on how the scan operation works.
  Args:
    body_fn: the body of the loop of type (c, x) -> (c, y).
    init: initial value for the carry.
    xs: a pytree of tensors to scan over.
    axis: the axis to scan over.
    keepdims: keep the dimensions that are scanned over.
  Returns:
    A tuple of the final carry and the values returned by the body.
  """
  if not isinstance(axis, Iterable):
    axis = (axis,)

  def transpose_in(x):
    perm = axis + tuple(np.delete(np.arange(x.ndim), axis))
    return x.transpose(perm)
  def transpose_out(x):
    perm = axis + tuple(np.delete(np.arange(x.ndim), axis))
    return x.transpose(_invert_perm(perm))

  def body_wrapper(c, xs):
    if keepdims:
      xs = jax.tree_map(lambda x: x.reshape((1,) * len(axis) + x.shape), xs)
      xs = jax.tree_map(transpose_out, xs)
    c, ys = body_fn(c, xs)
    if keepdims:
      ys = jax.tree_map(transpose_in, ys)
      ys = jax.tree_map(lambda x: x.reshape(x.shape[len(axis):]), ys)
    return c, ys

  xs = jax.tree_map(transpose_in, xs)
  c, ys = _scan_nd(body_wrapper, init, xs, n=len(axis))
  ys = jax.tree_map(transpose_out, ys)
  return c, ys
