# Lint as: python3

# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Internal utilities for using Jax.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax


def _replicate(x, devices=None):
  x = jax.numpy.array(x)
  if devices is None:
    devices = jax.local_devices()
  aval = jax.ShapedArray((len(devices),) + x.shape, x.dtype)
  buffers = [jax.interpreters.xla.device_put(x, device=d) for d in devices]
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


def partial_eval_by_shape(fn, input_spec, *args, **kwargs):
  """Lazily evaluate a function by using the shapes of the inputs.

  This function is similar to `jax.eval_shape` with the key difference that
  function outputs that can be computed without a concrete value of the
  inputs are returned as is instead of only the shape. See for example
  `module.create_by_shape` where this functionality is used to initialize a
  model without using input data lr computation.

  Args:
    fn: the function to be lazily evaluated.
    input_spec: an iterable of (shape, dtype) pairs specifying the inputs
    *args: other arguments passed to the module's apply function
    **kwargs: keyword arguments passed to the module's apply function
  Returns:
    A pair consisting of the model output and an instance of Model
  """
  # output cannot be returned in lazy_create because jax.eval_shape will only
  # return the shape and dtype.
  output_traced = None
  master = None
  def lazy_fn(*inputs):  # pylint: disable=missing-docstring
    nonlocal output_traced, master
    leaves = jax.tree_leaves(inputs)
    if leaves:
      # TODO(akolesnikov): revert this check after 10.02.20 (ICML deadline)
      if hasattr(leaves[0], '_trace'):
        master = leaves[0]._trace.master  # pylint: disable=protected-access
      else:
        master = leaves[0].trace.master  # pylint: disable=protected-access
    output = fn(*(inputs + args), **kwargs)
    output_traced = output
    return output

  input_structs = [jax.ShapeDtypeStruct(shape, dtype)
                   for shape, dtype in input_spec]
  output_shapes = jax.eval_shape(lazy_fn, *input_structs)
  def merge_results(traced, shape):  # pylint: disable=missing-docstring
    # Only return the shape when an output depends on any unknown inputs.
    # pylint: disable=protected-access
    if isinstance(traced, jax.core.Tracer):
      # TODO(akolesnikov): revert this check after 10.02.20 (ICML deadline)
      if hasattr(traced, '_trace'):
        traced_master = traced._trace.master  # pylint: disable=protected-access
      else:
        traced_master = traced.trace.master  # pylint: disable=protected-access
      if traced_master == master:
        return shape
    return traced
    # pylint: enable=protected-access
  return jax.tree_multimap(merge_results, output_traced, output_shapes)



