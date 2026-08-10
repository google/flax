# Copyright 2024 The Flax Authors.
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


def _zeros_like_sds(x):
  if isinstance(x, jax.ShapeDtypeStruct):
    return jnp.zeros(x.shape, x.dtype, device=x.sharding)
  return x


def lazy_init(fn):
  """Evaluates a function with zeros arrays in place of ShapeDtypeStructs.

  The returned function accepts the same arguments as ``fn``, except any
  argument may instead be given as a ``jax.ShapeDtypeStruct`` specifying only
  its shape and dtype. Each ``ShapeDtypeStruct`` is replaced with an all-zeros
  array before evaluating ``fn`` as usual.

  This API is used by ``core.lazy_init`` or ``Module.lazy_init``
  to initialize variables without providing concrete input data.

  Note that if the result of ``fn`` depends on the values (rather than just
  the shape or dtype) of an argument given as a ``ShapeDtypeStruct``, the
  result is silently computed as if that argument were all zeros.

  Args:
    fn: the function to be evaluated.
  Returns:
    A new function that accepts a mix of concrete values and
    ``jax.ShapeDtypeStruct`` instances.
  """

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    args, kwargs = jax.tree_util.tree_map(_zeros_like_sds, (args, kwargs))
    return fn(*args, **kwargs)

  return wrapper
