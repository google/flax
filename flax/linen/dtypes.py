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
"""APIs for handling dtypes in Linen Modules."""

from typing import Any, Optional, List

from jax import numpy as jnp
import jax


Dtype = Any
Array = Any


def canonicalize_dtype(*args, 
                       dtype: Optional[Dtype] = None,
                       inexact: bool = True) -> Dtype:
  """Canonicalize an optinonal dtype to the definitive dtype.
  
  If the ``dtype`` is None this function will infer the dtype
  from the input arguments using ``jnp.result_type``.

  Args:
    *args: JAX array compatible values. None values
      are ignored.
    dtype: Optional dtype specified by the Module caller
    inexact: When True, the output dtype must be a subdtype
    of `jnp.inexact`.
  Returns:
    The dtype that *args should be cast to.
  """
  if dtype is None:
    args_filtered = [jnp.asarray(x) for x in args if x is not None]
    dtype = jnp.result_type(*args_filtered)
    if inexact and not jnp.issubdtype(dtype, jnp.inexact):
      dtype = jnp.promote_types(jnp.float32, dtype)
  if inexact and not jnp.issubdtype(dtype, jnp.inexact):
    raise ValueError(f'Dtype must be inexact: {dtype}')
  return dtype


def promote_dtype(*args, dtype=None, inexact=True) -> List[Array]:
  """"Promotes input arguments to the appropiate dtype.
  
  All args are cast to the same dtype according to the
  result of ``canonicalize_dtype``.

  Args:
    *args: JAX array compatible values. None values
      are returned as is.
    dtype: Optional dtype specified by the Module caller
    inexact: When True, the dtype must be a subdtype
    of `jnp.inexact`.
  Returns:
    The arguments cast to arrays of the same dtype.
  """
  args = [jnp.asarray(x) if x is not None else None for x in args]
  dtype = canonicalize_dtype(*args, dtype=dtype, inexact=inexact)
  return [jnp.asarray(x, dtype) if x is not None else None
          for x in args]
