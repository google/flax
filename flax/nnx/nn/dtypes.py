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

import typing as tp
from flax.typing import Dtype
from jax import numpy as jnp

T = tp.TypeVar('T', bound=tuple)


def canonicalize_dtype(
  *args, dtype: Dtype | None = None, inexact: bool = True
) -> Dtype:
  """Canonicalize an optional dtype to the definitive dtype.

  If the ``dtype`` is None this function will infer the dtype from the
  input arguments using ``jnp.result_type``. If it is not None it will
  be returned unmodified or an exception is raised if the dtype is
  invalid.

  Example usage::

    >>> import jax.numpy as jnp
    >>> from flax.nnx.nn.dtypes import canonicalize_dtype

    >>> # Infer dtype from inputs
    >>> canonicalize_dtype(jnp.ones(3, jnp.float16))
    dtype('float16')

    >>> # Explicit dtype override
    >>> canonicalize_dtype(
    ...     jnp.ones(3), dtype=jnp.dtype('float16')
    ... )
    dtype('float16')

    >>> # Integer inputs are promoted to float32 by default
    >>> canonicalize_dtype(jnp.ones(3, jnp.int32))
    dtype('float32')

    >>> # Set inexact=False to allow integer dtypes
    >>> canonicalize_dtype(
    ...     jnp.ones(3, jnp.int32), inexact=False
    ... )
    dtype('int32')

  Args:
    *args: JAX array compatible values. None values
      are ignored.
    dtype: Optional dtype override. If specified, this dtype is
      returned directly and dtype inference from ``*args`` is
      disabled.
    inexact: When ``True``, the output dtype must be a subdtype of
      ``jnp.inexact``. Inexact dtypes are real or complex floating
      points. This is useful when you want to apply operations that
      don't work directly on integers like taking a mean for
      example.

  Returns:
    The dtype that ``*args`` should be cast to.

  Raises:
    ValueError: If ``inexact=True`` and the resolved dtype is not
      a subdtype of ``jnp.inexact`` (e.g. passing
      ``dtype=jnp.int32``).
  """
  if dtype is None:
    args_filtered = [jnp.asarray(x) for x in args if x is not None]
    dtype = jnp.result_type(*args_filtered)
    if inexact and not jnp.issubdtype(dtype, jnp.inexact):
      dtype = jnp.promote_types(jnp.float32, dtype)
  if inexact and not jnp.issubdtype(dtype, jnp.inexact):
    raise ValueError(f'Dtype must be inexact: {dtype}')
  return dtype


def promote_dtype(args: T, /, *, dtype=None, inexact=True) -> T:
  """Promotes input arguments to a specified or inferred dtype.

  All args are cast to the same dtype. See ``canonicalize_dtype`` for
  how this dtype is determined.

  The behavior of ``promote_dtype`` is mostly a convenience wrapper
  around ``canonicalize_dtype``. It automatically casts all inputs
  to the inferred dtype, allows inference to be overridden by a
  forced dtype, and has an optional check to guarantee the resulting
  dtype is inexact.

  .. note::
    Unlike the Linen version (``flax.linen.dtypes.promote_dtype``),
    this function takes a *tuple* of arguments rather than variadic
    ``*args``. Unpack the result accordingly::

      x, kernel = promote_dtype((x, kernel), dtype=dtype)

  Example usage::

    >>> import jax.numpy as jnp
    >>> from flax.nnx.nn.dtypes import promote_dtype

    >>> # Both arrays are promoted to the wider dtype
    >>> x = jnp.ones(3, jnp.float16)
    >>> w = jnp.ones(3, jnp.float32)
    >>> x_, w_ = promote_dtype((x, w))
    >>> x_.dtype, w_.dtype
    (dtype('float32'), dtype('float32'))

    >>> # None values pass through unchanged
    >>> a, b = promote_dtype((jnp.ones(2), None))
    >>> b is None
    True

    >>> # Use inexact=False to keep integer dtypes
    >>> ids = jnp.array([0, 1], jnp.int32)
    >>> (ids_,) = promote_dtype((ids,), inexact=False)
    >>> ids_.dtype
    dtype('int32')

    >>> # Force a specific dtype (e.g. downcast to float16)
    >>> x = jnp.ones(3, jnp.float32)
    >>> (x_,) = promote_dtype((x,), dtype=jnp.float16)
    >>> x_.dtype
    dtype('float16')

  Args:
    args: Tuple of JAX array compatible values. ``None`` values
      are returned as-is.
    dtype: Optional dtype override. If specified, the arguments are
      cast to the specified dtype instead and dtype inference is
      disabled.
    inexact: When ``True``, the output dtype must be a subdtype of
      ``jnp.inexact``. Inexact dtypes are real or complex floating
      points. This is useful when you want to apply operations that
      don't work directly on integers like taking a mean for
      example.

  Returns:
    A tuple of the same length as ``args``, with each non-``None``
    element cast to the resolved dtype. ``None`` elements are
    returned as ``None``.

  Raises:
    ValueError: If ``inexact=True`` and the resolved dtype is not
      a subdtype of ``jnp.inexact`` (e.g. passing
      ``dtype=jnp.int32``).
  """
  dtype = canonicalize_dtype(*args, dtype=dtype, inexact=inexact)
  arrays = tuple(jnp.asarray(x, dtype) if x is not None else None for x in args)
  return arrays  # type: ignore[return-value]
