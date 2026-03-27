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
import warnings
from typing import TypeVar
from collections.abc import Callable

F = TypeVar('F', bound=Callable)


def deprecated(new_fn: F) -> F:
  """Creates a deprecated alias for a renamed function.

  .. deprecated::
     This decorator is for marking functions as deprecated. The returned
     wrapper emits a :class:`DeprecationWarning` on every call and then
     delegates to ``new_fn``.

  The returned callable copies the signature, type annotations, and
  docstring of ``new_fn``, with a deprecation notice prepended to the
  docstring. This keeps IDE autocomplete and type-checking working while
  clearly communicating that callers should migrate.

  Args:
    new_fn: The current, non-deprecated function to delegate to.

  Returns:
    A wrapper that emits a :class:`DeprecationWarning` and then calls
    ``new_fn`` with the same arguments.

  Example::

    >>> from flax.nnx.deprecations import deprecated
    >>> def new_api(x):
    ...   return x * 2
    >>> old_api = deprecated(new_api)

  """

  @functools.wraps(new_fn)
  def wrapper(*args, **kwargs):
    warnings.warn(
      f'This function is deprecated. Use {new_fn.__qualname__} instead.',
      DeprecationWarning,
      stacklevel=2,
    )
    return new_fn(*args, **kwargs)

  dep_notice = (
    f'.. deprecated::\n'
    f'   Use :func:`{new_fn.__qualname__}` instead.\n\n'
  )
  wrapper.__doc__ = dep_notice + (new_fn.__doc__ or '')

  return wrapper  # type: ignore[return-value]
