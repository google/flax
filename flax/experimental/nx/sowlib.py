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

from __future__ import annotations
import contextlib
import dataclasses
import threading
import typing as tp

from jax.experimental.attrs import jax_setattr, jax_getattr

A = tp.TypeVar('A')
B = tp.TypeVar('B')


@dataclasses.dataclass
class Context(threading.local):
  sow_stack: list[SowContext] = dataclasses.field(default_factory=list)


CONTEXT = Context()


class SowContext:
  if tp.TYPE_CHECKING:

    def __getattr__(self, name) -> tp.Any:
      pass


@contextlib.contextmanager
def sow_context(
  *collections: str,
):
  _sow_context = SowContext()
  for collection in collections:
    setattr(_sow_context, collection, SowContext())

  CONTEXT.sow_stack.append(_sow_context)

  try:
    yield _sow_context
  finally:
    CONTEXT.sow_stack.pop()


tuple_reduce = lambda xs, x: xs + (x,)
tuple_init = lambda: ()


def sow(
  collection: str,
  name: str,
  value: A,
  *,
  reduce_fn: tp.Callable[[B, A], B] = tuple_reduce,
  init_fn: tp.Callable[[], B] = tuple_init,  # type: ignore
) -> None:
  if not CONTEXT.sow_stack:
    raise RuntimeError(
      'sow() called outside of a `sow_context()` context manager.'
    )

  sow_context = CONTEXT.sow_stack[-1]
  if not hasattr(sow_context, collection):
    raise ValueError(f'Collection {collection} not found in sow_context.')
  collection_context = getattr(sow_context, collection)
  if not isinstance(collection_context, SowContext):
    raise ValueError(f'Collection {collection} is not a SowContext.')

  if hasattr(collection_context, name):
    stored_value: B = jax_getattr(collection_context, name)
    stored_value = reduce_fn(stored_value, value)
  else:
    stored_value = reduce_fn(init_fn(), value)

  jax_setattr(collection_context, name, stored_value)
