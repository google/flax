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

import dataclasses
import typing as tp
import typing_extensions as tpe

from flax import struct
from flax.nnx import object as objectlib

A = tp.TypeVar('A')
T = tp.TypeVar('T', bound=type[objectlib.Object])


class StaticTag:
  ...


Static = tp.Annotated[A, StaticTag]  # type: ignore[invalid-typevar]


def _is_static(annotation: type, cls_attr: tp.Any) -> bool:
  return (
    annotation != tp.ClassVar
    and getattr(annotation, '__metadata__', None) == (StaticTag,)
  ) or (
    isinstance(cls_attr, dataclasses.Field)
    and (
      cls_attr.metadata.get('static', False)
      or not cls_attr.metadata.get('pytree_node', True)
    )
  )


# def field(*, default=MISSING, default_factory=MISSING, init=True, repr=True,
#           hash=None, compare=True, metadata=None, kw_only=MISSING):
MISSING = dataclasses.MISSING


@tp.overload # type: ignore[misc]
def field(  # type: ignore[misc]
  *,
  default: tp.Any = MISSING,
  default_factory: tp.Callable[[], A] | tp.Any = MISSING,
  init: bool = True,
  repr: bool = True,
  hash: bool | None = None,
  compare: bool = True,
  metadata: tp.Mapping[str, tp.Any] | None = None,
  kw_only: bool = False,
  static: bool = False,
) -> tp.Any: ...


def field(
  *,
  static: bool = False,
  **kwargs,
):
  metadata = kwargs.pop('metadata', None)
  metadata = dict(metadata) if metadata else {}
  if 'static' in metadata and metadata['static'] != static:
    raise ValueError(
      f'Inconsistent static metadata, field specified {static=} '
      f'but also got {metadata["static"]=}'
    )
  else:
    metadata['static'] = static
  kwargs['metadata'] = metadata
  return dataclasses.field(**kwargs)  # type: ignore[wrong-arg-type]


@tp.overload # type: ignore[misc]
def static(  # type: ignore[misc]
  *,
  default: tp.Any = MISSING,
  default_factory: tp.Callable[[], A] | tp.Any = MISSING,
  init: bool = True,
  repr: bool = True,
  hash: bool | None = None,
  compare: bool = True,
  metadata: tp.Mapping[str, tp.Any] | None = None,
  kw_only: bool = False,
) -> tp.Any: ...


def static(**kwargs):
  return field(
    static=True,
    **kwargs,
  )


@tp.overload
def dataclass(
    cls: T,
    /,
    *,
    init: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
) -> T:
  ...


@tp.overload
def dataclass(
    *,
    init: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
) -> tp.Callable[[T], T]:
  ...


@tpe.dataclass_transform(  # type: ignore[not-supported-yet]
  field_specifiers=(field, static, dataclasses.field, struct.field),
)
def dataclass(
    cls: T | None = None,
    /,
    *,
    init: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    match_args: bool = True,
    kw_only: bool = False,
    slots: bool = False,
) -> T | tp.Callable[[T], T]:
  """Makes an nnx.Object type as a dataclass and defines its pytree node attributes using type hints.

  ``nnx.dataclass`` can be used to create pytree dataclass types using type
  hints instead of the ``__data__`` attribute. By default, all fields are
  considered to be nodes, to mark a field as static annotate it with
  ``nnx.Static[T]``.

  Example::

    from flax import nnx
    import jax

    @nnx.dataclass
    class Foo(nnx.Object):
      a: int
      b: jax.Array
      c: nnx.Static[int]
        tree = Foo(a=1, b=jax.numpy.array(1), c=1)

    assert len(jax.tree.leaves(tree)) == 2 # a and b

  ``dataclass`` will raise a ``ValueError`` if the class does not derive from
  ``nnx.Object``, if the parent Object has ``pytree`` set to anything other than
  ``'strict'``, or if the class has a ``__data__`` attribute.

  ``nnx.dataclass`` doesn't accept ``repr`` and defines it as ``False`` to avoid
  overwriting the default ``__repr__`` method from ``Object``.
  """

  def _dataclass(cls: T):
    if not issubclass(cls, objectlib.Object):
      raise ValueError(
          'dataclass can only be used with a class derived from nnx.Object'
      )
    if cls._object__nodes in ('auto', 'all'):
      raise ValueError(
        "dataclass cannot be used with a class that has __data__ set to 'auto' or 'all', "
        f'got {cls._object__nodes}'
      )

    # here we redefine _object__nodes using the type hints
    hints = cls.__annotations__
    if cls._object__nodes is None:
      all_nodes = set()
    else:
      all_nodes = set(cls._object__nodes)
    for name, typ in hints.items():
      class_attr = getattr(cls, name, None)
      if not _is_static(typ, class_attr):
        all_nodes.add(name)
    cls._object__nodes = frozenset(all_nodes)

    cls = dataclasses.dataclass(  # type: ignore
      cls,
      init=init,
      repr=False,
      eq=eq,
      order=order,
      unsafe_hash=unsafe_hash,
      match_args=match_args,
      kw_only=kw_only,
      slots=slots,
    )
    return cls

  if cls is None:
    return _dataclass
  else:
    return _dataclass(cls)
