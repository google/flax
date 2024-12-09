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

import contextlib
import dataclasses
import threading
import typing as tp

A = tp.TypeVar('A')
B = tp.TypeVar('B')


@dataclasses.dataclass
class ReprContext(threading.local):
  indent_stack: tp.List[str] = dataclasses.field(default_factory=lambda: [''])


REPR_CONTEXT = ReprContext()


@dataclasses.dataclass
class Object:
  type: tp.Union[str, type]
  start: str = '('
  end: str = ')'
  value_sep: str = '='
  elem_indent: str = '  '
  empty_repr: str = ''


@dataclasses.dataclass
class Attr:
  key: str
  value: tp.Union[str, tp.Any]
  start: str = ''
  end: str = ''


class Representable:
  __slots__ = ()

  def __nnx_repr__(self) -> tp.Iterator[tp.Union[Object, Attr]]:
    raise NotImplementedError

  def __repr__(self) -> str:
    return get_repr(self)


@contextlib.contextmanager
def add_indent(indent: str) -> tp.Iterator[None]:
  REPR_CONTEXT.indent_stack.append(REPR_CONTEXT.indent_stack[-1] + indent)

  try:
    yield
  finally:
    REPR_CONTEXT.indent_stack.pop()


def get_indent() -> str:
  return REPR_CONTEXT.indent_stack[-1]


def get_repr(obj: Representable) -> str:
  if not isinstance(obj, Representable):
    raise TypeError(f'Object {obj!r} is not representable')

  iterator = obj.__nnx_repr__()
  config = next(iterator)
  if not isinstance(config, Object):
    raise TypeError(f'First item must be Config, got {type(config).__name__}')

  def _repr_elem(elem: tp.Any) -> str:
    if not isinstance(elem, Attr):
      raise TypeError(f'Item must be Elem, got {type(elem).__name__}')

    value = elem.value if isinstance(elem.value, str) else repr(elem.value)

    value = value.replace('\n', '\n' + config.elem_indent)

    return f'{config.elem_indent}{elem.start}{elem.key}{config.value_sep}{value}{elem.end}'

  with add_indent(config.elem_indent):
    elems = ',\n'.join(map(_repr_elem, iterator))

  if elems:
    elems = '\n' + elems + '\n'
  else:
    elems = config.empty_repr

  type_repr = (
    config.type if isinstance(config.type, str) else config.type.__name__
  )

  return f'{type_repr}{config.start}{elems}{config.end}'

class MappingReprMixin(tp.Mapping[A, B]):
  def __nnx_repr__(self):
    yield Object(type='', value_sep=': ', start='{', end='}')

    for key, value in self.items():
      yield Attr(repr(key), value)

@dataclasses.dataclass(repr=False)
class PrettyMapping(Representable):
  mapping: tp.Mapping

  def __nnx_repr__(self):
    yield Object(type='', value_sep=': ', start='{', end='}')

    for key, value in self.mapping.items():
      yield Attr(repr(key), value)

@dataclasses.dataclass(repr=False)
class PrettySequence(Representable):
  list: tp.Sequence

  def __nnx_repr__(self):
    yield Object(type='', value_sep='', start='[', end=']')

    for value in self.list:
      yield Attr('', value)