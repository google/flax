# Copyright 2023 The Flax Authors.
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

import dataclasses
import typing as tp

import typing_extensions as tpe

from flax.experimental import nnx
from flax.experimental.nnx.nnx import pytreelib, variables

A = tp.TypeVar('A')


def field(
  *,
  default: tp.Any = dataclasses.MISSING,
  default_factory: tp.Any = dataclasses.MISSING,
  init: bool = True,
  repr: bool = True,
  hash: tp.Optional[bool] = None,
  compare: bool = True,
  metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
):
  return dataclasses.field(  # type: ignore
    default=default,
    default_factory=default_factory,
    init=init,
    repr=repr,
    hash=hash,
    compare=compare,
    metadata=metadata,
  )


def treenode_field(
  *,
  default: tp.Any = dataclasses.MISSING,
  default_factory: tp.Any = dataclasses.MISSING,
  init: bool = True,
  repr: bool = True,
  hash: tp.Optional[bool] = None,
  compare: bool = True,
  metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
):
  if metadata is None:
    metadata = {}
  else:
    metadata = dict(metadata)

  if 'nnx_variable_constructor' in metadata:
    raise ValueError("'nnx_variable_constructor' found in metadata")

  metadata['nnx_variable_constructor'] = lambda value: pytreelib.TreeNode(value)

  return field(
    default=default,
    default_factory=default_factory,
    init=init,
    repr=repr,
    hash=hash,
    compare=compare,
    metadata=metadata,
  )


def variable_field(
  variable_type: tp.Type[variables.Variable[tp.Any]],
  *,
  default: tp.Any = dataclasses.MISSING,
  default_factory: tp.Any = dataclasses.MISSING,
  init: bool = True,
  repr: bool = True,
  hash: tp.Optional[bool] = None,
  compare: bool = True,
  metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
) -> tp.Any:
  if metadata is None:
    metadata = {}
  else:
    metadata = dict(metadata)

  if 'nnx_variable_constructor' in metadata:
    raise ValueError("'nnx_variable_constructor' found in metadata")

  metadata['nnx_variable_constructor'] = lambda value: variable_type(value)

  return field(
    default=default,
    default_factory=default_factory,
    init=init,
    repr=repr,
    hash=hash,
    compare=compare,
    metadata=metadata,
  )


def param_field(
  default: tp.Any = dataclasses.MISSING,
  *,
  default_factory: tp.Any = dataclasses.MISSING,
  init: bool = True,
  repr: bool = True,
  hash: tp.Optional[bool] = None,
  compare: bool = True,
  metadata: tp.Optional[tp.Mapping[str, tp.Any]] = None,
) -> tp.Any:
  return variable_field(
    variables.Param,
    default=default,
    default_factory=default_factory,
    init=init,
    repr=repr,
    hash=hash,
    compare=compare,
    metadata=metadata,
  )


@tp.overload
def dataclass(cls: tp.Type[A]) -> tp.Type[A]:
  ...


@tp.overload
def dataclass(
  *,
  init: bool = True,
  repr: bool = True,
  eq: bool = True,
  order: bool = False,
  unsafe_hash: bool = False,
  frozen: bool = False,
) -> tp.Callable[[tp.Type[A]], tp.Type[A]]:
  ...


@tpe.dataclass_transform(
  field_specifiers=(
    field,
    treenode_field,
    variable_field,
    param_field,
  )
)
def dataclass(
  cls: tp.Optional[tp.Type[A]] = None,
  init: bool = True,
  repr: bool = True,
  eq: bool = True,
  order: bool = False,
  unsafe_hash: bool = False,
  frozen: bool = False,
) -> tp.Union[tp.Type[A], tp.Callable[[tp.Type[A]], tp.Type[A]]]:
  def decorator(cls: tp.Type[A]) -> tp.Type[A]:
    cls = dataclasses.dataclass(
      init=init,
      repr=repr,
      eq=eq,
      order=order,
      unsafe_hash=unsafe_hash,
      frozen=frozen,
    )(cls)
    if issubclass(cls, nnx.Module):

      def hash_fn(module: nnx.Module):
        return hash(module._module__state.id)

      cls.__hash__ = hash_fn

    return cls

  if cls is None:
    return decorator

  return decorator(cls)
