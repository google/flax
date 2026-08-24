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

"""A library to conveniently specify shardable parameters in Flax.

While linen provides a functional API to specifying parameters, it involves
non-trivial boilerplate if you need to specify all of: (1) the initializer,
(2) the shape, and (3) the partitioning / sharding information.

The Param library radically reduces the amount of boilerplate involved.
The library is centered around (1) the Param type, and (2) the
`param_schema` function used to declare parameters.

To use the Param library, simply declare a `nn.Module`'s fields using
`param_schema`, and then use the initialized Param's to create trainable
variables. See the following for an example::

  class SimpleLinear(nn.Module):
    features: int
    kernel: Param = param_schema(features=Attr, input_features=Given)

    @nn.compact
    def __call__(self, x):
      kernel = self.kernel(input_features=x.shape[-1])
      return kernel @ x

In the above example, we declare `kernel` to be a trainable variable of shape:
`[features, input_features]`, where `features` is inferred from `self.features`,
and `input_features` is supplied on the first line of `__call__` based on the
innermost dimension of `x`. The default kernel initializer is used.

To specify an alternate initializer, pass it as the first (positional-only)
argument to `param_schema`::

  bias: Param = param_schema(zeros_like, bias_dim=1)

You can also provide a "template" Param for the first argument to configure
the default ``dtype``::

  quantized_bias: Param = param_schema(
      Param(dtype=jnp.int8, initializer=zeros_like),
      bias_dim=1)

Finally, you can pass a ``dataclasses.Field`` instance (optionally with
``default`` set a template instance) to configure dataclass parameters such as
``kw_only``::

  kw_only_param = param_schema(
      dataclasses.field(default=Param(dtype=jnp.bfloat16), kw_only=True),
      features=Attr,
      input_features=Given)

The ``jnp.array`` returned from calling the ``Param`` has already been
sharded according to the user-supplied configuration. If no configuration has
been provided, the parameter will not be sharded. For tools to conveniently
manage model shardings, please see the companion library: spec_shardings.
"""

import abc
from collections.abc import Collection, Sequence
import copy
import dataclasses
import functools
import types
from typing import Any, Callable, NamedTuple, Optional, Type, Union

from flax import linen as nn
from flax.linen import initializers
import jax
from jax import numpy as jnp

Shape = Sequence[int]  # TODO(saeta): Expose this type from jax._src.typing!
DTypeLike = Any
ParamInitializer = Callable[[jax.random.KeyArray, Shape, DTypeLike], jnp.array]


_DEFAULT_INITIALIZER = initializers.lecun_normal()


class AxisInit(abc.ABC):
  pass


@dataclasses.dataclass(frozen=True)
class Attr(AxisInit):
  """Indicates an axis' size is specified by an attribute on the module.

  By default, the name is inferred from the axis name, but it is possible to
  specify an alternative name.

  Examples::

    class MyModule(ps.Module):
      num_features: int
      other_value: int
      kernel: Param = ps.param_schema(
          num_features=ps.Attr,  # Pulls from MyModule.num_features
          model_axis=ps.Attr('other_value'),  # Pulls from MyModule.other_value
      )
  """

  from_name: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class Given(AxisInit):
  """Indicates an axis' size is given as an argument to Param.__call__.

  TODO(saeta): FILL ME IN!
  """

  pass


@dataclasses.dataclass(frozen=True)
class Const(AxisInit):
  """When an axis always has a constant size.

  An example for this is: if you're combining the q, k, and v projection weights
  into a single array for performance, you will always have an axis with size 3.
  """

  value: int


Schema = dict[str, AxisInit]


class Param(nn.Module):
  """Encapsulates information to create a sharded learnable parameter."""

  initializer: ParamInitializer = _DEFAULT_INITIALIZER
  sharding: Optional[dict[str, str]] = None
  dtype: jnp.dtype = jnp.float32

  @property
  def schema(self) -> Schema:
    if self.name is None:
      raise RuntimeError(
          'Schema not valid yet. Assign to an appropriate module instance.'
      )
    if not dataclasses.is_dataclass(self.parent):
      raise RuntimeError(f'Parent is not a dataclass. Got: {self.parent}')
    parent_fields = dataclasses.fields(self.parent)
    field = next(f for f in parent_fields if f.name == self.name)
    schema = get_param_schema(field)
    if schema is None:
      raise TypeError(
          'Schema not available on field; did you declare field'
          f' {self.name} with `param_schema`?'
      )
    return schema

  @nn.compact
  def __call__(self, **kwargs) -> jnp.array:
    shape = self.shape(**kwargs)
    # TODO(saeta): Don't use partial because can't assume keywords match??
    init_fn = functools.partial(self.initializer, shape=shape, dtype=self.dtype)

    # TODO(saeta): Wrap `init_fn` to make it return a Box with suitable metadata
    array = self.param('w', init_fn)
    # TODO(saeta): Apply sharding here to array.
    return array

  @nn.nowrap
  def shape(self, **kwargs) -> Shape:
    """Computes the shape given a set of `kwargs`."""
    self._validate()
    if diff := _sequence_diff(
        expected=set(self.variable_axis_names), actual=set(kwargs.keys())
    ):
      raise ValueError(
          'Wrong arguments specified when creating parameter '
          f'{self.name} —- extra: {diff.extra!r}, '
          f'missing: {diff.missing!r}'
      )
    for k, v in kwargs.items():
      if not isinstance(v, int):
        raise ValueError(
            f'The shape for axis {k!r} (for param '
            f'{self.name!r}) specified as {type(v)}; '
            'expected int.'
        )

    shape = []
    for name, value in self.schema.items():
      if isinstance(value, Given):
        shape.append(kwargs[name])
      elif isinstance(value, Const):
        shape.append(value.value)
      else:
        assert isinstance(value, Attr)
        attribute_name = value.from_name or name
        shape.append(getattr(self.parent, attribute_name))
    return tuple(shape)

  @nn.nowrap
  def sharding_spec(self):
    raise NotImplementedError('sharding spec not yet computed')

  @property
  def variable_axis_names(self) -> Sequence[str]:
    return [k for k, v in self.schema.items() if isinstance(v, Given)]

  @nn.nowrap
  def _validate(self):
    """Checks for any violated invariants once the parent has been specified."""
    _ = self.schema  # Ensures the schema is accessible.
    if self.sharding:
      # If sharding is specified, all axes must be specified.
      if diff := _sequence_diff(
          expected=set(self.schema.keys()), actual=set(self.sharding.keys())
      ):
        raise ValueError(
            f'Incorrect sharding specified for {self.name} '
            f'-- extra: {diff.extra!r}, missing: {diff.missing!r}'
        )


_DATACLASS_METADATA_KEY = object()


def get_param_schema(field: dataclasses.Field[Param]) -> Optional[Schema]:
  return field.metadata.get(_DATACLASS_METADATA_KEY, None)


def param_schema(
    spec_or_field_or_init: Optional[
        Union[Param, dataclasses.Field[Param], ParamInitializer]
    ] = None,
    /,
    **kwargs: Union[AxisInit, Type[AxisInit], int],
) -> Any:
  """Specifies the schema for a `Param`.

  Important Note: the order of the kwargs is significant! Each kwarg names an
  axis of the resulting parameter (in order), and declares how its size will be
  determined.

  Args:
    spec_or_field_or_init: A way to specify either (1) a custom initializer
      (e.g. ``zeros_like``), (2) extra Param information (e.g. default
      ``dtype``), and/or (3) `dataclass.field` parameters, such as
      ``kw_only=True``.
    **kwargs: The schema in ordered, `name: AxisInit` pairs.

  Returns:
    A `dataclasses.Field` instance with extra metadata information included.
    Note: the type signature says `Any` because that is required to make type-
    checking work.
  """
  if spec_or_field_or_init is None:
    field = dataclasses.field()
    template = Param()
  elif isinstance(spec_or_field_or_init, dataclasses.Field):
    field = spec_or_field_or_init
    if field.default_factory is not dataclasses.MISSING:
      raise TypeError('You cannot specify a `default_factory`.')
    template = (
        field.default
        if field.default is not dataclasses.MISSING
        else Param()
    )
    if not isinstance(template, Param):
      raise ValueError(
          f'Unexpected default value: {template}; expected a '
          f'Param, got a {type(template)}.'
      )
  elif isinstance(spec_or_field_or_init, Param):
    field = dataclasses.field()
    template = spec_or_field_or_init
  elif callable(spec_or_field_or_init):
    # TODO(saeta): Validate callable here?
    field = dataclasses.field()
    template = Param(initializer=spec_or_field_or_init)
  else:
    raise TypeError(
        f'Unexpected value {spec_or_field_or_init}, expected Param.'
    )

  for k in kwargs:
    v = kwargs[k]
    if v in (Attr, Given):
      v = v()  # Create an instance.
      kwargs[k] = v
    if isinstance(v, int):
      v = Const(v)
      kwargs[k] = v
    if not isinstance(v, AxisInit):
      raise TypeError(
          f'Unexpected value {v} (type: {type(v)}, name: {k!r}), '
          'expected AxisInit.'
      )
    if isinstance(v, Const):
      if v.value <= 0:
        raise ValueError(f'The size of axis {k!r} must be > 0; got: {v.value}.')

  metadata = types.MappingProxyType(
      field.metadata if field.metadata is not dataclasses.MISSING else {}
  )
  metadata = {
      **metadata,
      _DATACLASS_METADATA_KEY: types.MappingProxyType(kwargs),
  }
  # Is there a more elegant way to do this? Use dir?
  return dataclasses.field(  # pytype: disable=not-supported-yet
      default_factory=lambda: copy.copy(template),
      init=field.init,
      repr=field.repr,
      hash=field.hash,
      compare=field.compare,
      metadata=metadata,
      kw_only=field.kw_only,
  )


class SequenceDifferences(NamedTuple):
  extra: Collection[str]
  missing: Collection[str]


def _sequence_diff(
    expected: set[str], actual: set[str]
) -> Optional[SequenceDifferences]:
  if expected == actual:
    return None
  missing = expected - actual
  extra = actual - expected
  return SequenceDifferences(
      missing=tuple(sorted(missing)), extra=tuple(sorted(extra))
  )
