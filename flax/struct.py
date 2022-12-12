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

"""Utilities for defining custom classes that can be used with jax transformations.
"""

import dataclasses
import typing
from typing import TypeVar, Callable, Tuple, Union, Any

from . import serialization

import jax
from typing_extensions import dataclass_transform  # pytype: disable=not-supported-yet


_T = TypeVar("_T")


def field(pytree_node=True, **kwargs):
  return dataclasses.field(metadata={'pytree_node': pytree_node}, **kwargs)


@dataclass_transform(field_descriptors=(field,))
def dataclass(clz: _T) -> _T:
  """Create a class which can be passed to functional transformations.

  NOTE: Inherit from ``PyTreeNode`` instead to avoid type checking issues when
  using PyType.

  Jax transformations such as `jax.jit` and `jax.grad` require objects that are
  immutable and can be mapped over using the `jax.tree_util` methods.
  The `dataclass` decorator makes it easy to define custom classes that can be
  passed safely to Jax. For example::

    from flax import struct

    @struct.dataclass
    class Model:
      params: Any
      # use pytree_node=False to indicate an attribute should not be touched
      # by Jax transformations.
      apply_fn: FunctionType = struct.field(pytree_node=False)

      def __apply__(self, *args):
        return self.apply_fn(*args)

    model = Model(params, apply_fn)

    model.params = params_b  # Model is immutable. This will raise an error.
    model_b = model.replace(params=params_b)  # Use the replace method instead.

    # This class can now be used safely in Jax to compute gradients w.r.t. the
    # parameters.
    model = Model(params, apply_fn)
    model_grad = jax.grad(some_loss_fn)(model)

  Note that dataclasses have an auto-generated ``__init__`` where
  the arguments of the constructor and the attributes of the created
  instance match 1:1. This correspondence is what makes these objects
  valid containers that work with JAX transformations and
  more generally the `jax.tree_util` library.

  Sometimes a "smart constructor" is desired, for example because
  some of the attributes can be (optionally) derived from others.
  The way to do this with Flax dataclasses is to make a static or
  class method that provides the smart constructor.
  This way the simple constructor used by `jax.tree_util` is
  preserved. Consider the following example::

    @struct.dataclass
    class DirectionAndScaleKernel:
      direction: Array
      scale: Array

      @classmethod
      def create(cls, kernel):
        scale = jax.numpy.linalg.norm(kernel, axis=0, keepdims=True)
        directin = direction / scale
        return cls(direction, scale)

  Args:
    clz: the class that will be transformed by the decorator.
  Returns:
    The new class.
  """
  # check if already a flax dataclass
  if '_flax_dataclass' in clz.__dict__:
    return clz

  data_clz = dataclasses.dataclass(frozen=True)(clz) # type: ignore
  meta_fields = []
  data_fields = []
  for field_info in dataclasses.fields(data_clz):
    is_pytree_node = field_info.metadata.get('pytree_node', True)
    if is_pytree_node:
      data_fields.append(field_info.name)
    else:
      meta_fields.append(field_info.name)

  def replace(self, **updates):
    """"Returns a new object replacing the specified fields with new values."""
    return dataclasses.replace(self, **updates)

  data_clz.replace = replace

  def iterate_clz(x):
    meta = tuple(getattr(x, name) for name in meta_fields)
    data = tuple(getattr(x, name) for name in data_fields)
    return data, meta

  def clz_from_iterable(meta, data):
    meta_args = tuple(zip(meta_fields, meta))
    data_args = tuple(zip(data_fields, data))
    kwargs = dict(meta_args + data_args)
    return data_clz(**kwargs)

  jax.tree_util.register_pytree_node(data_clz,
                                     iterate_clz,
                                     clz_from_iterable)

  if tuple(map(int, jax.version.__version__.split('.'))) >= (0, 3, 1):
    def keypaths(_):
      return [jax.tree_util.AttributeKeyPathEntry(name) for name in data_fields]
    jax.tree_util.register_keypaths(data_clz, keypaths)

  def to_state_dict(x):
    state_dict = {name: serialization.to_state_dict(getattr(x, name))
                  for name in data_fields}
    return state_dict

  def from_state_dict(x, state):
    """Restore the state of a data class."""
    state = state.copy()  # copy the state so we can pop the restored fields.
    updates = {}
    for name in data_fields:
      if name not in state:
        raise ValueError(f'Missing field {name} in state dict while restoring'
                         f' an instance of {clz.__name__},'
                         f' at path {serialization.current_path()}')
      value = getattr(x, name)
      value_state = state.pop(name)
      updates[name] = serialization.from_state_dict(value, value_state, name=name)
    if state:
      names = ','.join(state.keys())
      raise ValueError(f'Unknown field(s) "{names}" in state dict while'
                       f' restoring an instance of {clz.__name__}'
                       f' at path {serialization.current_path()}')
    return x.replace(**updates)

  serialization.register_serialization_state(
      data_clz, to_state_dict, from_state_dict)

  # add a _flax_dataclass flag to distinguish from regular dataclasses
  data_clz._flax_dataclass = True # type: ignore[attr-defined]

  return data_clz # type: ignore


TNode = TypeVar('TNode', bound='PyTreeNode')


@dataclass_transform(field_descriptors=(field,))
class PyTreeNode:
  """Base class for dataclasses that should act like a JAX pytree node.

  See ``flax.struct.dataclass`` for the ``jax.tree_util`` behavior.
  This base class additionally avoids type checking errors when using PyType.

  Example::

    from flax import struct

    class Model(struct.PyTreeNode):
      params: Any
      # use pytree_node=False to indicate an attribute should not be touched
      # by Jax transformations.
      apply_fn: FunctionType = struct.field(pytree_node=False)

      def __apply__(self, *args):
        return self.apply_fn(*args)

    model = Model(params, apply_fn)

    model.params = params_b  # Model is immutable. This will raise an error.
    model_b = model.replace(params=params_b)  # Use the replace method instead.

    # This class can now be used safely in Jax to compute gradients w.r.t. the
    # parameters.
    model = Model(params, apply_fn)
    model_grad = jax.grad(some_loss_fn)(model)

  """

  def __init_subclass__(cls):
    dataclass(cls)  # pytype: disable=wrong-arg-types

  def __init__(self, *args, **kwargs):
    # stub for pytype
    raise NotImplementedError

  def replace(self: TNode, **overrides) -> TNode:
    # stub for pytype
    raise NotImplementedError
