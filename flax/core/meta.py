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

"""Boxed Metadata API

Boxed metadata enables tracking arbitrary metadata for linen variables
that is compatible with lifted transformations.

See ``Partitioned`` for a practical example on how to use this metadata
to keep track of how variables should be partitioned with ``jax.pjit``.
"""

import abc
import functools
from typing import Any, Callable, Dict, Mapping, Tuple, TypeVar, Union

from flax import errors
from flax import struct
import jax
from jax.experimental import maps
from jax.experimental import pjit


TAxisMetadata = Any # TypeVar('TAxisMetadata', bound='AxisMetadata')


class AxisMetadata(metaclass=abc.ABCMeta):
  """Abstract base class for boxed Metadata.

  ``AxisMetadata`` enables arbitrary, per axis metadata for variables.
  By using ``unbox`` the metadata is stripped away to obtain the original
  variables. By using unboxing, most code handling variables does not need
  to handle ``AxisMetadata`` specifically, but can directly operate on the JAX
  arrays that they wrap.

  Additionally, ``AxisMetadata`` supports updating metadata whenever an axis
  is added or removed by a functional transformation
  (e.g.: ``nn.scan`` or ``nn.vmap``) using the ``add_axis`` and ``remove_axis``
  methods.

  By extending ``AxisMetadata``, custom metadata can be stored. See
  ``Partitioned`` for a specific implementation.
  """

  @abc.abstractmethod
  def unbox(self) -> Any:
    """Returns the content of the AxisMetadata box.

    Note that unlike ``meta.unbox`` the unbox call should recursively unbox
    metadata. It should simply return value that it wraps directly even
    if that value itself is an instance of AxisMetadata.

    In practise, AxisMetadata subclasses should be registered as PyTree nodes to
    support passing instances to JAX and Flax APIs. The leaves returned for this
    note should correspond to the value returned by unbox.

    Returns:
      The unboxed value.
    """
    pass

  @abc.abstractmethod
  def replace_boxed(self, val: Any) -> TAxisMetadata:
    """Replaces the boxed value with the provided value.

    Args:
      val: The new value to be boxed by this AxisMetadata wrapper
    Returns:
      A new instance of the same type as self with `val` as the new ``unbox``
      content
    """
    pass

  @abc.abstractmethod
  def add_axis(self: TAxisMetadata, index: int,
               params: Dict[Any, Any]) -> TAxisMetadata:
    """Adds a new axis to the axis metadata.

    Note that add_axis and remove_axis should act as each other's inverse
    (meaning: ``x.add_axis(i, p).remove_axis(i, p) == x``)

    Args:
      index: The position at which the new axis will be inserted
      params: An arbitrary dictionary of parameters passed by the transformation
        that introduces the new axis (e.g.: ``nn.scan`` or ``nn.vmap``). The
        user passes this dictionary as the `metadata_param` argument to the
        transformation.
    Returns:
      A new instance of the same type as self and with the same ``unbox``
      content with updated axis metadata.
    """
    pass

  @abc.abstractmethod
  def remove_axis(self: TAxisMetadata, index: int,
                  params: Dict[Any, Any]) -> TAxisMetadata:
    """Removes an axis from the axis metadata.

    Note that add_axis and remove_axis should act as each other's inverse
    (meaning: ``x.remove_axis(i, p).add_axis(i, p) == x``)

    Args:
      index: The position of the axis that is to be removed
      params: An arbitrary dictionary of parameters passed by the transformation
        that introduced the axis (e.g.: ``nn.scan`` or ``nn.vmap``). The
        user passes this dictionary as the `metadata_param` argument to the
        transformation.
    Returns:
      A new instance of the same type as self and with the same ``unbox``
      content with updated axis metadata.
    """
    pass


def is_axis_metadata(val: Any) -> bool:
  """Returns whether the argument is an instance of AxisMetadata."""
  return isinstance(val, AxisMetadata)


def map_axis_meta(fn: Callable[[AxisMetadata], Any], tree: Any) -> Any:
  """Maps over all PyTree nodes that are AxisMetadata instances."""
  def wrapper(x):
    if isinstance(x, AxisMetadata):
      return fn(x)
    else:
      return x
  return jax.tree_map(wrapper, tree, is_leaf=is_axis_metadata)


def add_axis(tree: Any, index: int, params: Dict[Any, Any]) -> Any:
  """Add an axis to each AxisMetadata node in a PyTree."""
  return map_axis_meta(lambda x: x.add_axis(index, params), tree)


def remove_axis(tree: Any, index: int, params: Dict[Any, Any]) -> Any:
  """Remove an axis from each AxisMetadata node in a PyTree."""
  return map_axis_meta(lambda x: x.remove_axis(index, params), tree)


def unbox(tree: Any) -> Any:
  """Strips all AxisMetadata boxes from a PyTree."""
  return map_axis_meta(lambda x: unbox(x.unbox()), tree)


def replace_boxed(tree: Any, updates: Any) -> Any:
  """Updates all AxisMetadata boxes with the values in updates."""
  def inner_update(c, v):
    if isinstance(c, AxisMetadata):
      return c.replace_boxed(replace_boxed(c.unbox(), v))
    else:
      return v
  return jax.tree_map(inner_update, tree, updates, is_leaf=is_axis_metadata)


PARTITION_NAME = 'partition_name'
LogicalNames = Tuple[Union[str, None], ...]


def _global_mesh_defined() -> bool:
  """Checks if global xmap/pjit mesh resource environment is defined."""
  maps_env = maps.thread_resources.env
  return maps_env.physical_mesh.devices.shape != ()  # pylint: disable=g-explicit-bool-comparison


class Partitioned(struct.PyTreeNode, AxisMetadata):
  """Wrapper for partitioning metadata.

  ``Partitioned`` is used to extend variables with partitioning information
  required for ``jax.experimental.pjit``.

  The easiest way to define Partitioned variables is by using the
  ``with_partitioning`` wrapper around the variable initializer.

  Example::

    class MLP(nn.Module):
      hidden_size: int
      @nn.compact
      def __call__(self, x):
        ki = nn.linear.default_kernel_init
        h = nn.Dense(
            self.hidden_size,
            kernel_init=nn.with_partitioning(ki, ('data', 'model')))(x)
        h = nn.relu(h)
        return nn.Dense(
            x.shape[-1],
            kernel_init=nn.with_partitioning(ki, ('model', 'data')))(h)
    mlp = MLP(4096)
    x = jnp.ones((8 * 1024, 1024))
    # use eval_shape to get the Partitioned instances for the variables.
    # this way we can determinte the PartitionSpecs for the init variables
    # before we call the init fn.
    var_spec = nn.get_partition_spec(
        jax.eval_shape(mlp.init, random.PRNGKey(0), x))
    init_fn = mesh(pjit(mlp.init,
                        (None, PartitionSpec("data", "model")), var_spec))
    variables = init_fn(random.PRNGKey(0), x)
    apply_fn = mesh(pjit(
        mlp.apply,
        (var_spec, PartitionSpec("data", "model")),
         PartitionSpec("data", "model")))
    apply_fn(variables, x)


  ``Partitioned`` values can gain additional axes when using transformations
  like ``nn.vmap`` and ``nn.scan``. In this case you can specify the name of
  the new axis with the `metadata_params` args in vmap/scan::

    class Model(nn.Module):
    @nn.compact
    def __call__(self, x):
      def body(mdl, c):
        c = MLP(4096)(c)
        return c, ()
      c, _ = nn.scan(
          body, variable_axes={"params": 0}, split_rngs={"params": 0}, length=8,
          metadata_params={nn.meta.PARTITION_NAME: "layers"})(self, x)
      return c

  """
  value: Any
  names: LogicalNames = struct.field(pytree_node=False)

  def unbox(self, apply_constraint=True) -> Any:
    """Returns the wrapped value with the partitioning applied as a sharding constraint."""
    if apply_constraint and _global_mesh_defined():
      return pjit.with_sharding_constraint(
          self.value, self.get_partition_spec())
    else:
      return self.value

  def replace_boxed(self, val: Any) -> TAxisMetadata:
    return self.replace(value=val)

  def _get_partition_name(self, params: Dict[Any, Any]) -> str:
    if PARTITION_NAME not in params:
      raise errors.PartitioningUnspecifiedError(self)
    return params[PARTITION_NAME]

  def add_axis(self, index: int, params: Dict[Any, Any]) -> TAxisMetadata:
    axis_name = self._get_partition_name(params)
    names = list(self.names)
    while len(names) < index:
      names.append(None) # type: ignore
    names.insert(index, axis_name) # type: ignore
    return self.replace(names=tuple(names))

  def remove_axis(self, index: int, params: Dict[Any, Any]) -> TAxisMetadata:
    axis_name = self._get_partition_name(params)
    names = list(self.names)
    assert names.pop(index) == axis_name
    return self.replace(names=tuple(names))

  def get_partition_spec(self) -> jax.sharding.PartitionSpec:
    """Returns the ``Partitionspec`` for this partitioned value."""
    return jax.sharding.PartitionSpec(*self.names)


def with_partitioning(
    fn: Callable[..., Any],
    names: LogicalNames) ->  Callable[..., Partitioned]:
  """Wraps a function's return value with Partitioned.

  Example::

    kernel_init = with_partitioning(
        nn.initializers.lecun_normal, (None, "data"))
    partitioned_dense = nn.Dense(features, kernel_init=kernel_init)

  Args:
    fn: The function to be wrapped. Typically this is an initializer.
    names: The logical axis passed to ``Partitioned``.
  Returns:
    A function wrapping ``fn`` that will return an instance of ``Partitioned``.
  """
  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    return Partitioned(fn(*args, **kwargs), names)
  return wrapper


def get_partition_spec(tree: Any) -> Any:
  """Extracts a PartitionSpec tree from a PyTree containing ``Partitioned`` values."""
  def f(x):
    if isinstance(x, Partitioned):
      return x.get_partition_spec()
    else:
      return None
  return jax.tree_map(f, tree,
                      is_leaf=lambda x: isinstance(x, Partitioned))
