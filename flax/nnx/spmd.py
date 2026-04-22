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

import flax.core.spmd as core_spmd
from flax.nnx import variablelib, graphlib
from flax.nnx.transforms.transforms import eval_shape
from flax.typing import (
  Sharding,
)
import jax
from jax.sharding import PartitionSpec
from flax.nnx.deprecations import deprecated

A = tp.TypeVar('A')
F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])
PARTITION_NAME = 'partition_name'


# Transform axis change helpers
# ------------------------------------------------------------------------------


def add_axis(tree: A, index: int, transform_metadata: tp.Mapping) -> A:
  axis_name, other_meta = _get_partition_name_and_metadata(transform_metadata)

  def insert_field(fields, index, value):
    iterable = list(fields)
    while len(iterable) < index:
      iterable.append(None)
    iterable.insert(index, value)
    return tuple(iterable)

  def _add_axis(x: tp.Any):
    if isinstance(x, variablelib.Variable):
      metadata = x.get_metadata()
      if 'out_sharding' in metadata and metadata['out_sharding']:
        sharding = metadata['out_sharding']
        x.set_metadata(out_sharding=insert_field(sharding, index, axis_name))

      for k, v in other_meta.items():
        if hasattr(x, k) and (t := getattr(x, k)) and isinstance(t, tuple):
          x.set_metadata(k, insert_field(t, index, v))

      assert isinstance(x, variablelib.Variable)
      x.add_axis(index, axis_name)
    return x

  return jax.tree.map(
    _add_axis, tree, is_leaf=lambda x: isinstance(x, variablelib.Variable)
  )


def remove_axis(
  tree: A, index: int, transform_metadata: tp.Mapping[tp.Any, tp.Any]
) -> A:
  axis_name, other_meta = _get_partition_name_and_metadata(transform_metadata)

  def remove_field(fields, index, value):
    iterable = list(fields)
    removed = iterable.pop(index)
    if removed != value:
      raise ValueError(
        f'Expected to remove {value!r} at index {index} from '
        f'{fields!r}, but found {removed!r}.'
      )
    return tuple(iterable)

  def _remove_axis(x: tp.Any):
    if isinstance(x, variablelib.Variable):
      if hasattr(x, 'out_sharding') and x.out_sharding is not None:
        x.set_metadata(
          out_sharding=remove_field(x.out_sharding, index, axis_name)
        )

      for k, v in other_meta.items():
        if hasattr(x, k) and (t := getattr(x, k)) and isinstance(t, tuple):
          x.set_metadata(k, remove_field(t, index, v))

      x.remove_axis(index, axis_name)
    return x

  return jax.tree.map(
    _remove_axis,
    tree,
    is_leaf=lambda x: isinstance(x, variablelib.Variable),
  )


def _get_partition_name_and_metadata(
  transform_metadata: tp.Mapping[tp.Any, tp.Any],
) -> tuple[str, tp.Mapping[tp.Any, tp.Any]]:
  if PARTITION_NAME not in transform_metadata:
    raise ValueError(
      'Trying to transform a Partitioned variable but "partition_name" '
      f'is not specified in transform_metadata: {transform_metadata}'
    )
  other_meta = dict(transform_metadata)  # shallow copy
  other_meta.pop(PARTITION_NAME)
  return transform_metadata[PARTITION_NAME], other_meta


# Annotation handling
# ------------------------------------------------------------------------------


def with_partitioning(
  initializer: F,
  sharding: Sharding,
  mesh: tp.Optional[jax.sharding.Mesh] = None,
  **metadata: tp.Any,
) -> F:
  """A wrapper over any initializer to add sharding annotation data to a `Variable`."""
  return variablelib.with_metadata(
    initializer,
    out_sharding=sharding,
    mesh=mesh,
    **metadata,
  )


def get_var_pspec(v: variablelib.Variable) -> PartitionSpec | None:
  """Given an `nnx.Variable`, return its `PartitionSpec`."""
  metadata = v.get_metadata()
  if 'out_sharding' in metadata and metadata['out_sharding']:
    sharding = metadata['out_sharding']
    if core_spmd.get_logical_axis_rules() or 'sharding_rules' in metadata:
      context_rules = core_spmd.get_logical_axis_rules()
      local_rules = metadata.get('sharding_rules', ())
      rules = core_spmd.composite_rules(context_rules, local_rules)
      return PartitionSpec(*core_spmd.from_sharding_rules(sharding, rules))
    return PartitionSpec(*sharding)
  elif hasattr(v, 'shape'):
      return PartitionSpec()
  return None


def get_partition_spec(tree: A) -> A:
  """Extracts a PartitionSpec tree from a PyTree containing ``Variable`` values."""

  def f(x):
    if isinstance(x, variablelib.Variable):
      return x.replace(get_var_pspec(x))
    elif hasattr(x, 'shape'):
        return PartitionSpec()
    return None

  return jax.tree.map(
    f, tree, is_leaf=lambda x: isinstance(x, variablelib.Variable)
  )


def get_named_sharding(tree: A, mesh: jax.sharding.Mesh) -> A:
  spec = get_partition_spec(tree)
  sharding = jax.tree.map(lambda p: jax.sharding.NamedSharding(mesh, p), spec)
  return sharding


# Other utilities
# ------------------------------------------------------------------------------


def get_abstract_model(init_fn, mesh, *, graph: bool | None = None):
  with jax.set_mesh(mesh):
    abs_model = eval_shape(init_fn, graph=graph)
    gdef, abs_state = graphlib.split(abs_model, graph=graph)
    abs_state = jax.tree.map(
      lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s),
      abs_state, get_named_sharding(abs_state, mesh)
    )
  return gdef, abs_state


def as_abstract(
    tree: A, graph: bool | None = None
) -> A:
  """Add sharding information to abstract Variables.

  When creating models with :func:`eval_shape`, Variables are abstract
  (backed by ``jax.ShapeDtypeStruct``) and may not carry sharding
  information, especially when using meshes with
  :attr:`jax.sharding.AxisType.Auto` axes. ``abstract_with_sharding`` inspects each
  Variable in ``tree`` and, if it has ``out_sharding`` metadata but no
  sharding already set, attaches a :class:`jax.sharding.NamedSharding`
  derived from the Variable's ``out_sharding`` and either its ``mesh``
  metadata or the current abstract mesh (``jax.sharding.get_abstract_mesh``).

  Example usage::

    from flax import nnx
    import jax

    mesh = jax.make_mesh((2, 2), ('a', 'b'),
      axis_types=(jax.sharding.AxisType.Auto,) * 2)
    with jax.set_mesh(mesh):
      abs_model = nnx.eval_shape(
        lambda: nnx.Linear(4, 8, rngs=nnx.Rngs(0),
          kernel_metadata={'out_sharding': ('a', 'b')}))
      abs_model = nnx.as_abstract(abs_model)
    assert abs_model.kernel.sharding.spec == jax.P('a', 'b')

  Args:
    tree: A graph node (e.g. an :class:`nnx.Module`) whose Variables should
      be annotated with sharding (via ``out_sharding`` metadata).
    graph: Forwarded to :func:`nnx.map`. If ``True``, uses graph-mode;
      if ``False``, uses tree-mode.
  Returns:
    A tree with sharding-annotated ShapeDtypeStruct values inside Variables.
  """
  def add_sharding(_path, x):
    if (
        isinstance(x, variablelib.Variable)
        and hasattr(value := x.get_value(), 'shape')
        and hasattr(value, 'dtype')
        and getattr(value, 'sharding', None) is None
        and x.has_metadata('out_sharding')
    ):
      if x.has_metadata('mesh'):
        mesh = x.get_metadata('mesh')
      else:
        mesh = jax.sharding.get_abstract_mesh()
      specs = get_var_pspec(x)
      sharding = jax.sharding.NamedSharding(mesh, specs)  # pyrefly: ignore [bad-argument-type]
      abs_var = x.replace(
          jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=sharding)
      )
      return abs_var
    return x
  return graphlib.map(add_sharding, tree, graph=graph)

abstract_with_sharding = deprecated(as_abstract)
