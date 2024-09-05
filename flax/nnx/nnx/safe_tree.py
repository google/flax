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
from flax.nnx.nnx import graph

import jax


def _get_is_leaf(is_leaf):
  def _is_leaf_or_graph_node(x):
    if graph.is_graph_node(x):
      return True
    elif is_leaf is None:
      return False
    else:
      return is_leaf(x)

  return _is_leaf_or_graph_node


if tp.TYPE_CHECKING:
  map = jax.tree_util.tree_map
  leaves = jax.tree_util.tree_leaves
  flatten = jax.tree_util.tree_flatten
  structure = jax.tree_util.tree_structure
  flatten_with_path = jax.tree_util.tree_flatten_with_path
  leaves_with_path = jax.tree_util.tree_leaves_with_path
  map_with_path = jax.tree_util.tree_map_with_path
else:

  def map(*args, is_leaf=None, **kwargs):
    is_leaf = _get_is_leaf(is_leaf)
    return jax.tree_util.tree_map(*args, is_leaf=is_leaf, **kwargs)

  def leaves(*args, is_leaf=None, **kwargs):
    is_leaf = _get_is_leaf(is_leaf)
    return jax.tree_util.tree_leaves(*args, is_leaf=is_leaf, **kwargs)

  def flatten(*args, is_leaf=None, **kwargs):
    is_leaf = _get_is_leaf(is_leaf)
    return jax.tree_util.tree_flatten(*args, is_leaf=is_leaf, **kwargs)

  def structure(*args, is_leaf=None, **kwargs):
    is_leaf = _get_is_leaf(is_leaf)
    return jax.tree_util.tree_structure(*args, is_leaf=is_leaf, **kwargs)

  def flatten_with_path(*args, is_leaf=None, **kwargs):
    is_leaf = _get_is_leaf(is_leaf)
    return jax.tree_util.tree_flatten_with_path(
      *args, is_leaf=is_leaf, **kwargs
    )

  def leaves_with_path(*args, is_leaf=None, **kwargs):
    is_leaf = _get_is_leaf(is_leaf)
    return jax.tree_util.tree_leaves_with_path(*args, is_leaf=is_leaf, **kwargs)

  def map_with_path(*args, is_leaf=None, **kwargs):
    is_leaf = _get_is_leaf(is_leaf)
    return jax.tree_util.tree_map_with_path(*args, is_leaf=is_leaf, **kwargs)
