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

import dataclasses
import importlib.util
import typing as tp

import jax

from flax.experimental import nnx

penzai_installed = importlib.util.find_spec('penzai') is not None
try:
  from IPython import get_ipython

  in_ipython = get_ipython() is not None
except ImportError:
  in_ipython = False


def display(*args):
  """Display the given objects using a Penzai visualizer.

  If Penzai is not installed or the code is not running in IPython, ``display``
  will print the objects instead.
  """
  if not penzai_installed or not in_ipython:
    for x in args:
      print(x)
    return

  from penzai import pz

  with pz.ts.active_autovisualizer.set_scoped(pz.ts.ArrayAutovisualizer()):
    for x in args:
      value = to_dataclass(x)
      pz.ts.display(value, ignore_exceptions=True)


def to_dataclass(node):
  seen_nodes = set()
  return _treemap_to_dataclass(node, seen_nodes)


def _to_dataclass(x, seen_nodes: set[int]):
  if nnx.graph.is_graph_node(x):
    if id(x) in seen_nodes:
      dc_type = _make_dataclass_obj(
        type(x),
        {'repeated': True},
      )
      return dc_type
    seen_nodes.add(id(x))
    node_impl = nnx.graph.get_node_impl(x)
    node_dict = node_impl.node_dict(x)
    node_dict = {
      str(key): _treemap_to_dataclass(value, seen_nodes)
      for key, value in node_dict.items()
    }
    dc_type = _make_dataclass_obj(
      type(x),
      node_dict,
    )
    return dc_type
  elif isinstance(x, (nnx.Variable, nnx.VariableState)):
    obj_vars = vars(x).copy()
    if 'raw_value' in obj_vars:
      obj_vars['value'] = obj_vars.pop('raw_value')
    if '_trace_state' in obj_vars:
      del obj_vars['_trace_state']
    for name in list(obj_vars):
      if name.endswith('_hooks'):
        del obj_vars[name]
    obj_vars = {
      key: _treemap_to_dataclass(value, seen_nodes)
      for key, value in obj_vars.items()
    }
    dc_type = _make_dataclass_obj(
      type(x),
      obj_vars,
      penzai_dataclass=not isinstance(x, nnx.VariableState),
    )
    return dc_type
  elif isinstance(x, nnx.State):
    return _treemap_to_dataclass(x._mapping, seen_nodes)
  return x


def _treemap_to_dataclass(node, seen_nodes: set[int]):
  def _to_dataclass_fn(x):
    return _to_dataclass(x, seen_nodes)

  return jax.tree.map(
    _to_dataclass_fn,
    node,
    is_leaf=lambda x: isinstance(x, (nnx.VariableState, nnx.State)),
  )


def _make_dataclass_obj(
  cls, fields: dict[str, tp.Any], penzai_dataclass: bool = True
) -> tp.Type:
  from penzai import pz

  dataclass = pz.pytree_dataclass if penzai_dataclass else dataclasses.dataclass
  base = pz.Layer if penzai_dataclass else object

  attributes = {
    '__annotations__': {key: type(value) for key, value in fields.items()},
  }

  if hasattr(cls, '__call__'):
    attributes['__call__'] = cls.__call__

  dc_type = type(cls.__name__, (base,), attributes)
  dc_type = dataclass(dc_type)
  return dc_type(**fields)