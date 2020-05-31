# Copyright 2020 The Flax Authors.
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

# Lint as: python3
"""NN base modules for JAX."""

import contextlib
import threading
import jax
import numpy as onp
from typing import Dict, Any 

class CallStack(object):
  """Utility for tracking data across a call stack."""

  def __init__(self):
    self._stack = threading.local()

  @property
  def _frames(self):
    if not hasattr(self._stack, 'frames'):
      self._stack.frames = []
    return self._stack.frames

  @contextlib.contextmanager
  def frame(self, data=None):
    if data is None:
      data = {}
    self._frames.append(data)
    try:
      yield data
    finally:
      self._frames.pop(-1)

  def __iter__(self):
    return iter(self._frames)

  def __len__(self):
    return len(self._frames)

  def __getitem__(self, key):
    return self._frames.__getitem__(key)


def classproperty(f):
  """decorator that registers a function as a read-only property of the class."""

  class _ClassProperty:

    def __get__(self, _, cls):
      # python will call the __get__ magic function whenever the property is
      # read from the class.
      return f(cls)

  return _ClassProperty()


def _masters():
  """Returns a list of currently active Jax tracers."""
  stack = jax.core.trace_state.trace_stack
  return stack.downward[::-1] + stack.upward


def _trace_level(master):
  """Returns the level of the trace of -infinity if it is None."""
  if master:
    return master.level
  return float('-inf')


def _current_trace():
  """Returns the innermost Jax tracer."""
  tracers = _masters()
  if tracers:
    return tracers[-1]
  return None


def _level_of_value(xs):
  """Returns the tracer level associated with a value if any."""
  xs = jax.tree_leaves(xs)
  max_level = float('-inf')
  for x in xs:
    if hasattr(x, '_trace'):
      level = _trace_level(x._trace.master)
      max_level = max(level, max_level)
  return max_level


def flatten_dict(input_dict: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
  """Flattens the keys of a nested dictionary."""
  output_dict = {}
  for key, value in input_dict.items():
    nested_key = "{}/{}".format(prefix, key) if prefix else key
    if isinstance(value, dict):
      output_dict.update(flatten_dict(value, prefix=nested_key))
    else:
      output_dict[nested_key] = value
  return output_dict


def _name_idx(name: str):
  """Returns the layer index of the parameter name."""
  index = name[name.find('_') + 1 : name.find('/')]
  return int(index) if index[0] in '0123456789' else -1


def param_info(param: Dict[str, onp.ndarray]):
  """Returns dictionary of parameter shapes, numbers, types and bytes."""

  if not isinstance(param, dict):
    raise ValueError('Please provide a dictionary of parameters.')

  param = flatten_dict(param)
  # Sort parameter names by the order of layers in the module 
  param_names, param_values = map(list, tuple(zip(*sorted(
    param.items(), key=lambda item: _name_idx(item[0])))))
  
  param_info = {}
  for idx in range(len(param_values)):
    name = param_names[idx]
    value = param_values[idx]
    param_info[name] = {
      "shape": onp.shape(value), 
      "number": value.size,
      "type": value.dtype, 
      "bytes": value.dtype.itemsize * value.size
    }
  
  return param_info


def show_param_info(params: Dict[str, onp.ndarray], max_lines: int = None):
  """Returns a summary of the parameters.

  Args:
    params: dictionary of parameters.
    max_lines: number of paramters to be summarized.
  Returns:
    A string summarizing the parameters.
  ---------------------------------------------------------
  Parameters                     Shape     Number      Type
  =========================================================
  Conv_0/bias                    (32,)         32    float32
  Conv_0/kernel          (3, 3, 1, 32)        288    float32
  BatchNorm_1/bias               (32,)         32    float32
  BatchNorm_1/scale              (32,)         32    float32
  Conv_2/bias                    (64,)         64    float32
  Conv_2/kernel         (3, 3, 32, 64)      18432    float32
  Dense_3/bias                  (256,)        256    float32
  Dense_3/kernel           (3136, 256)     802816    float32
  Dense_4/bias                   (10,)         10    float32
  Dense_4/kernel             (256, 10)       2560    float32
  =========================================================
  Total Parameters: 824,522
  Total Size: 3.1 MB
  ---------------------------------------------------------
  """

  if not isinstance(params, dict):
    raise ValueError('Please provide a dictionary of parameters.')

  param_info_dict = param_info(params)
  names = list(param_info_dict.keys())
  values = list(param_info_dict.values())

  class _Column:

    def __init__(self, name, values):
      self.name = name
      self.values = values
      self.width = max(len(v) for v in values + [name])

  columns = [
    _Column("Parameters", names),
    _Column("Shape", [str(v["shape"]) for v in values]),
    _Column("Number", [str(v["number"]) for v in values]),
    _Column("Type", [str(v["type"]) for v in values]),
  ]

  offset = 2 # set distance between columns
  name_format = f"{{: <{columns[0].width + offset}s}}" # align parameters to the left
  value_format = name_format + "".join(f"{{: >{c.width + offset}s}}" for c in columns[1:])
  header = value_format.format(*[c.name for c in columns])

  dash_format = value_format.replace(" ", "-")
  equals_format = value_format.replace(" ", "=")
  separator_dash = dash_format.format(*["" for c in columns])
  separator_equals = equals_format.format(*["" for c in columns])

  lines = [separator_dash, header, separator_equals]
  for i in range(len(names)):
    if max_lines and len(lines) >= max_lines+3:
      lines.append("[...]")
      break
    lines.append(value_format.format(*[c.values[i] for c in columns]))

  total_parameters = sum(v["number"] for v in values)
  total_size = sum(v["bytes"] for v in values)
  lines.append(separator_equals)
  lines.append("Total parameters: {:,}".format(total_parameters))
  lines.append("Total size: {:,} MB".format(round(total_size / 2**20, 1)))
  lines.append(separator_dash)
  return "\n".join(lines)
