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
from collections import namedtuple

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


def model_summary(model):
  """Returns a summary of the model's parameters.

  Args:
    model: the nn.Model of the model.
  Returns:
    A string summarizing the model.
  ----------------------------------------------------------
  Parameters                     Shape     Number       Type
  ==========================================================
  BatchNorm_1/bias               (32,)         32    float32
  BatchNorm_1/scale              (32,)         32    float32
  Conv_0/bias                    (32,)         32    float32
  Conv_0/kernel          (3, 3, 1, 32)        288    float32
  Conv_2/bias                    (64,)         64    float32
  Conv_2/kernel         (3, 3, 32, 64)      18432    float32
  Dense_3/bias                  (256,)        256    float32
  Dense_3/kernel           (3136, 256)     802816    float32
  Dense_4/bias                   (10,)         10    float32
  Dense_4/kernel             (256, 10)       2560    float32
  ==========================================================
  Total Parameters: 824,522
  Total Size: 3.1 MB
  ----------------------------------------------------------
  """
  Parameters = namedtuple("Parameters", ["shape", "number", "type", "size"])

  # Get parameter names
  param_names = []
  for layer in model.params.keys():
    for params in list(model.params[layer].keys()):
      param_names.append("{}/{}".format(layer, params))

  # Get parameter shapes, numbers, types, sizes
  param_info = []
  for params in jax.tree_flatten(model.params)[0]:
    param_info.append(Parameters(
      onp.shape(params),
      onp.prod(onp.shape(params)),
      params.dtype,
      params.dtype.itemsize * params.size / 2**20
    ))

  summary_str = "----------------------------------------------------------\n"
  summary_str += "{:<20} {:>15} {:>10} {:>10}\n".format("Parameters", "Shape", "Number", "Type")
  summary_str += "==========================================================\n"
  
  # Totals
  total_params = total_size = 0
  for i in range(len(param_names)):
    summary_str += "{:<20} {:>15} {:>10} {:>10}\n".format(param_names[i],
      str(param_info[i].shape), param_info[i].number, str(param_info[i].type))
    total_params += param_info[i].number
    total_size += param_info[i].size

  summary_str += "==========================================================\n"
  summary_str += "Total Parameters: {:,d}\n".format(total_params)
  summary_str += "Total Size: {} MB\n".format(round(total_size, 1))
  summary_str += "----------------------------------------------------------\n"

  return summary_str
