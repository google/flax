# Lint as: python3

# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NN base modules for JAX."""

import contextlib
import threading
import jax


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


def _current_trace():
  """Returns the innermost Jax tracer."""
  tracers = _masters()
  if tracers:
    return tracers[-1]
  return None


def _tracer_of_value(x):
  """Returns the tracer associated with a value if any."""
  if hasattr(x, '_trace'):
    return x._trace.master
  return None
