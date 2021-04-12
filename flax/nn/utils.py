# Copyright 2021 The Flax Authors.
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

"""DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  NN base modules for JAX."""

import contextlib
import threading
import jax


class CallStack(object):
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Utility for tracking data across a call stack."""

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
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  decorator that registers a function as a read-only property of the class."""

  class _ClassProperty:

    def __get__(self, _, cls):
      # python will call the __get__ magic function whenever the property is
      # read from the class.
      return f(cls)

  return _ClassProperty()


def _masters():
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Returns a list of currently active Jax tracers."""
  # TODO(jheek): consider re-introducing the tracer check
  # for now we pretent there are never any tracers
  return ()


def _trace_level(master):
  """DEPRECATION WARNING:
 The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Returns the level of the trace of -infinity if it is None."""
  if master:
    return master.level
  return float('-inf')


def _current_trace():
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Returns the innermost Jax tracer."""
  tracers = _masters()
  if tracers:
    return tracers[-1]
  return None


def _level_of_value(xs):
  """DEPRECATION WARNING:
  The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Returns the tracer level associated with a value if any."""
  xs = jax.tree_leaves(xs)
  max_level = float('-inf')
  # TODO(jheek): consider re-introducing the tracer check
  # for x in xs:
  #   if hasattr(x, '_trace'):
  #     level = _trace_level(x._trace.master)
  #     max_level = max(level, max_level)
  return max_level
