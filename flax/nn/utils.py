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

