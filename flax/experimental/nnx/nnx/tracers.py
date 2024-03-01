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

# Taken from flax/core/tracer.py 🏴‍☠️

import typing as tp

import jax
import jax.core
from jax.core import MainTrace

from flax.experimental.nnx.nnx import reprlib


@tp.runtime_checkable
class Tracer(tp.Protocol):
  _trace: jax.core.Trace


def get_top_trace(pytree: tp.Union[tp.Any, Tracer]) -> MainTrace:
  """Returns the main top trace of a sequence of tracers."""
  if isinstance(pytree, Tracer):
    return pytree._trace.main

  return jax.core.find_top_trace(jax.tree_util.tree_leaves(pytree)).main


def current_jax_trace() -> MainTrace:
  """Returns the innermost Jax tracer."""
  return get_top_trace(())


class TraceState(reprlib.Representable):
  __slots__ = ['_jax_trace']

  def __init__(self):
    self._jax_trace = current_jax_trace()

  @property
  def jax_trace(self):
    return self._jax_trace

  def is_valid(self) -> bool:
    return self._jax_trace is current_jax_trace()

  def __nnx_repr__(self):
    yield reprlib.Object(f'{type(self).__name__}')
    yield reprlib.Attr('jax_trace', self._jax_trace)

  def __eq__(self, other):
    return isinstance(other, TraceState) and self._jax_trace is other._jax_trace
