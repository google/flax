# Copyright 2023 The Flax Authors.
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

import contextlib
import dataclasses
import threading
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


def get_all_traces(pytree: tp.Union[tp.Any, Tracer]) -> tp.Set[MainTrace]:
  """Returns True if all tracers have the same main trace."""
  if isinstance(pytree, Tracer):
    return {pytree._trace.main}
  else:
    return {
      trace._trace.main
      for trace in jax.tree_util.tree_leaves(pytree)
      if isinstance(trace, Tracer)
    }


def trace_level(main):
  """Returns the level of the trace of -infinity if it is None."""
  if main:
    return main.level
  return float('-inf')


@dataclasses.dataclass
class TraceContext(threading.local):
  nnx_trace_stack: tp.List[MainTrace] = dataclasses.field(
    default_factory=lambda: [current_jax_trace()]
  )


TRACE_CONTEXT = TraceContext()


@contextlib.contextmanager
def nnx_trace(trace: MainTrace):
  TRACE_CONTEXT.nnx_trace_stack.append(trace)
  try:
    yield
  finally:
    TRACE_CONTEXT.nnx_trace_stack.pop()


def current_nnx_trace() -> MainTrace:
  return TRACE_CONTEXT.nnx_trace_stack[-1]


class TraceState(reprlib.Representable):
  __slots__ = ['_jax_trace', '_nnx_trace']

  def __init__(self):
    self._jax_trace = current_jax_trace()
    self._nnx_trace = current_nnx_trace()

  @property
  def jax_trace(self):
    return self._jax_trace

  @property
  def nnx_trace(self):
    return self._nnx_trace

  def is_valid(self) -> bool:
    return (
      self._jax_trace is current_jax_trace()
      and self._nnx_trace is current_nnx_trace()
    )

  def __nnx_repr__(self):
    yield reprlib.Object(f'{type(self).__name__}')
    yield reprlib.Attr('jax_trace', self._jax_trace)
    yield reprlib.Attr('nnx_trace', self._nnx_trace)
