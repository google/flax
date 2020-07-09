# Lint as: python3
"""Functionality for inspecting jax tracers."""


import jax


def _masters():
  """Returns a list of currently active Jax tracers."""
  # stack = jax.core.trace_state.trace_stack
  # return stack.downward[::-1] + stack.upward
  return []


def current_trace():
  """Returns the innermost Jax tracer."""
  tracers = _masters()
  if tracers:
    return tracers[-1]
  return None


def trace_level(master):
  """Returns the level of the trace of -infinity if it is None."""
  if master:
    return master.level
  return float('-inf')


def check_trace_level(base_level):
  level = trace_level(current_trace())
  if level > base_level:
    raise ValueError('Jax transforms and modules cannot be mixed.')
