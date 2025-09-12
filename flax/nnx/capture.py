from flax.nnx.module import iter_modules, Capture
from flax.nnx.graph import pop
import functools as ft

def capture_intermediates(
  module, *args, method='__call__', method_outputs=False, variable_type=Capture):
    for path, m in iter_modules(module):
      m.__capture__ = variable_type({})
      if method_outputs:
        _add_capturing(type(m), variable_type)
    try:
      result = getattr(module, method)(*args)
    finally:
      if method_outputs:
        for _, m in iter_modules(module):
            _remove_capturing(type(m))
      interms = {}
      _extract_state(pop(module, variable_type), interms)
    return result, interms

def path_join(path, k):
  return "/".join(filter(None,[path, str(k)]))

def _extract_state(state, interms):
  for (k,v) in state.items():
    if k == '__capture__':
      interms.update(v.get_value())
    else:
      interms2 = {}
      _extract_state(v, interms2)
      interms[k] = interms2

def _add_capturing(cls, variable_type):
  """Adds capturing to methods of a Module.
  Does not instrument superclass methods."""
  for name, method in cls.__dict__.items():
    if callable(method) and (not name.startswith('_') or name == '__call__'):
      if not hasattr(method, '_does_capturing'):
        def closure(name, method): # Necessary to make 'name' immutable during iteration
          @ft.wraps(method)
          def wrapper(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            self.sow(variable_type, name, result)
            return result
          wrapper._does_capturing = True
          setattr(cls, name, wrapper)
        closure(name, method)
  return cls

def _remove_capturing(cls):
  """Remove capturing methods from a Module."""
  for name, method in cls.__dict__.items():
    if hasattr(method, '_does_capturing'):
      setattr(cls, name, method.__wrapped__)
  return cls
