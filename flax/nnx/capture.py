import jax
from contextlib import contextmanager
from jax._src.hijax import Box
from functools import partial
from threading import local

BOXES = local()

@contextmanager
def capture_intermediates(module=None, method_outputs=False):
    if not hasattr(BOXES, 'local'):
     BOXES.local = []
    BOXES.local.append(Box({}))
    if module: module._save_paths(method_outputs=method_outputs)
    try:
        yield
    finally:
        BOXES.local.pop()
        if module: module._del_paths(method_outputs=method_outputs)

def capture_fwd(name, value):
    "Record an intermediate value"
    if len(BOXES.local) > 0:
        box = BOXES.local[-1]
        box.set({**box.get(), name: jax.lax.stop_gradient(value)})
    else:
        raise ValueError("`capture_fwd` must be called within a `capture_intermediates` context.")

def get_intermediates():
  if BOXES.local:
    return BOXES.local[-1].get()
  else:
    raise ValueError("`get_intermediates` must be called within a `capture_intms`")

@partial(jax.custom_vjp, nondiff_argnums=(0,))
def capture_bwd(name, a):
  "Record the intermediate gradient of the given value"
  return a

def _fn_fwd(name, a):
    return a, None

def _fn_bwd(name, _, g):
    capture_fwd(name, g)
    return g,
capture_bwd.defvjp(_fn_fwd, _fn_bwd)
