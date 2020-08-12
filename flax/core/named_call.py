"""Named call primitive.

Only differs from JAX core.call_p by providing an explicit name.
Used for annotating profiles.
"""

import functools
import jax
from jax import core
from jax.interpreters import ad
from jax.interpreters import xla

# Registering named call as a primitive
named_call_p = core.CallPrimitive('named_call')
named_call_p.def_impl(core.call_impl)
ad.primitive_transposes[named_call_p] = functools.partial(ad.call_transpose,
                                                          named_call_p)

def _named_call_translation_rule(c, axis_env, in_nodes, name_stack,
                                 *, name='core_call', backend, call_jaxpr):
  subc = xla.xb.make_computation_builder(name)
  args = [xla.xb.parameter(subc, i, c.GetShape(n))
          for i, n in enumerate(in_nodes)]
  out_nodes = xla.jaxpr_subcomp(subc, call_jaxpr, backend, axis_env, (),
                                jax.util.extend_name_stack(name_stack, name),
                                *args)
  subc = subc.Build(xla.xops.Tuple(subc, out_nodes))
  return xla.xops.Call(c, subc, list(in_nodes))

xla.call_translations[named_call_p] = _named_call_translation_rule
