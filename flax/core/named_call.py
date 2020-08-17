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
