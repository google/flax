# Copyright 2025 The Flax Authors.
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

import typing as tp

from flax.nnx import graph, rnglib
from flax.nnx.bridge import module as bdg_module
import flax.nnx.module as nnx_module
from flax.nnx.transforms.transforms import eval_shape as nnx_eval_shape
from flax.nnx.transforms.compilation import jit as nnx_jit


def wrap_nnx_mdl(factory: tp.Callable[[rnglib.Rngs], nnx_module.Module],
                 name: str | None = None):
  """Create module at init time, or make abstract module and let parent bind it with its state. Use current bridge module scope for RNG generation."""
  parent_ctx, parent = bdg_module.current_context(), bdg_module.current_module()
  assert parent_ctx is not None and parent is not None, 'wrap_nnx_mdl only needed inside bridge Module'
  parent = parent_ctx.module
  assert parent.scope is not None

  if parent.is_initializing():
    module = factory(parent.scope.rngs)
  else:
    rngs = parent.scope.rngs if parent.scope.rngs else rnglib.Rngs(7)  # dummy
    module = nnx_eval_shape(factory, rngs)

    @nnx_jit
    def rng_state(rngs):
      return graph.state(factory(rngs), rnglib.RngState)

    # Make sure the internal rng state is not abstract - other vars shall be
    if parent.scope.rngs:
      graph.update(module, rng_state(parent.scope.rngs))

  # Automatically set the attribute if compact. If setup, user is responsible
  # for setting the attribute of the superlayer.
  if parent_ctx.in_compact:
    if name is None:
      name = bdg_module._auto_submodule_name(parent_ctx, type(module))
    setattr(parent, name, module)
  return module
