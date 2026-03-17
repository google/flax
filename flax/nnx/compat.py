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
"""Compat API.

Compatibility wrappers for NNX APIs. Each function in this module mirrors the
corresponding ``nnx.*`` API but enforces ``graph=True`` (and
``graph_updates=True`` for transforms), preserving the pre-tree-mode behavior.
"""

import functools

from flax.nnx import graphlib as _graphlib
from flax.nnx import module as _module
from flax.nnx import rnglib as _rnglib
from flax.nnx.transforms import autodiff as _autodiff
from flax.nnx.transforms import compilation as _compilation
from flax.nnx.transforms import iteration as _iteration
from flax.nnx.transforms import transforms as _transforms
from flax.nnx import spmd as _spmd

# graphlib
split = functools.partial(_graphlib.split, graph=True)
state = functools.partial(_graphlib.state, graph=True)
clone = functools.partial(_graphlib.clone, graph=True)
graphdef = functools.partial(_graphlib.graphdef, graph=True)
flatten = functools.partial(_graphlib.flatten, graph=True)
iter_graph = functools.partial(_graphlib.iter_graph, graph=True)
recursive_map = functools.partial(_graphlib.recursive_map, graph=True)

# module
view = functools.partial(_module.view, graph=True)
view_info = functools.partial(_module.view_info, graph=True)
iter_modules = functools.partial(_module.iter_modules, graph=True)
iter_children = functools.partial(_module.iter_children, graph=True)  # type: ignore[has-type]

# rnglib
split_rngs = functools.partial(_rnglib.split_rngs, graph=True)
fork_rngs = functools.partial(_rnglib.fork_rngs, graph=True)
reseed = functools.partial(_rnglib.reseed, graph=True)
backup_keys = functools.partial(_rnglib.backup_keys, graph=True)

# transforms - compilation
jit = functools.partial(_compilation.jit, graph=True, graph_updates=True)
shard_map = functools.partial(
    _compilation.shard_map, graph=True, graph_updates=True
)

# transforms - autodiff
grad = functools.partial(_autodiff.grad, graph=True, graph_updates=True)
value_and_grad = functools.partial(
    _autodiff.value_and_grad, graph=True, graph_updates=True
)
custom_vjp = functools.partial(
    _autodiff.custom_vjp, graph=True, graph_updates=True
)
vjp = functools.partial(_autodiff.vjp, graph=True, graph_updates=True)
jvp = functools.partial(_autodiff.jvp, graph=True, graph_updates=True)
remat = functools.partial(_autodiff.remat, graph=True, graph_updates=True)

# transforms - iteration
vmap = functools.partial(_iteration.vmap, graph=True, graph_updates=True)
scan = functools.partial(_iteration.scan, graph=True, graph_updates=True)
pmap = functools.partial(_iteration.pmap, graph=True, graph_updates=True)
while_loop = functools.partial(
    _iteration.while_loop, graph=True, graph_updates=True
)
fori_loop = functools.partial(
    _iteration.fori_loop, graph=True, graph_updates=True
)

# transforms - general
eval_shape = functools.partial(
    _transforms.eval_shape, graph=True, graph_updates=True
)
cond = functools.partial(_transforms.cond, graph=True, graph_updates=True)
switch = functools.partial(_transforms.switch, graph=True, graph_updates=True)
checkify = functools.partial(
    _transforms.checkify, graph=True, graph_updates=True
)

# spmd
get_abstract_model = functools.partial(_spmd.get_abstract_model, graph=True)
