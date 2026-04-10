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
"""NNX Compat API.

The compat module provides wrappers that use the legacy graph-mode
implementation by default for many NNX APIs like ``split``, ``state``, and all
the transforms such as ``jit``, ``grad``, etc. It does so by changing the
default values to ``graph=True`` and ``graph_updates=True``.

Example::

  from flax import nnx

  graphdef, state = nnx.compat.split(model)  # graph=True by default

  @nnx.compat.jit  # graph=True, graph_updates=True by default
  def train_step(model, optimizer, x, y):
    ...

See
[Tree Mode
NNX](https://flax.readthedocs.io/en/latest/flip/5310-tree-mode-nnx.html#prefix-filters)
for more details.
"""

import functools

from flax.nnx.graphlib import split
from flax.nnx.graphlib import state
from flax.nnx.graphlib import clone
from flax.nnx.graphlib import graphdef
from flax.nnx.graphlib import flatten
from flax.nnx.graphlib import iter_graph
from flax.nnx.graphlib import recursive_map
from flax.nnx.graphlib import cached_partial
from flax.nnx.module import view
from flax.nnx.module import view_info
from flax.nnx.module import iter_modules
from flax.nnx.module import iter_children
from flax.nnx.rnglib import split_rngs
from flax.nnx.rnglib import fork_rngs
from flax.nnx.rnglib import reseed
from flax.nnx.rnglib import backup_keys
from flax.nnx.transforms.compilation import jit
from flax.nnx.transforms.compilation import shard_map
from flax.nnx.transforms.autodiff import grad
from flax.nnx.transforms.autodiff import value_and_grad
from flax.nnx.transforms.autodiff import custom_vjp
from flax.nnx.transforms.autodiff import vjp
from flax.nnx.transforms.autodiff import jvp
from flax.nnx.transforms.autodiff import remat
from flax.nnx.transforms.iteration import vmap
from flax.nnx.transforms.iteration import scan
from flax.nnx.transforms.iteration import pmap
from flax.nnx.transforms.iteration import while_loop
from flax.nnx.transforms.iteration import fori_loop
from flax.nnx.transforms.transforms import eval_shape
from flax.nnx.transforms.transforms import cond
from flax.nnx.transforms.transforms import switch
from flax.nnx.transforms.transforms import checkify
from flax.nnx.spmd import get_abstract_model
import typing as _tp

if not _tp.TYPE_CHECKING:
  # graphlib
  split = functools.partial(split, graph=True)
  state = functools.partial(state, graph=True)
  clone = functools.partial(clone, graph=True)
  graphdef = functools.partial(graphdef, graph=True)
  flatten = functools.partial(flatten, graph=True)
  iter_graph = functools.partial(iter_graph, graph=True)
  recursive_map = functools.partial(recursive_map, graph=True)
  cached_partial = functools.partial(cached_partial, graph=True, graph_updates=True)

  # module
  view = functools.partial(view, graph=True)
  view_info = functools.partial(view_info, graph=True)
  iter_modules = functools.partial(iter_modules, graph=True)
  iter_children = functools.partial(iter_children, graph=True)  # type: ignore[has-type]

  # rnglib
  split_rngs = functools.partial(split_rngs, graph=True, graph_updates=True)
  fork_rngs = functools.partial(fork_rngs, graph=True, graph_updates=True)
  reseed = functools.partial(reseed, graph=True)
  backup_keys = functools.partial(backup_keys, graph=True)

  # transforms - compilation
  jit = functools.partial(jit, graph=True, graph_updates=True)
  shard_map = functools.partial(shard_map, graph=True, graph_updates=True)

  # transforms - autodiff
  grad = functools.partial(grad, graph=True, graph_updates=True)
  value_and_grad = functools.partial(value_and_grad, graph=True, graph_updates=True)
  custom_vjp = functools.partial(custom_vjp, graph=True, graph_updates=True)
  vjp = functools.partial(vjp, graph=True, graph_updates=True)
  jvp = functools.partial(jvp, graph=True, graph_updates=True)
  remat = functools.partial(remat, graph=True, graph_updates=True)

  # transforms - iteration
  vmap = functools.partial(vmap, graph=True, graph_updates=True)
  scan = functools.partial(scan, graph=True, graph_updates=True)
  pmap = functools.partial(pmap, graph=True, graph_updates=True)
  while_loop = functools.partial(while_loop, graph=True, graph_updates=True)
  fori_loop = functools.partial(fori_loop, graph=True, graph_updates=True)

  # transforms - general
  eval_shape = functools.partial(eval_shape, graph=True, graph_updates=True)
  cond = functools.partial(cond, graph=True, graph_updates=True)
  switch = functools.partial(switch, graph=True, graph_updates=True)
  checkify = functools.partial(checkify, graph=True, graph_updates=True)

  # spmd
  get_abstract_model = functools.partial(get_abstract_model, graph=True)
