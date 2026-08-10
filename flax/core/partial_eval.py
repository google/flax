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

import functools

import jax
from jax.extend import core as jex_core
from jax.interpreters import partial_eval as pe

from flax import errors


def lazy_init(fn):
  """Lazily evaluates a function by using the shapes of the inputs.

  The returned function accepts a combination of JAX values and
  ``jax.ShapeDtypeStruct`` instances for the inputs for which we
  don't need concrete values (only the shape and dtype).

  This API is used by ``core.lazy_init`` or ``Module.lazy_init``
  to initialize variables without doing any actual computation on the
  inputs.

  Args:
    fn: the function to be lazily evaluated.
  Returns:
    A new function that accepts a mix of concrete values and
    ``jax.ShapeDtypeStruct`` instances.
  """

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    # Trace the function to a jaxpr, taking only the ShapeDtypeStruct
    # arguments as jaxpr inputs and closing over the concrete arguments. The
    # concrete arguments stay concrete during tracing (so e.g. Python control
    # flow on a concrete bool argument works), and any jax arrays among them
    # become constants of the jaxpr. Then use dead code elimination to check
    # that no output depends on an abstract input, and evaluate the jaxpr.
    paths_and_leaves, treedef = jax.tree_util.tree_flatten_with_path(
      (args, kwargs)
    )
    paths = [p for p, _ in paths_and_leaves]
    leaves = [x for _, x in paths_and_leaves]
    abstract_idxs = [
      i for i, x in enumerate(leaves) if isinstance(x, jax.ShapeDtypeStruct)
    ]

    def fn_closing_over_concrete(*abstract_leaves):
      leaves_ = list(leaves)
      for i, x in zip(abstract_idxs, abstract_leaves):
        leaves_[i] = x
      args_, kwargs_ = jax.tree_util.tree_unflatten(treedef, leaves_)
      return fn(*args_, **kwargs_)

    abstract_leaves = [leaves[i] for i in abstract_idxs]
    traced = jax.jit(fn_closing_over_concrete).trace(*abstract_leaves)
    jaxpr = traced.jaxpr.jaxpr
    # TODO(mattjj): dce_jaxpr_consts is not a public API, use one when possible
    dced_jaxpr, used_consts, used_inputs = pe.dce_jaxpr_consts(
      jaxpr, [True] * len(jaxpr.outvars)
    )
    assert len(used_inputs) == len(abstract_idxs)
    for used, i in zip(used_inputs, abstract_idxs):
      if used:
        raise errors.LazyInitError(_arg_name(paths[i]))
    consts = [c for c, used in zip(traced.jaxpr.consts, used_consts) if used]
    out_flat = jex_core.jaxpr_as_fun(jex_core.ClosedJaxpr(dced_jaxpr, consts))()
    return jax.tree_util.tree_unflatten(traced.out_tree, out_flat)

  return wrapper


def _arg_name(path) -> str:
  # path is a key path into the (args, kwargs) pair, e.g. the path
  # (SequenceKey(0), SequenceKey(1)) refers to the second positional argument.
  root, *rest = path
  prefix = 'args' if root.idx == 0 else 'kwargs'
  return prefix + jax.tree_util.keystr(tuple(rest))
