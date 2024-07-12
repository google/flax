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

import functools
import typing as tp

from flax import struct
from flax.nnx.nnx import (
  extract,
  graph,
)
from flax.nnx.nnx.module import GraphDef
from flax.nnx.nnx.state import State

A = tp.TypeVar('A')
F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])

class Missing:
  pass


MISSING = Missing()

# -------------------------------
# (split|merge)_inputs
# -------------------------------


class ArgState(extract.ExtractionIndex, extract.ExtractableStates):
  _graphdef: GraphDef[tp.Any] = struct.field(pytree_node=False)
  state: State = struct.field(pytree_node=True)

  @property
  def graphdef(self) -> GraphDef[tp.Any]:
    return self._graphdef

  @property
  def states(self) -> tp.Iterable[State]:
    yield self.state

@tp.overload
def split_inputs(
  *,
  ctx_tag: str = 'split_merge_inputs',
) -> tp.Callable[[F], F]: ...
@tp.overload
def split_inputs(
  f: F,
  *,
  ctx_tag: str = 'split_merge_inputs',
) -> F: ...
def split_inputs(
  f: F | Missing = MISSING,
  *,
  ctx_tag: str = 'split_merge_inputs',
) -> F | tp.Callable[[F], F]:
  if isinstance(f, Missing):
    return functools.partial(split_inputs, ctx_tag=ctx_tag)

  @graph.update_context(ctx_tag)
  @functools.wraps(f)
  def split_inputs_wrapper(*args):
    ctx = graph.current_update_context(ctx_tag)
    args, input_graph_nodes = extract.extract_graph_nodes(args)
    graphdef, states = ctx.split(input_graph_nodes)
    args = extract.replace_indexes(
      args,
      lambda x: ArgState(
        x.index,
        graphdef,
        states[x.index],  # type: ignore
      ),
    )
    args_out, out = f(*args)
    arg_states_out = extract.extract_indexes((args_out, out), types=ArgState)

    if arg_states_out:
      graphdef_out, states_out = extract.merge_extractable_states(
        arg_states_out
      )
      output_nodes = ctx.merge(graphdef_out, states_out)
      out = extract.insert_graph_nodes(out, output_nodes)

    return out

  return split_inputs_wrapper  # type: ignore

@tp.overload
def merge_inputs(
  *,
  ctx_tag: str = 'split_merge_inputs',
) -> tp.Callable[[F], F]: ...
@tp.overload
def merge_inputs(
  f: F,
  *,
  ctx_tag: str = 'split_merge_inputs',
) -> F: ...
def merge_inputs(
  f: F | Missing = MISSING,
  *,
  ctx_tag: str = 'split_merge_inputs',
) -> F | tp.Callable[[F], F]:
  if isinstance(f, Missing):
    return functools.partial(merge_inputs, ctx_tag=ctx_tag)

  @functools.wraps(f)
  def merge_inputs_wrapper(*args):
    ctx = graph.current_update_context(ctx_tag)
    arg_states = extract.extract_indexes(args, types=ArgState)

    if arg_states:
      graphdef, states = extract.merge_extractable_states(arg_states)
      inputs_graph_nodes = ctx.merge(graphdef, states)
      args = extract.insert_graph_nodes(args, inputs_graph_nodes)

    out = f(*args)

    (args_out, out), output_graph_nodes = extract.extract_graph_nodes(
      (args, out)
    )

    graphdef_out, states_out = ctx.split(output_graph_nodes)

    def replace_index(x: extract.Extractable):
      return ArgState(
        x.index,
        graphdef_out,
        states_out[x.index],  # type: ignore
      )

    out = extract.replace_indexes(out, replace_index)
    args_out = extract.replace_indexes(args_out, replace_index, clear=True)

    return args_out, out

  return merge_inputs_wrapper  # type: ignore
