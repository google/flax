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

from functools import partial
import jax
import pytest

from flax.experimental import nnx
from flax import struct


class TestGraphUtils:
  def test_flatten(self):
    a = {'a': 1, 'b': nnx.Param(2)}
    g = [a, 3, a, nnx.Param(4)]

    graphdef, state, refmap = nnx.graph.flatten(g)
    assert refmap is not None

    state[0]['b'].raw_value = 2
    state[3].raw_value = 4

    assert len(refmap) == 2
    assert a['b'] in refmap
    assert g[3] in refmap

  def test_unflatten(self):
    a = nnx.Dict(a=1, b=nnx.Param(2))
    g = nnx.List([a, 3, a, nnx.Param(4)])

    graphdef, state = nnx.split(g)
    g = nnx.merge(graphdef, state)

    assert g[0] is g[2]

  def test_unflatten_pytree(self):
    a = {'a': 1, 'b': nnx.Param(2)}
    g = [a, 3, a, nnx.Param(4)]

    graphdef, state = nnx.split(g)
    g = nnx.merge(graphdef, state)

    assert g[0] is not g[2]

  def test_unflatten_empty(self):
    a = nnx.Dict({'a': 1, 'b': nnx.Param(2)})
    g = nnx.List([a, 3, a, nnx.Param(4)])

    graphdef, state = nnx.split(g)

    with pytest.raises(
      ValueError, match='Expected key for Variable but was not found in state'
    ):
      nnx.graph.unflatten(graphdef, nnx.State({}))

  def test_update_dynamic(self):
    a = {'a': 1, 'b': nnx.Param(2)}
    g = [a, 3, a, nnx.Param(4)]

    graphdef, state = nnx.split(g)

    state[0]['b'].value = 3
    nnx.graph.update(g, state)

    assert g[0]['b'].value == 3
    assert g[2]['b'].value == 3

  def test_update_static(self):
    a = nnx.Dict({'a': 1, 'b': nnx.Param(2)})
    g = nnx.List([a, 3, a, nnx.Param(4)])

    g2 = nnx.graph.clone(g)
    g2[0]['a'] = 5

    nnx.graph.graph_update_static(g, g2)

    assert g[0]['a'] == 5
    assert g[2]['a'] == 5

  def test_update_static_inconsistent_types(self):
    a = {'a': 1, 'b': nnx.Param(2)}
    g = [a, 3, a, nnx.Param(4)]
    g2 = [a, a, 3, nnx.Param(4)]

    with pytest.raises(
      ValueError, match='Trying to update a node with a different type'
    ):
      nnx.graph.graph_update_static(g, g2)

  def test_update_static_add_new(self):
    a = nnx.Dict({'a': 1, 'b': nnx.Param(2)})
    b = nnx.List([5, 6])
    g = nnx.List([a, 3, a, nnx.Param(4)])
    g2 = nnx.List([a, 3, a, nnx.Param(4), b])

    nnx.graph.graph_update_static(g, g2)

    assert g[4][0] == 5
    assert g[4][1] == 6

  def test_update_static_add_shared_error(self):
    a = nnx.Dict({'a': 1, 'b': nnx.Param(2)})
    g = nnx.List([a, 3, a, nnx.Param(4)])
    g2 = nnx.List([a, 3, a, nnx.Param(4), a])

    with pytest.raises(ValueError, match='Trying to add a new node at path'):
      nnx.graph.graph_update_static(g, g2)

  def test_module_list(self):
    rngs = nnx.Rngs(0)
    ls = [
      nnx.Linear(2, 2, rngs=rngs),
      nnx.BatchNorm(2, rngs=rngs),
    ]

    graphdef, state = nnx.split(ls)

    assert state[0]['kernel'].value.shape == (2, 2)
    assert state[0]['bias'].value.shape == (2,)
    assert state[1]['scale'].value.shape == (2,)
    assert state[1]['bias'].value.shape == (2,)
    assert state[1]['mean'].value.shape == (2,)
    assert state[1]['var'].value.shape == (2,)

  def test_shared_variables(self):
    v = nnx.Param(1)
    g = [v, v]

    graphdef, state = nnx.split(g)

    assert len(state.flat_state()) == 1

    g2 = nnx.merge(graphdef, state)

    assert g2[0] is g2[1]

  def test_tied_weights(self):
    class Foo(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.bar = nnx.Linear(2, 2, rngs=rngs)
        self.baz = nnx.Linear(2, 2, rngs=rngs)

        # tie the weights
        self.baz.kernel = self.bar.kernel

    node = Foo(rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(node)

    assert len(state.flat_state()) == 3  # 2 bias + 1 kernel

    node2 = nnx.merge(graphdef, state)

    assert node2.bar.kernel is node2.baz.kernel

  def test_tied_weights_example(self):
    class LinearTranspose(nnx.Module):
      def __init__(self, dout: int, din: int, *, rngs: nnx.Rngs) -> None:
        self.kernel = nnx.Param(
          nnx.initializers.lecun_normal()(rngs(), (dout, din))
        )

      def __call__(self, x):
        return x @ self.kernel.value.T

    class Encoder(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.embed = nnx.Embed(10, 2, rngs=rngs)
        ...
        self.linear_out = LinearTranspose(10, 2, rngs=rngs)

        # tie the weights
        self.linear_out.kernel = self.embed.embedding

      def __call__(self, x):
        x = self.embed(x)
        ...
        return self.linear_out(x)

    model = Encoder(rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(model)

    assert len(state.flat_state()) == 1

    x = jax.random.randint(jax.random.key(0), (2,), 0, 10)
    y = model(x)

    assert y.shape == (2, 10)

  def test_state_variables_not_shared_with_graph(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.Param(1)

    m = Foo()
    graphdef, state = nnx.split(m)

    assert isinstance(m.a, nnx.Param)
    assert issubclass(state.a.type, nnx.Param)
    assert m.a is not state.a
    assert m.a.value == state.a.value

    m2 = nnx.merge(graphdef, state)

    assert isinstance(m2.a, nnx.Param)
    assert issubclass(state.a.type, nnx.Param)
    assert m2.a is not state.a
    assert m2.a.value == state.a.value

  def test_shared_state_variables_not_shared_with_graph(self):
    class Foo(nnx.Module):
      def __init__(self):
        p = nnx.Param(1)
        self.a = p
        self.b = p

    m = Foo()
    graphdef, state = nnx.split(m)

    assert isinstance(m.a, nnx.Param)
    assert isinstance(m.b, nnx.Param)
    assert issubclass(state.a.type, nnx.Param)
    assert 'b' not in state
    assert m.a is not state.a
    assert m.b is not state.a
    assert m.a.value == state.a.value
    assert m.b.value == state.a.value

    m2 = nnx.merge(graphdef, state)

    assert isinstance(m2.a, nnx.Param)
    assert isinstance(m2.b, nnx.Param)
    assert issubclass(state.a.type, nnx.Param)
    assert m2.a is not state.a
    assert m2.b is not state.a
    assert m2.a.value == state.a.value
    assert m2.b.value == state.a.value
    assert m2.a is m2.b

  def test_pytree_flatten(self):
    @struct.dataclass
    class Tree:
      a: int
      b: str = struct.field(pytree_node=False)

    p = Tree(1, 'a')

    leaves, treedef = nnx.graph._flatten_pytree(p)
    fields = dict(leaves)

    assert 'a' in fields
    assert 'b' not in fields
    assert fields['a'] == 1

    p2 = nnx.graph._unflatten_pytree(leaves, treedef)

    assert isinstance(p2, Tree)
    assert p2.a == 1

  def test_pytree_node(self):
    @struct.dataclass
    class Tree:
      a: nnx.Param[int]
      b: str = struct.field(pytree_node=False)

    class Foo(nnx.Module):
      def __init__(self):
        self.tree = Tree(nnx.Param(1), 'a')

    m = Foo()

    graphdef, state = nnx.split(m)

    assert 'tree' in state
    assert 'a' in state.tree
    assert graphdef.nodedef.subgraphs['tree'].type is nnx.graph.PytreeType

    m2 = nnx.merge(graphdef, state)

    assert isinstance(m2.tree, Tree)
    assert m2.tree.a.raw_value == 1
    assert m2.tree.b == 'a'
    assert m2.tree.a is not m.tree.a
    assert m2.tree is not m.tree

  def test_cached_unflatten(self):
    class Foo(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.a = nnx.Linear(2, 2, rngs=rngs)
        self.b = nnx.BatchNorm(2, rngs=rngs)

    def f(m: Foo):
      m.a, m.b = m.b, m.a

    m = Foo(rngs=nnx.Rngs(0))
    a = m.a
    b = m.b

    graphdef: nnx.graph.GraphDef[Foo]
    graphdef, state, ref_out_idx_out = nnx.graph.flatten(m)

    @partial(jax.jit, static_argnums=(0,))
    def f_pure(graphdef: nnx.graph.GraphDef[Foo], state):
      m, idx_out_ref_in = nnx.graph.unflatten(graphdef, state)
      f(m)
      graphdef, state, ref_in_idx_in = nnx.graph.flatten(m)
      idx_out_idx_in = nnx.graph.compose_mapping(idx_out_ref_in, ref_in_idx_in)
      static_out = nnx.graph.Static((graphdef, idx_out_idx_in))
      return state, static_out

    static_out: nnx.graph.Static
    state, static_out = f_pure(graphdef, state)
    idx_out_idx_in: dict[int, int]
    graphdef, idx_out_idx_in = static_out.value
    idx_in_ref_out = nnx.graph.compose_mapping_reversed(
      ref_out_idx_out, idx_out_idx_in
    )
    m2, _ = nnx.graph.unflatten(graphdef, state, idxmap=idx_in_ref_out)
    assert m2 is m
    assert m2.a is b
    assert m2.b is a

  def test_cached_unflatten_swap_variables(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.Param(1)
        self.b = nnx.Param(2)

    def f(m: Foo):
      m.a, m.b = m.b, m.a

    m = Foo()
    a = m.a
    b = m.b

    graphdef: nnx.graph.GraphDef[Foo]
    graphdef, state, ref_out_idx_out = nnx.graph.flatten(m)

    @partial(jax.jit, static_argnums=(0,))
    def f_pure(graphdef: nnx.graph.GraphDef[Foo], state):
      m, idx_out_ref_in = nnx.graph.unflatten(graphdef, state)
      f(m)
      graphdef, state, ref_in_idx_in = nnx.graph.flatten(m)
      idx_out_idx_in = nnx.graph.compose_mapping(idx_out_ref_in, ref_in_idx_in)
      static_out = nnx.graph.Static((graphdef, idx_out_idx_in))
      return state, static_out

    static_out: nnx.graph.Static
    state, static_out = f_pure(graphdef, state)
    idx_out_idx_in: dict[int, int]
    graphdef, idx_out_idx_in = static_out.value
    idx_in_ref_out = nnx.graph.compose_mapping_reversed(
      ref_out_idx_out, idx_out_idx_in
    )
    m2, _ = nnx.graph.unflatten(graphdef, state, idxmap=idx_in_ref_out)
    assert m2 is m
    assert m2.a is b
    assert m2.b is a

  def test_cached_unflatten_add_self_reference(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.ref = None

    def f(m: Foo):
      m.ref = m

    m = Foo()

    graphdef: nnx.graph.GraphDef[Foo]
    graphdef, state, ref_out_idx_out = nnx.graph.flatten(m)

    @partial(jax.jit, static_argnums=(0,))
    def f_pure(graphdef: nnx.graph.GraphDef[Foo], state):
      m, idx_out_ref_in = nnx.graph.unflatten(graphdef, state)
      f(m)
      graphdef, state, ref_in_idx_in = nnx.graph.flatten(m)
      idx_out_idx_in = nnx.graph.compose_mapping(idx_out_ref_in, ref_in_idx_in)
      static_out = nnx.graph.Static((graphdef, idx_out_idx_in))
      return state, static_out

    static_out: nnx.graph.Static
    state, static_out = f_pure(graphdef, state)
    idx_out_idx_in: dict[int, int]
    graphdef, idx_out_idx_in = static_out.value
    idx_in_ref_out = nnx.graph.compose_mapping_reversed(
      ref_out_idx_out, idx_out_idx_in
    )
    m2, _ = nnx.graph.unflatten(graphdef, state, idxmap=idx_in_ref_out)
    assert m2 is m
    assert m2.ref is m2
