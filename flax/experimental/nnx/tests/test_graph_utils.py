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

    refmap, static, state = nnx.full_split(g)
    assert refmap is not None

    state[0]['b'].raw_value = 2
    state[3].raw_value = 4

    assert len(refmap) == 2
    assert a['b'] in refmap
    assert g[3] in refmap

  def test_unflatten(self):
    a = nnx.Dict(a=1, b=nnx.Param(2))
    g = nnx.List([a, 3, a, nnx.Param(4)])

    static, state = nnx.split(g)
    g = nnx.merge(static, state)

    assert g[0] is g[2]

  def test_unflatten_pytree(self):
    a = {'a': 1, 'b': nnx.Param(2)}
    g = [a, 3, a, nnx.Param(4)]

    static, state = nnx.split(g)
    g = nnx.merge(static, state)

    assert g[0] is not g[2]

  def test_unflatten_empty(self):
    a = nnx.Dict({'a': 1, 'b': nnx.Param(2)})
    g = nnx.List([a, 3, a, nnx.Param(4)])

    static, state = nnx.split(g)
    g = static.merge(nnx.State({}))

    assert g[0] is g[2]
    assert g[0]['b'].raw_value is nnx.EMPTY
    assert g[3].raw_value is nnx.EMPTY

  def test_update_dynamic(self):
    a = {'a': 1, 'b': nnx.Param(2)}
    g = [a, 3, a, nnx.Param(4)]

    static, state = nnx.split(g)

    state[0]['b'].raw_value = 3
    nnx.graph_utils.update(g, state)

    assert g[0]['b'].raw_value == 3
    assert g[2]['b'].raw_value == 3

  def test_update_static(self):
    a = nnx.Dict({'a': 1, 'b': nnx.Param(2)})
    g = nnx.List([a, 3, a, nnx.Param(4)])

    g2 = nnx.graph_utils.clone(g)
    g2[0]['a'] = 5

    nnx.graph_utils.graph_update_static(g, g2)

    assert g[0]['a'] == 5
    assert g[2]['a'] == 5

  def test_update_static_inconsistent_types(self):
    a = {'a': 1, 'b': nnx.Param(2)}
    g = [a, 3, a, nnx.Param(4)]
    g2 = [a, a, 3, nnx.Param(4)]

    with pytest.raises(
      ValueError, match='Trying to update a node with a different type'
    ):
      nnx.graph_utils.graph_update_static(g, g2)

  def test_update_static_add_new(self):
    a = nnx.Dict({'a': 1, 'b': nnx.Param(2)})
    b = nnx.List([5, 6])
    g = nnx.List([a, 3, a, nnx.Param(4)])
    g2 = nnx.List([a, 3, a, nnx.Param(4), b])

    nnx.graph_utils.graph_update_static(g, g2)

    assert g[4][0] == 5
    assert g[4][1] == 6

  def test_update_static_add_shared_error(self):
    a = nnx.Dict({'a': 1, 'b': nnx.Param(2)})
    g = nnx.List([a, 3, a, nnx.Param(4)])
    g2 = nnx.List([a, 3, a, nnx.Param(4), a])

    with pytest.raises(ValueError, match='Trying to add a new node at path'):
      nnx.graph_utils.graph_update_static(g, g2)

  def test_module_list(self):
    rngs = nnx.Rngs(0)
    ls = [
      nnx.Linear(2, 2, rngs=rngs),
      nnx.BatchNorm(2, rngs=rngs),
    ]

    static, state = nnx.split(ls)

    assert state[0]['kernel'].raw_value.shape == (2, 2)
    assert state[0]['bias'].raw_value.shape == (2,)
    assert state[1]['scale'].raw_value.shape == (2,)
    assert state[1]['bias'].raw_value.shape == (2,)
    assert state[1]['mean'].raw_value.shape == (2,)
    assert state[1]['var'].raw_value.shape == (2,)

  def test_shared_variables(self):
    v = nnx.Param(1)
    g = [v, v]

    static, state = nnx.split(g)

    assert len(state.flat_state()) == 1

    g2 = static.merge(state)

    assert g2[0] is g2[1]

  def test_tied_weights(self):
    class Foo(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.bar = nnx.Linear(2, 2, rngs=rngs)
        self.baz = nnx.Linear(2, 2, rngs=rngs)

        # tie the weights
        self.baz.kernel = self.bar.kernel

    node = Foo(rngs=nnx.Rngs(0))
    static, state = nnx.split(node)

    assert len(state.flat_state()) == 3  # 2 bias + 1 kernel

    node2 = static.merge(state)

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
    static, state = model.split()

    assert len(state.flat_state()) == 1

    x = jax.random.randint(jax.random.key(0), (2,), 0, 10)
    y = model(x)

    assert y.shape == (2, 10)

  def test_state_variables_not_shared_with_graph(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.Param(1)

    m = Foo()
    static, state = m.split()

    assert isinstance(m.a, nnx.Param)
    assert isinstance(state.a, nnx.Param)
    assert m.a is not state.a
    assert m.a.value == state.a.raw_value

    m2 = static.merge(state)

    assert isinstance(m2.a, nnx.Param)
    assert isinstance(state.a, nnx.Param)
    assert m2.a is not state.a
    assert m2.a.value == state.a.raw_value

  def test_shared_state_variables_not_shared_with_graph(self):
    class Foo(nnx.Module):
      def __init__(self):
        p = nnx.Param(1)
        self.a = p
        self.b = p

    m = Foo()
    static, state = m.split()

    assert isinstance(m.a, nnx.Param)
    assert isinstance(m.b, nnx.Param)
    assert isinstance(state.a, nnx.Param)
    assert 'b' not in state
    assert m.a is not state.a
    assert m.b is not state.a
    assert m.a.value == state.a.raw_value
    assert m.b.value == state.a.raw_value

    m2 = static.merge(state)

    assert isinstance(m2.a, nnx.Param)
    assert isinstance(m2.b, nnx.Param)
    assert isinstance(state.a, nnx.Param)
    assert m2.a is not state.a
    assert m2.b is not state.a
    assert m2.a.value == state.a.raw_value
    assert m2.b.value == state.a.raw_value
    assert m2.a is m2.b

  def test_pytree_flatten(self):
    @struct.dataclass
    class Tree:
      a: int
      b: str = struct.field(pytree_node=False)

    p = Tree(1, 'a')

    leaves, treedef = nnx.graph_utils._flatten_pytree(p)
    fields = dict(leaves)

    assert 'a' in fields
    assert 'b' not in fields
    assert fields['a'] == 1

    p2 = nnx.graph_utils._unflatten_pytree(leaves, treedef)

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

    static, state = m.split()

    assert 'tree' in state
    assert 'a' in state.tree
    assert static.nodedef.subgraphs['tree'].type is nnx.graph_utils.PytreeType

    m2 = static.merge(state)

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

    static: nnx.graph_utils.GraphDef[Foo]
    static, state, ref_out_idx_out = nnx.graph_utils.graph_flatten(m)

    @partial(jax.jit, static_argnums=(0,))
    def f_pure(static: nnx.graph_utils.GraphDef[Foo], state):
      m, idx_out_ref_in = nnx.graph_utils.graph_unflatten(static, state)
      f(m)
      static, state, ref_in_idx_in = nnx.graph_utils.graph_flatten(m)
      idx_out_idx_in = nnx.graph_utils.compose_mapping(
        idx_out_ref_in, ref_in_idx_in
      )
      static_out = nnx.graph_utils.Static((static, idx_out_idx_in))
      return state, static_out

    static_out: nnx.graph_utils.Static
    state, static_out = f_pure(static, state)
    idx_out_idx_in: dict[int, int]
    static, idx_out_idx_in = static_out.value
    idx_in_ref_out = nnx.graph_utils.compose_mapping_reversed(
      ref_out_idx_out, idx_out_idx_in
    )
    m2, _ = nnx.graph_utils.graph_unflatten(
      static, state, idxmap=idx_in_ref_out
    )
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

    static: nnx.graph_utils.GraphDef[Foo]
    static, state, ref_out_idx_out = nnx.graph_utils.graph_flatten(m)

    @partial(jax.jit, static_argnums=(0,))
    def f_pure(static: nnx.graph_utils.GraphDef[Foo], state):
      m, idx_out_ref_in = nnx.graph_utils.graph_unflatten(static, state)
      f(m)
      static, state, ref_in_idx_in = nnx.graph_utils.graph_flatten(m)
      idx_out_idx_in = nnx.graph_utils.compose_mapping(
        idx_out_ref_in, ref_in_idx_in
      )
      static_out = nnx.graph_utils.Static((static, idx_out_idx_in))
      return state, static_out

    static_out: nnx.graph_utils.Static
    state, static_out = f_pure(static, state)
    idx_out_idx_in: dict[int, int]
    static, idx_out_idx_in = static_out.value
    idx_in_ref_out = nnx.graph_utils.compose_mapping_reversed(
      ref_out_idx_out, idx_out_idx_in
    )
    m2, _ = nnx.graph_utils.graph_unflatten(
      static, state, idxmap=idx_in_ref_out
    )
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

    static: nnx.graph_utils.GraphDef[Foo]
    static, state, ref_out_idx_out = nnx.graph_utils.graph_flatten(m)

    @partial(jax.jit, static_argnums=(0,))
    def f_pure(static: nnx.graph_utils.GraphDef[Foo], state):
      m, idx_out_ref_in = nnx.graph_utils.graph_unflatten(static, state)
      f(m)
      static, state, ref_in_idx_in = nnx.graph_utils.graph_flatten(m)
      idx_out_idx_in = nnx.graph_utils.compose_mapping(
        idx_out_ref_in, ref_in_idx_in
      )
      static_out = nnx.graph_utils.Static((static, idx_out_idx_in))
      return state, static_out

    static_out: nnx.graph_utils.Static
    state, static_out = f_pure(static, state)
    idx_out_idx_in: dict[int, int]
    static, idx_out_idx_in = static_out.value
    idx_in_ref_out = nnx.graph_utils.compose_mapping_reversed(
      ref_out_idx_out, idx_out_idx_in
    )
    m2, _ = nnx.graph_utils.graph_unflatten(
      static, state, idxmap=idx_in_ref_out
    )
    assert m2 is m
    assert m2.ref is m2

  def test_init_rngs(self):
    class Linear(nnx.Module):
      def __init__(self, din: int, dout: int):
        self.kernel = nnx.Param(
          nnx.empty((din, dout)), initializer=nnx.initializers.lecun_normal()
        )
        self.bias = nnx.Param(
          nnx.empty((dout,)), initializer=nnx.initializers.zeros
        )

      def __call__(self, x):
        return x @ self.kernel.value + self.bias.value

    m = Linear(2, 2)
    nnx.init(m, nnx.Rngs(0))
    assert isinstance(m.kernel.value, jax.Array)
    assert isinstance(m.bias.value, jax.Array)

  def test_init_seed(self):
    class Linear(nnx.Module):
      def __init__(self, din: int, dout: int):
        self.kernel = nnx.Param(
          nnx.empty((din, dout)), initializer=nnx.initializers.lecun_normal()
        )
        self.bias = nnx.Param(
          nnx.empty((dout,)), initializer=nnx.initializers.zeros
        )

      def __call__(self, x):
        return x @ self.kernel.value + self.bias.value

    m = Linear(2, 2)
    nnx.init(m, 0)
    assert isinstance(m.kernel.value, jax.Array)
    assert isinstance(m.bias.value, jax.Array)

  def test_init_key(self):
    class Linear(nnx.Module):
      def __init__(self, din: int, dout: int):
        self.kernel = nnx.Param(
          nnx.empty((din, dout)), initializer=nnx.initializers.lecun_normal()
        )
        self.bias = nnx.Param(
          nnx.empty((dout,)), initializer=nnx.initializers.zeros
        )

      def __call__(self, x):
        return x @ self.kernel.value + self.bias.value

    m = Linear(2, 2)
    nnx.init(m, jax.random.key(0))
    assert isinstance(m.kernel.value, jax.Array)
    assert isinstance(m.bias.value, jax.Array)
