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

import dataclasses
from functools import partial
from threading import Thread
from typing import Any

from absl.testing import absltest, parameterized
import numpy as np
from flax import linen, nnx, struct
import jax
import jax.numpy as jnp


class StatefulLinear(nnx.Module):
  def __init__(self, din, dout, rngs):
    self.w = nnx.Param(jax.random.uniform(rngs(), (din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))
    self.count = nnx.Variable(jnp.array(0, dtype=jnp.uint32))

  def increment(self):
    self.count[...] += 1

  def __call__(self, x):
    self.count[...] += 1
    return x @ self.w + self.b[None]


class TestGraphUtils(absltest.TestCase):
  def test_flatten(self):
    a = {'a': 1, 'b': nnx.Param(2)}
    g = [a, 3, a, nnx.Param(4)]

    refmap = nnx.graph.RefMap()
    graphdef, flat_state = nnx.graph.flatten(g, ref_index=refmap)

    assert flat_state[0][1].get_value() == 2
    assert flat_state[1][1].get_value() == 4

    assert len(refmap) == 2  # 2 Variables
    assert a['b'] in refmap
    assert g[3] in refmap

  def test_flatten_no_paths(self):
    a = {'a': 1, 'b': nnx.Param(jnp.array(2))}
    g = [a, 3, a, nnx.Param(jnp.array(4))]

    refmap = nnx.graph.RefMap()
    graphdef, flat_state = nnx.graph.flatten(
      g, ref_index=refmap, with_paths=False
    )

    assert flat_state[0][...] == 2
    assert flat_state[1][...] == 4

    assert len(refmap) == 2  # 2 Variables
    assert a['b'] in refmap
    assert g[3] in refmap

  def test_unflatten(self):
    a = nnx.Dict(a=1, b=nnx.Param(2))
    g = nnx.List([a, 3, a, nnx.Param(4)])

    graphdef, state = nnx.split(g)
    g = nnx.merge(graphdef, state)

    assert g[0] is g[2]

  def test_flatten_unflatten_unkown_leaves(self):
    x = jnp.array(1.0)
    graphdef, flat_state = nnx.graph.flatten(x)

    self.assertIs(flat_state[0][1], x)

    x1 = nnx.merge(graphdef, flat_state)
    self.assertIs(x1, x)

  def test_split_merge_unkown_leaves(self):
    x = jnp.array(1.0)
    graphdef, state = nnx.graph.split(x)

    self.assertIs(state, x)

    x1 = nnx.merge(graphdef, state)
    self.assertIs(x1, x)

  def test_split_merge_unkown_leaves_with_filters(self):
    x = jnp.array(1.0)
    graphdef, state, rest = nnx.graph.split(x, jax.Array, ...)

    self.assertIs(state, x)

    x1 = nnx.merge(graphdef, state, rest)
    self.assertIs(x1, x)

  def test_unflatten_pure_dict(self):
    a = nnx.Dict(a=1, b=nnx.Param(2))
    g = nnx.List([a, 3, a, nnx.Param(4)])

    graphdef, state = nnx.split(g)
    pure_state = nnx.to_pure_dict(state)

    g = nnx.merge(graphdef, pure_state)

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

    with self.assertRaisesRegex(ValueError, 'Incorrect number of leaves'):
      nnx.graph.unflatten(graphdef, nnx.State({}))

  def test_unflatten_return_variables(self):
    a = nnx.Dict({'a': 1, 'b': nnx.Param(2)})
    g = nnx.List([a, 3, a, nnx.Param(4)])

    graphdef, state = nnx.graph.flatten(
      g, with_paths=True
    )

    self.assertLen(state, 2)
    self.assertIsInstance(state, nnx.graph.FlatState)
    self.assertIsInstance(state[0][1], nnx.Param)
    self.assertIsInstance(state[1][1], nnx.Param)

  def test_update_dynamic(self):
    a = {'a': 1, 'b': nnx.Param(jnp.array(2))}
    g = [a, 3, a, nnx.Param(jnp.array(4))]

    graphdef, state = nnx.split(g)

    state[0]['b'][...] = 3
    nnx.update(g, state)

    assert g[0]['b'][...] == 3
    assert g[2]['b'][...] == 3

  def test_update_from_pure_dict(self):
    a = {'a': 1, 'b': nnx.Param(jnp.array(2))}
    g = [a, 3, a, nnx.Param(jnp.array(4))]

    graphdef, state = nnx.split(g)
    pure_state = nnx.to_pure_dict(state)

    pure_state[0]['b'] = jnp.array(3)
    nnx.update(g, pure_state)

    assert g[0]['b'][...] == 3
    assert g[2]['b'][...] == 3

  def test_module_list(self):
    rngs = nnx.Rngs(0)
    ls = [
      nnx.Linear(2, 2, rngs=rngs),
      nnx.BatchNorm(2, rngs=rngs),
    ]

    graphdef, state = nnx.split(ls)

    assert state[0]['kernel'].shape == (2, 2)
    assert state[0]['bias'].shape == (2,)
    assert state[1]['scale'].shape == (2,)
    assert state[1]['bias'].shape == (2,)
    assert state[1]['mean'].shape == (2,)
    assert state[1]['var'].shape == (2,)

  def test_shared_variables(self):
    v = nnx.Param(1)
    g = [v, v]

    graphdef, state = nnx.split(g)

    assert len(nnx.to_flat_state(state)) == 1

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

    assert len(nnx.to_flat_state(state)) == 3  # 2 bias + 1 kernel

    node2 = nnx.merge(graphdef, state)

    assert node2.bar.kernel is node2.baz.kernel

  def test_tied_weights_example(self):
    class LinearTranspose(nnx.Module):
      def __init__(self, dout: int, din: int, *, rngs: nnx.Rngs) -> None:
        self.kernel = nnx.Param(
          nnx.initializers.lecun_normal()(rngs(), (dout, din))
        )

      def __call__(self, x):
        return x @ self.kernel.T

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

    assert len(nnx.to_flat_state(state)) == 1

    x = jax.random.randint(jax.random.key(0), (2,), 0, 10)
    y = model(x)

    assert y.shape == (2, 10)

  def test_state_variables_shared_with_graph(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.Param(jnp.array(1))

    m = Foo()
    graphdef, state = nnx.split(m)

    assert isinstance(m.a, nnx.Param)
    assert isinstance(state['a'], nnx.Param)
    assert m.a is state['a']
    assert m.a[...] == state['a'][...]

    m2 = nnx.merge(graphdef, state)

    assert isinstance(m2.a, nnx.Param)
    assert isinstance(state['a'], nnx.Param)
    assert m2.a is state['a']
    assert m2.a[...] == state['a'][...]

  def test_shared_state_variables_shared_with_graph(self):
    class Foo(nnx.Module):
      def __init__(self):
        p = nnx.Param(jnp.array(1))
        self.a = p
        self.b = p

    m = Foo()
    graphdef, state = nnx.split(m)

    assert isinstance(m.a, nnx.Param)
    assert isinstance(m.b, nnx.Param)
    assert isinstance(state['a'], nnx.Param)
    assert 'b' not in state
    assert m.a is state['a']
    assert m.b is state['a']
    assert m.a[...] == state['a'][...]
    assert m.b[...] == state['a'][...]

    m2 = nnx.merge(graphdef, state)

    assert isinstance(m2.a, nnx.Param)
    assert isinstance(m2.b, nnx.Param)
    assert isinstance(state['a'], nnx.Param)
    assert m2.a is state['a']
    assert m2.b is state['a']
    assert m2.a[...] == state['a'][...]
    assert m2.b[...] == state['a'][...]
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
        self.tree = nnx.data(Tree(nnx.Param(1), 'a'))

    m = Foo()

    graphdef, state = nnx.split(m)

    assert 'tree' in state
    assert 'a' in state['tree']

    m2 = nnx.merge(graphdef, state)

    assert isinstance(m2.tree, Tree)
    assert m2.tree.a.get_value() == 1
    assert m2.tree.b == 'a'
    assert m2.tree.a is m.tree.a
    assert m2.tree is not m.tree

  def test_cached_unflatten(self):
    class Foo(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.a = nnx.Linear(2, 2, rngs=rngs)
        self.b = nnx.BatchNorm(2, rngs=rngs)

    def f(m: Foo):
      m.a, m.b = m.b, m.a  # type: ignore

    m = Foo(rngs=nnx.Rngs(0))
    a = m.a
    b = m.b

    ref_out_idx_out = nnx.graph.RefMap()
    graphdef: nnx.graph.GraphDef[Foo]
    graphdef, state = nnx.graph.flatten(m, ref_index=ref_out_idx_out)
    state = state.to_nested_state()

    @partial(jax.jit, static_argnums=(0,))
    def f_pure(graphdef: nnx.graph.GraphDef[Foo], state):
      idx_out_ref_in = nnx.graph.IndexMap()
      m = nnx.graph.unflatten(graphdef, state, index_ref=idx_out_ref_in)
      ref_in_idx_out = nnx.graph.RefMap.from_indexmap(idx_out_ref_in)
      f(m)
      ref_in_idx_in = nnx.graph.RefMap()
      graphdef, state = nnx.graph.flatten(
        m, ref_index=ref_in_idx_in, ref_outer_index=ref_in_idx_out
      )
      state = state.to_nested_state()
      return state, graphdef

    state, graphdef_out = f_pure(graphdef, state)
    idx_out_ref_out = nnx.graph.IndexMap.from_refmap(ref_out_idx_out)
    m2 = nnx.graph.unflatten(
      graphdef_out, state, outer_index_outer_ref=idx_out_ref_out
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

    ref_out_idx_out = nnx.graph.RefMap()
    graphdef: nnx.graph.GraphDef[Foo]
    graphdef, state = nnx.graph.flatten(m, ref_index=ref_out_idx_out)
    idx_out_ref_out = {v: k for k, v in ref_out_idx_out.items()}
    state = state.to_nested_state()

    @partial(jax.jit, static_argnums=(0,))
    def f_pure(graphdef: nnx.graph.GraphDef[Foo], state):
      idx_out_ref_in = nnx.graph.IndexMap()
      m = nnx.graph.unflatten(graphdef, state, index_ref=idx_out_ref_in)
      ref_in_idx_out = nnx.graph.RefMap.from_indexmap(idx_out_ref_in)
      f(m)
      ref_in_idx_in = nnx.graph.RefMap()
      graphdef, state = nnx.graph.flatten(
        m, ref_index=ref_in_idx_in, ref_outer_index=ref_in_idx_out
      )
      state = state.to_nested_state()
      return state, graphdef

    state, graphdef = f_pure(graphdef, state)
    m2 = nnx.graph.unflatten(
      graphdef, state, outer_index_outer_ref=idx_out_ref_out
    )
    assert m2 is m
    assert m2.a is b
    assert m2.b is a

  def test_cached_unflatten_add_self_reference(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.ref = nnx.data(None)

    def f(m: Foo):
      m.ref = m

    m = Foo()

    ref_out_idx_out = nnx.graph.RefMap()
    graphdef: nnx.graph.GraphDef[Foo]
    graphdef, state = nnx.graph.flatten(m, ref_index=ref_out_idx_out)
    idx_out_ref_out = nnx.graph.IndexMap.from_refmap(ref_out_idx_out)
    state = state.to_nested_state()

    @partial(jax.jit, static_argnums=(0,))
    def f_pure(graphdef: nnx.graph.GraphDef[Foo], state):
      idx_out_ref_in = nnx.graph.IndexMap()
      m = nnx.graph.unflatten(graphdef, state, index_ref=idx_out_ref_in)
      ref_in_idx_out = nnx.graph.RefMap.from_indexmap(idx_out_ref_in)
      f(m)
      ref_in_idx_in = nnx.graph.RefMap()
      graphdef, state = nnx.graph.flatten(
        m, ref_index=ref_in_idx_in, ref_outer_index=ref_in_idx_out
      )
      state = state.to_nested_state()
      return state, graphdef

    state, graphdef_out = f_pure(graphdef, state)
    m2 = nnx.graph.unflatten(
      graphdef_out, state, outer_index_outer_ref=idx_out_ref_out
    )
    assert m2 is m
    assert m2.ref is m2

  def test_call_jit_update(self):
    class Counter(nnx.Module):
      def __init__(self):
        self.count = nnx.Param(jnp.zeros(()))

      def inc(self):
        self.count[...] += 1
        return 1

    graph_state = nnx.split(Counter())

    @jax.jit
    def update(graph_state: nnx.PureState[Counter]):
      out, graph_state = nnx.call(graph_state).inc()
      self.assertEqual(out, 1)
      return graph_state

    graph_state = update(graph_state)
    graph_state = update(graph_state)

    counter = nnx.merge(*graph_state)

    self.assertEqual(counter.count[...], 2)

  def test_stateful_linear(self):
    linear = StatefulLinear(3, 2, nnx.Rngs(0))
    linear_state = nnx.split(linear)

    @jax.jit
    def forward(x, pure_linear: nnx.PureState[StatefulLinear]):
      y, pure_linear = nnx.call(pure_linear)(x)
      return y, pure_linear

    x = jnp.ones((1, 3))
    y, linear_state = forward(x, linear_state)
    y, linear_state = forward(x, linear_state)

    self.assertEqual(linear.count[...], 0)
    new_linear = nnx.merge(*linear_state)
    self.assertEqual(new_linear.count[...], 2)

  def test_getitem(self):
    rngs = nnx.Rngs(0)
    nodes = dict(
      a=StatefulLinear(3, 2, rngs),
      b=StatefulLinear(2, 1, rngs),
    )
    node_state = nnx.split(nodes)
    _, node_state = nnx.call(node_state)['b'].increment()

    nodes = nnx.merge(*node_state)

    self.assertEqual(nodes['a'].count[...], 0)
    self.assertEqual(nodes['b'].count[...], 1)

  def test_object_state_propagation(self):
    test = self

    class Foo(nnx.Module):
      def __call__(self):
        test.assertTrue(self._pytree__state.initializing)
        self = nnx.merge(*nnx.split(self))
        test.assertTrue(self._pytree__state.initializing)

    module = Foo()
    nnx.bridge.lazy_init(module)

  def test_object_state_propagation_nested(self):
    class NNXOuter(nnx.Module):
      def __init__(self, dout: int, rngs: nnx.Rngs):
        self.inner = nnx.bridge.ToNNX(linen.Dense(dout), rngs=rngs)
        self.rngs = rngs

      def __call__(self, x):
        @nnx.split_rngs(splits=5)
        @nnx.vmap(in_axes=(0, None), axis_size=5)
        def vmap_fn(inner, x):
          return inner(x)

        return vmap_fn(self.inner, x)

    x = jax.random.normal(jax.random.key(0), (2, 4))
    model = NNXOuter(3, rngs=nnx.Rngs(0))
    nnx.bridge.lazy_init(model, x)

    self.assertEqual(model.inner.kernel.shape, (5, 4, 3))
    self.assertEqual(model.inner.bias.shape, (5, 3))

  def test_split_merge_context(self):
    m = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    with nnx.graph.split_context() as ctx:
      graphdef1, state1 = ctx.split(m)
      graphdef2, state2 = ctx.split(m)

    self.assertFalse(hasattr(ctx, 'ref_index'))
    self.assertFalse(hasattr(ctx, 'ctxtag'))
    self.assertIsInstance(graphdef1.nodes[0], nnx.graph.NodeDef)
    self.assertIsInstance(graphdef2.nodes[0], nnx.graph.NodeRef)
    self.assertLen(nnx.to_flat_state(state1), 2)
    self.assertLen(nnx.to_flat_state(state2), 0)

    with nnx.graph.merge_context() as ctx:
      m1 = ctx.merge(graphdef1, state1)
      m2 = ctx.merge(graphdef2, state2)

    self.assertIs(m1, m2)
    self.assertFalse(hasattr(ctx, 'index_ref'))
    self.assertFalse(hasattr(ctx, 'ctxtag'))

  def test_split_merge_context_example(self):
    m1 = nnx.Dict({})
    with nnx.update_context('example'):
      with nnx.split_context('example') as ctx:
        graphdef, state = ctx.split(m1)

      @jax.jit
      def f(graphdef, state):
        with nnx.merge_context('example', True) as ctx:
          m2 = ctx.merge(graphdef, state)
        m2.a = 1
        m2.ref = m2  # create a reference cycle
        with nnx.split_context('example') as ctx:
          return ctx.split(m2)

      graphdef_out, state_out = f(graphdef, state)
      with nnx.merge_context('example', False) as ctx:
        m3 = ctx.merge(graphdef_out, state_out)

  def test_split_merge_context_nested(self):
    m2 = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    m1 = nnx.Sequential(m2)
    with nnx.graph.split_context() as ctx:
      graphdef1, state1 = ctx.split(m1)
      graphdef2, state2 = ctx.split(m2)

    self.assertIsInstance(graphdef1.nodes[0], nnx.graph.NodeDef)
    self.assertIsInstance(graphdef2.nodes[0], nnx.graph.NodeRef)
    self.assertLen(nnx.to_flat_state(state1), 2)
    self.assertLen(nnx.to_flat_state(state2), 0)

    with nnx.graph.merge_context() as ctx:
      m1 = ctx.merge(graphdef1, state1)
      m2 = ctx.merge(graphdef2, state2)

    self.assertIs(m2, m1.layers[0])
    self.assertFalse(hasattr(ctx, 'index_ref'))
    self.assertFalse(hasattr(ctx, 'ctxtag'))

  def test_split_merge_update_context(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.Param(1)
        self.b = nnx.data(2)

    m = Foo()
    ctxtag = 'test'

    with nnx.update_context(ctxtag):
      with nnx.graph.split_context(ctxtag) as ctx:
        graphdef1, state1 = ctx.split(m)
        graphdef2, state2 = ctx.split(m)

      self.assertFalse(hasattr(ctx, 'ref_index'))
      self.assertFalse(hasattr(ctx, 'ctxtag'))
      self.assertIsInstance(graphdef1.nodes[0], nnx.graph.NodeDef)
      self.assertIsInstance(graphdef2.nodes[0], nnx.graph.NodeRef)
      self.assertLen(nnx.to_flat_state(state1), 1)
      self.assertLen(nnx.to_flat_state(state2), 0)

      @jax.jit
      def f(graphdef1, state1, graphdef2, state2):
        with nnx.graph.merge_context(ctxtag, True) as ctx:
          m1 = ctx.merge(graphdef1, state1)
          m2 = ctx.merge(graphdef2, state2)

        self.assertIs(m1, m2)
        self.assertFalse(hasattr(ctx, 'index_ref'))
        self.assertFalse(hasattr(ctx, 'ctxtag'))

        # swap a and b
        m1.a, m1.b = m1.b, m1.a

        with nnx.graph.split_context(ctxtag) as ctx:
          graphdef1, state1 = ctx.split(m1)
          graphdef2, state2 = ctx.split(m2)

        return graphdef1, state1, graphdef2, state2

      graphdef1, state1, graphdef2, state2 = f(
        graphdef1, state1, graphdef2, state2
      )

      with nnx.graph.merge_context(ctxtag, False) as ctx:
        m1_out = ctx.merge(graphdef1, state1)
        m2_out = ctx.merge(graphdef2, state2)

      self.assertIs(m, m1_out)
      self.assertIs(m, m2_out)
      self.assertEqual(m.a, 2)
      self.assertEqual(m.b[...], 1)  # type: ignore

      self.assertFalse(hasattr(ctx, 'index_ref'))
      self.assertFalse(hasattr(ctx, 'ctxtag'))

  def test_to_tree_simple(self):
    m = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    impure_tree = (m, 1, {'b': m})

    pure_tree = nnx.to_tree(impure_tree)

    t1 = pure_tree[0]
    t2 = pure_tree[2]['b']

    self.assertEqual(pure_tree[1], 1)
    self.assertIsInstance(t1, nnx.NodeStates)
    assert isinstance(t1, nnx.NodeStates)
    self.assertIsInstance(t2, nnx.NodeStates)
    assert isinstance(t2, nnx.NodeStates)
    self.assertIsInstance(t1.graphdef.nodes[0], nnx.graph.NodeDef)
    self.assertIsInstance(t2.graphdef.nodes[0], nnx.graph.NodeRef)
    self.assertLen(nnx.to_flat_state(t1.states[0]), 2)
    self.assertLen(nnx.to_flat_state(t2.states[0]), 0)

    impure_tree2 = nnx.from_tree(pure_tree)

    m1_out = impure_tree2[0]
    m2_out = impure_tree2[2]['b']

    self.assertIs(m1_out, m2_out)
    self.assertEqual(impure_tree2[1], 1)

  def test_to_tree_update_context(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = nnx.Param(1)
        self.b = nnx.data(2)

    m = Foo()
    impure_tree = (m, 1, {'b': m})
    ctxtag = 'test'

    with nnx.update_context(ctxtag):
      pure_tree = nnx.to_tree(impure_tree, ctxtag=ctxtag)

      t1 = pure_tree[0]
      t2 = pure_tree[2]['b']

      self.assertEqual(pure_tree[1], 1)
      self.assertIsInstance(t1, nnx.NodeStates)
      assert isinstance(t1, nnx.NodeStates)
      self.assertIsInstance(t2, nnx.NodeStates)
      assert isinstance(t2, nnx.NodeStates)
      self.assertIsInstance(t1.graphdef.nodes[0], nnx.graph.NodeDef)
      self.assertIsInstance(t2.graphdef.nodes[0], nnx.graph.NodeRef)
      self.assertLen(nnx.to_flat_state(t1.states[0]), 1)
      self.assertLen(nnx.to_flat_state(t2.states[0]), 0)

      @jax.jit
      def f(pure_tree):
        impure_tree2 = nnx.from_tree(pure_tree, ctxtag=ctxtag, is_inner=True)
        m1_out = impure_tree2[0]
        m2_out = impure_tree2[2]['b']

        self.assertIs(m1_out, m2_out)
        # self.assertEqual(impure_tree2[1], 1)

        # swap a and b
        m1_out.a, m1_out.b = m1_out.b, m1_out.a

        pure_tree2 = nnx.to_tree(impure_tree2, ctxtag=ctxtag)

        t1 = pure_tree2[0]
        t2 = pure_tree2[2]['b']

        # self.assertEqual(pure_tree2[1], 1)
        self.assertIsInstance(t1, nnx.NodeStates)
        assert isinstance(t1, nnx.NodeStates)
        self.assertIsInstance(t2, nnx.NodeStates)
        assert isinstance(t2, nnx.NodeStates)
        self.assertIsInstance(t1.graphdef.nodes[0], nnx.graph.NodeDef)
        self.assertIsInstance(t2.graphdef.nodes[0], nnx.graph.NodeRef)
        self.assertLen(nnx.to_flat_state(t1.states[0]), 1)
        self.assertLen(nnx.to_flat_state(t2.states[0]), 0)

        return pure_tree2

      pure_tree2 = f(pure_tree)

      impure_tree2 = nnx.from_tree(pure_tree2, ctxtag=ctxtag, is_inner=False)

      m1_out = impure_tree2[0]
      m2_out = impure_tree2[2]['b']

      self.assertIs(m, m1_out)
      self.assertIs(m, m2_out)
      self.assertEqual(m.a, 2)
      self.assertEqual(m.b[...], 1)  # type: ignore
      self.assertEqual(impure_tree2[1], 1)

  def test_to_tree_consistent_prefix(self):
    m = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    impure_tree = (m, 1, {'b': m})
    prefix = (0, None, 0)
    pure_tree = nnx.to_tree(impure_tree, prefix=prefix)

    prefix = (0, None, 1)
    with self.assertRaisesRegex(ValueError, 'Inconsistent aliasing detected'):
      nnx.to_tree(impure_tree, prefix=prefix)

  def test_simple_vmap(self):
    @dataclasses.dataclass(frozen=True)
    class StateAxes:
      params: Any
      batch_stats: Any

    class Foo(nnx.Module):
      def __init__(self, a, b):
        self.a = nnx.Param(a)
        self.b = nnx.BatchStat(b)

    ctxtag = 'test'
    with nnx.update_context(ctxtag):
      m1 = Foo(a=jnp.array(0), b=jnp.arange(5))
      m2 = Foo(a=jnp.array(1), b=jnp.array(2))

      args = (m1, m2, {'b': m1})
      m1_axes = StateAxes(None, 0)
      in_axes = (m1_axes, None, {'b': m1_axes})
      jax_in_axes = jax.tree.map(
        lambda x: nnx.NodeStates.from_prefixes((x.params, x.batch_stats))
        if isinstance(x, StateAxes)
        else x,
        in_axes,
      )
      out_axes = 0

      def split_fn(ctx: nnx.SplitContext, path, prefix, x):
        if isinstance(prefix, StateAxes):
          return nnx.NodeStates.from_split(
            *ctx.split(x, nnx.Param, nnx.BatchStat)
          )
        return nnx.NodeStates.from_split(*ctx.split(x))

      pure_args = nnx.to_tree(
        args, ctxtag=ctxtag, prefix=in_axes, split_fn=split_fn
      )

      @partial(jax.vmap, in_axes=jax_in_axes, out_axes=(jax_in_axes, out_axes))
      def f(*pure_args):
        args = nnx.from_tree(pure_args, ctxtag=ctxtag, is_inner=True)

        y = 0

        self.assertIs(args[0], args[2]['b'])
        for path, m in nnx.iter_graph(args):
          if isinstance(m, Foo):
            self.assertEqual(m.a.shape, ())
            self.assertEqual(m.b.shape, ())
            y += m.a + m.b

        args_out = nnx.extract.clear_non_graph_nodes(args)

        pure_args_out, y = nnx.to_tree(
          (args_out, y),
          prefix=(in_axes, out_axes),
          ctxtag=ctxtag,
          split_fn=split_fn,
        )
        return pure_args_out, y

      pure_args_out, y = f(*pure_args)

      args_out, y = nnx.from_tree(
        (pure_args_out, y), ctxtag=ctxtag, is_inner=False
      )

    self.assertEqual(y.shape, (5,))
    self.assertGreater(y.sum(), 5)
    self.assertIs(m1, args_out[0])
    self.assertIs(m1, args_out[2]['b'])
    self.assertIs(m2, args_out[1])

  def test_split_variable(self):
    v = nnx.Param(1)
    graphdef, state = nnx.split(v)

    self.assertIsInstance(graphdef.nodes[0], nnx.graph.VariableDef)
    self.assertIsInstance(state, nnx.Variable)

    v2 = nnx.merge(graphdef, state)
    self.assertIsInstance(v2, nnx.Param)

  def test_split_filter_variable(self):
    v = nnx.Param(1)
    graphdef, batch_stats, params, rest = nnx.split(
      v, nnx.BatchStat, nnx.Param, ...
    )

    self.assertIsInstance(graphdef.nodes[0], nnx.graph.VariableDef)
    self.assertIsInstance(params, nnx.Variable)
    self.assertIsInstance(batch_stats, nnx.State)
    self.assertEmpty(batch_stats)
    self.assertIsInstance(rest, nnx.State)
    self.assertEmpty(rest)

    v2 = nnx.merge(graphdef, batch_stats, params, rest)
    self.assertIsInstance(v2, nnx.Param)

  def test_split_update_variable(self):
    v = nnx.Param(jnp.array(1))
    graphdef, state = nnx.split(v)

    self.assertIsInstance(graphdef.nodes[0], nnx.graph.VariableDef)
    self.assertIsInstance(state, nnx.Variable)

    state[...] = 2
    nnx.update(v, state)

    self.assertEqual(v[...], 2)

  def test_split_update_filter_variable(self):
    v = nnx.Param(jnp.array(1))
    graphdef, batch_stats, params, rest = nnx.split(
      v, nnx.BatchStat, nnx.Param, ...
    )

    self.assertIsInstance(graphdef.nodes[0], nnx.graph.VariableDef)
    self.assertIsInstance(params, nnx.Variable)
    self.assertIsInstance(batch_stats, nnx.State)
    self.assertEmpty(batch_stats)
    self.assertIsInstance(rest, nnx.State)
    self.assertEmpty(rest)

    params[...] = 2
    nnx.update(v, batch_stats, params, rest)

    self.assertEqual(v[...], 2)

  def test_jit_variable(self):
    v = nnx.Param(1)

    @nnx.jit
    def f(v):
      v[...] += 1

    f(v)

    np.testing.assert_allclose(v[...], 2)

  def test_jit_pytree_of_variables(self):
    v1 = nnx.Param(1)
    v2 = nnx.Param(2)
    vs = [v1, v1, v2]

    @nnx.jit
    def f(vs):
      self.assertIs(vs[0], vs[1])
      self.assertIsNot(vs[0], vs[2])
      vs[0][...] += 10

    f(vs)

    self.assertIs(vs[0], vs[1])
    self.assertIsNot(vs[0], vs[2])
    np.testing.assert_allclose(vs[0][...], 11)
    np.testing.assert_allclose(vs[2][...], 2)

  def test_variable_reference_in_module(self):
    class Foo(nnx.Module):
      def __init__(self, var):
        self.var = var

    var = nnx.Param(1)
    foo = Foo(var)

    @nnx.jit
    def increment_var(var, foo):
      self.assertIs(var, foo.var)
      var[...] += 1

    increment_var(var, foo)
    self.assertEqual(foo.var[...], 2)

  def test_variables_example(self):
    def stateful_linear_init(din: int, dout: int, rngs: nnx.Rngs):
      w = nnx.Param(jax.random.normal(rngs(), (din, dout)))
      b = nnx.Param(jnp.zeros((dout,)))
      count = nnx.Variable(jnp.array(0))
      return w, b, count

    rngs = nnx.Rngs(0)
    w, b, count = stateful_linear_init(2, 3, rngs=rngs)

    @nnx.jit
    def stateful_linear(w, b, count, x):
      count[...] += 1
      return x @ w + b[None]

    x = jax.random.normal(rngs(), (1, 2))
    y = stateful_linear(w, b, count, x)
    self.assertEqual(count[...], 1)

    y = stateful_linear(w, b, count, x)
    self.assertEqual(count[...], 2)
    self.assertEqual(y.shape, (1, 3))

  def test_array_attributes(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = jnp.array(1)
        self.b = 'yes'

    m = Foo()

    graphdef, state = nnx.split(m)

    self.assertLen(state, 1)
    self.assertIsInstance(state['a'], jax.Array)

    m2 = nnx.merge(graphdef, state)

    self.assertIsInstance(m2.a, jax.Array)
    self.assertEqual(m2.a, 1)
    self.assertEqual(m2.b, 'yes')

  def test_transform_array_attributes(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.a = jnp.array(1)
        self.b = 'yes'

    m = Foo()

    @nnx.jit
    def f(m):
      m.a += 1
      self.assertEqual(m.b, 'yes')

    f(m)

    self.assertEqual(m.a, 2)

  def test_data_after_init(self):
    test = self
    class Foo(nnx.Module):
      def __init__(self):
        self.ls = []
        self.ls.append(jnp.array(1))

    with self.assertRaisesRegex(
        ValueError, 'Found unexpected Arrays on value of type'
    ):
      m = Foo()

  def test_update_dict(self):
    node = {
      'a': {
        'b': 1,
        'c': nnx.Param(jnp.array(2)),
        'd': 3,
      },
    }

    updates = {
      'a': {
        'b': 4,
        'c': jnp.array(10),
      },
    }

    nnx.update(node, updates)

    self.assertEqual(node['a']['b'], 4)
    self.assertEqual(node['a']['c'][...], 10)
    self.assertEqual(node['a']['d'], 3)

  def test_pop_dict(self):
    node = {
      'a': {
        'b': jnp.array(1),
        'c': nnx.Param(jnp.array(2)),
        'd': jnp.array(3.0),
      },
    }
    lt_2 = lambda _, x: x < 2
    popped = nnx.pop(node, (nnx.Param, lt_2))

    self.assertEqual(popped['a']['b'], 1)
    self.assertEqual(popped['a']['c'][...], 2)
    self.assertEqual(node['a']['d'], 3.0)
    self.assertLen(jax.tree.leaves(node), 1)
    self.assertLen(jax.tree.leaves(popped), 2)

  def test_iter_graph(self):
    arr0 = jnp.zeros(1)
    arr1 = jnp.zeros(1)
    var0 = nnx.Variable(jnp.zeros(1))
    var1 = nnx.Variable(jnp.zeros(1))

    child = nnx.Module()
    child.a = var0
    child.b = arr0
    child.c = var1
    child.d = var0
    child.e = arr1
    child.f = arr0

    root = nnx.Module()
    root.a = child
    root.b = var1
    root.c = arr1
    root.d = var1
    root.e = child
    root.f = var0
    root.g = arr1

    nodes = [node for _, node in nnx.iter_graph(root)]
    count = lambda e: sum(node is e for node in nodes)

    # All internal nodes must be visited exactly once.
    self.assertEqual(count(var0), 1)
    self.assertEqual(count(var1), 1)
    self.assertEqual(count(child), 1)
    self.assertEqual(count(root), 1)

    # Arrays must not be deduplicated.
    self.assertEqual(count(arr0), 2)
    self.assertEqual(count(arr1), 3)

    # Nodes must be yielded in DFS order.
    expected = [var0, arr0, var1, arr1, arr0, child, arr1, arr1, root]
    unique = (arr0, arr1, var0, var1, child, root)
    index = lambda e: next(node is e for node in unique)
    actual = [node for node in nodes if any(node is e for e in unique)]
    self.assertEqual(list(map(index, actual)), list(map(index, expected)))

  def test_cached_partial_docstring_example(self):
    import optax

    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adamw(1e-3), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, optimizer, x, y):
      def loss_fn(model):
        return jnp.mean((model(x) - y) ** 2)

      loss, grads = nnx.value_and_grad(loss_fn)(model)
      optimizer.update(model, grads)
      return loss

    cached_train_step = nnx.cached_partial(train_step, model, optimizer)

    for step in range(2):
      x, y = jnp.ones((10, 2)), jnp.ones((10, 3))
      loss = cached_train_step(x, y)
      self.assertIsInstance(loss, jax.Array)

  def test_find_duplicates(self):
    class SharedModules(nnx.Module):
      def __init__(self, rngs: nnx.Rngs):
        self.a = nnx.Linear(1, 1, rngs=rngs)
        self.b = nnx.Linear(1, 1, rngs=rngs)
        self.c = self.a  # shared Module

    model = SharedModules(nnx.Rngs(0))
    duplicates = nnx.find_duplicates(model)

    self.assertLen(duplicates, 1)
    self.assertEqual(duplicates[0], [('a',), ('c',)])

  def test_resursive_map(self):
    class Foo(nnx.Pytree):
      def __init__(self, d):
        self.d = d

    foo1 = Foo(10)
    foo2 = Foo(20)
    bar = [foo1, foo2, foo1]
    n = 0

    def inc_d(path, node):
      nonlocal n
      if isinstance(node, Foo):
        n += 1
        node.d += 1
      return node

    bar2 = nnx.recursive_map(inc_d, bar)
    self.assertIs(bar2[0], bar2[2])
    self.assertEqual(bar2[0].d, 11)
    self.assertEqual(bar2[1].d, 21)
    self.assertEqual(n, 2)

  def test_resursive_map_replace(self):
    class Foo(nnx.Pytree):
      def __init__(self, d):
        self.d = d

    foo1 = Foo(10)
    foo2 = Foo(20)
    bar = [foo1, foo2, foo1]
    n = 0

    def swap(path, node):
      nonlocal n
      if isinstance(node, Foo):
        n += 1
        node = Foo(-node.d)
      return node

    bar2 = nnx.recursive_map(swap, bar)
    self.assertIs(bar2[0], bar2[2])
    self.assertEqual(bar2[0].d, -10)
    self.assertEqual(bar2[1].d, -20)
    self.assertEqual(n, 2)

  def test_graphdef_hash_with_sequential(self):
    rngs = nnx.Rngs(0)
    net = nnx.Sequential(
        nnx.Linear(2, 1, rngs=rngs),
    )
    hash(nnx.graphdef(net))

class SimpleModule(nnx.Module):
  pass


class TestThreading(parameterized.TestCase):
  def test_threading(self):
    x = SimpleModule()

    class MyThread(Thread):
      def run(self) -> None:
        nnx.graph.split(x)

    thread = MyThread()
    thread.start()
    thread.join()


if __name__ == '__main__':
  absltest.main()
