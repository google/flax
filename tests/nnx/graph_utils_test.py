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

from collections.abc import Callable
import dataclasses
from functools import partial
from threading import Thread
from typing import Any

from absl.testing import absltest, parameterized
from flax import linen, nnx, struct
import jax
import jax.numpy as jnp

class List(nnx.Module):
  def __init__(self, items):
    self.items = list(items)

  def __getitem__(self, idx):
    return self.items[idx]

  def __setitem__(self, idx, value):
    self.items[idx] = value


class Dict(nnx.Module):
  def __init__(self, *args, **kwargs):
    self.items = dict(*args, **kwargs)

  def __getitem__(self, key):
    return self.items[key]

  def __setitem__(self, key, value):
    self.items[key] = value

class StatefulLinear(nnx.Module):
  def __init__(self, din, dout, rngs):
    self.w = nnx.Param(jax.random.uniform(rngs(), (din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))
    self.count = nnx.Variable(jnp.array(0, dtype=jnp.uint32))

  def increment(self):
    self.count.value += 1

  def __call__(self, x):
    self.count.value += 1
    return x @ self.w + self.b[None]


class TestGraphUtils(absltest.TestCase):
  def test_flatten(self):
    a = {'a': 1, 'b': nnx.Param(2)}
    g = [a, 3, a, nnx.Param(4)]

    refmap = nnx.graph.RefMap()
    graphdef, flat_state = nnx.graph.flatten(g, ref_index=refmap)

    assert flat_state[0][1].value == 2
    assert flat_state[1][1].value == 4

    assert len(refmap) == 2
    assert a['b'] in refmap
    assert g[3] in refmap

  def test_flatten_no_paths(self):
    a = {'a': 1, 'b': nnx.Param(2)}
    g = [a, 3, a, nnx.Param(4)]

    refmap = nnx.graph.RefMap()
    graphdef, flat_state = nnx.graph.flatten(
      g, ref_index=refmap, with_paths=False
    )

    assert flat_state[0] == 2
    assert flat_state[1] == 4

    assert len(refmap) == 2
    assert a['b'] in refmap
    assert g[3] in refmap

  def test_unflatten(self):
    a = Dict(a=1, b=nnx.Param(2))
    g = List([a, 3, a, nnx.Param(4)])

    graphdef, state = nnx.split(g)
    g = nnx.merge(graphdef, state)

    assert g[0] is g[2]

  def test_unflatten_pure_dict(self):
    a = Dict(a=1, b=nnx.Param(2))
    g = List([a, 3, a, nnx.Param(4)])

    graphdef, state = nnx.split(g)
    pure_state = state.to_pure_dict()

    g = nnx.merge(graphdef, pure_state)

    assert g[0] is g[2]

  def test_unflatten_pytree(self):
    a = {'a': 1, 'b': nnx.Param(2)}
    g = [a, 3, a, nnx.Param(4)]

    graphdef, state = nnx.split(g)
    g = nnx.merge(graphdef, state)

    assert g[0] is not g[2]

  def test_unflatten_empty(self):
    a = Dict({'a': 1, 'b': nnx.Param(2)})
    g = List([a, 3, a, nnx.Param(4)])

    graphdef, state = nnx.split(g)

    with self.assertRaisesRegex(
      ValueError, 'Not enough leaves to unflatten the graph'
    ):
      nnx.graph.unflatten(graphdef, nnx.State({}))

  def test_unflatten_return_variables(self):
    a = Dict({'a': 1, 'b': nnx.Param(2)})
    g = List([a, 3, a, nnx.Param(4)])

    graphdef, state = nnx.graph.flatten(
      g, with_paths=False, return_variables=True
    )

    self.assertLen(state, 2)
    self.assertIsInstance(state, list)
    self.assertIsInstance(state[0], nnx.Param)
    self.assertIsInstance(state[1], nnx.Param)

  def test_clone_with_same_variables(self):
    a = Dict({'a': 1, 'b': nnx.Param(2)})
    g = List([a, 3, a, nnx.Param(4)])

    graphdef, state = nnx.graph.flatten(
      g, with_paths=False, return_variables=True
    )

    g2 = nnx.graph.unflatten(graphdef, state)

    self.assertIsNot(g, g2)
    self.assertIsNot(g[0], g2[0])
    self.assertIsNot(g[2], g2[2])
    self.assertIs(g[0]['b'], g2[0]['b'])
    self.assertIs(g[3], g2[3])

  def test_update_dynamic(self):
    a = {'a': 1, 'b': nnx.Param(2)}
    g = [a, 3, a, nnx.Param(4)]

    graphdef, state = nnx.split(g)

    state[0]['b'].value = 3
    nnx.update(g, state)

    assert g[0]['b'].value == 3
    assert g[2]['b'].value == 3

  def test_update_from_pure_dict(self):
    a = {'a': 1, 'b': nnx.Param(2)}
    g = [a, 3, a, nnx.Param(4)]

    graphdef, state = nnx.split(g)
    pure_state = state.to_pure_dict()

    pure_state[0]['b'] = 3
    nnx.update(g, pure_state)

    assert g[0]['b'].value == 3
    assert g[2]['b'].value == 3


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
    assert graphdef.attributes[0][1].type is nnx.graph.GenericPytree

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
      idx_out_ref_in: dict[int, Any] = {}
      m = nnx.graph.unflatten(graphdef, state, index_ref=idx_out_ref_in)
      ref_in_idx_out = nnx.graph.RefMap(
        {v: k for k, v in idx_out_ref_in.items()}
      )
      f(m)
      ref_in_idx_in = nnx.graph.RefMap()
      graphdef, state = nnx.graph.flatten(
        m, ref_index=ref_in_idx_in, ref_outer_index=ref_in_idx_out
      )
      state = state.to_nested_state()
      return state, graphdef

    state, graphdef_out = f_pure(graphdef, state)
    idx_out_ref_out = {v: k for k, v in ref_out_idx_out.items()}
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
      idx_out_ref_in: dict[int, Any] = {}
      m = nnx.graph.unflatten(graphdef, state, index_ref=idx_out_ref_in)
      ref_in_idx_out = nnx.graph.RefMap(
        {v: k for k, v in idx_out_ref_in.items()}
      )
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
        self.ref = None

    def f(m: Foo):
      m.ref = m

    m = Foo()

    ref_out_idx_out = nnx.graph.RefMap()
    graphdef: nnx.graph.GraphDef[Foo]
    graphdef, state = nnx.graph.flatten(m, ref_index=ref_out_idx_out)
    idx_out_ref_out = {v: k for k, v in ref_out_idx_out.items()}
    state = state.to_nested_state()

    @partial(jax.jit, static_argnums=(0,))
    def f_pure(graphdef: nnx.graph.GraphDef[Foo], state):
      idx_out_ref_in: dict[int, Any] = {}
      m = nnx.graph.unflatten(graphdef, state, index_ref=idx_out_ref_in)
      ref_in_idx_out = nnx.graph.RefMap(
        {v: k for k, v in idx_out_ref_in.items()}
      )
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
        self.count.value += 1
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

    self.assertEqual(counter.count.value, 2)

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

    self.assertEqual(linear.count.value, 0)
    new_linear = nnx.merge(*linear_state)
    self.assertEqual(new_linear.count.value, 2)

  def test_getitem(self):
    rngs = nnx.Rngs(0)
    nodes = dict(
      a=StatefulLinear(3, 2, rngs),
      b=StatefulLinear(2, 1, rngs),
    )
    node_state = nnx.split(nodes)
    _, node_state = nnx.call(node_state)['b'].increment()

    nodes = nnx.merge(*node_state)

    self.assertEqual(nodes['a'].count.value, 0)
    self.assertEqual(nodes['b'].count.value, 1)

  def test_object_state_propagation(self):
    test = self

    class Foo(nnx.Module):
      def __call__(self):
        test.assertTrue(self._object__state.initializing)
        self = nnx.merge(*nnx.split(self))
        test.assertTrue(self._object__state.initializing)

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
    self.assertIsInstance(graphdef1, nnx.graph.NodeDef)
    self.assertIsInstance(graphdef2, nnx.graph.NodeRef)
    self.assertLen(state1.flat_state(), 2)
    self.assertLen(state2.flat_state(), 0)

    with nnx.graph.merge_context() as ctx:
      m1 = ctx.merge(graphdef1, state1)
      m2 = ctx.merge(graphdef2, state2)

    self.assertIs(m1, m2)
    self.assertFalse(hasattr(ctx, 'index_ref'))
    self.assertFalse(hasattr(ctx, 'ctxtag'))

  def test_split_merge_context_nested(self):
    m2 = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    m1 = nnx.Sequential(m2)
    with nnx.graph.split_context() as ctx:
      graphdef1, state1 = ctx.split(m1)
      graphdef2, state2 = ctx.split(m2)

    self.assertIsInstance(graphdef1, nnx.graph.NodeDef)
    self.assertIsInstance(graphdef2, nnx.graph.NodeRef)
    self.assertLen(state1.flat_state(), 2)
    self.assertLen(state2.flat_state(), 0)

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
        self.b = 2

    m = Foo()
    ctxtag = 'test'

    with nnx.update_context(ctxtag):
      with nnx.graph.split_context(ctxtag) as ctx:
        graphdef1, state1 = ctx.split(m)
        graphdef2, state2 = ctx.split(m)

      self.assertFalse(hasattr(ctx, 'ref_index'))
      self.assertFalse(hasattr(ctx, 'ctxtag'))
      self.assertIsInstance(graphdef1, nnx.graph.NodeDef)
      self.assertIsInstance(graphdef2, nnx.graph.NodeRef)
      self.assertLen(state1.flat_state(), 1)
      self.assertLen(state2.flat_state(), 0)

      @jax.jit
      def f(graphdef1, state1, graphdef2, state2):
        with nnx.graph.merge_context(True, ctxtag) as ctx:
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

      with nnx.graph.merge_context(False, ctxtag) as ctx:
        m1_out = ctx.merge(graphdef1, state1)
        m2_out = ctx.merge(graphdef2, state2)

      self.assertIs(m, m1_out)
      self.assertIs(m, m2_out)
      self.assertEqual(m.a, 2)
      self.assertEqual(m.b.value, 1)  # type: ignore

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
    self.assertIsInstance(t1.graphdef, nnx.graph.NodeDef)
    self.assertIsInstance(t2.graphdef, nnx.graph.NodeRef)
    self.assertLen(t1.states[0].flat_state(), 2)
    self.assertLen(t2.states[0].flat_state(), 0)

    impure_tree2 = nnx.from_tree(pure_tree)

    m1_out = impure_tree2[0]
    m2_out = impure_tree2[2]['b']

    self.assertIs(m1_out, m2_out)
    self.assertEqual(impure_tree2[1], 1)

  def test_to_tree_update_context(self):
    class Foo(nnx.Module):

      def __init__(self):
        self.a = nnx.Param(1)
        self.b = 2

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
      self.assertIsInstance(t1.graphdef, nnx.graph.NodeDef)
      self.assertIsInstance(t2.graphdef, nnx.graph.NodeRef)
      self.assertLen(t1.states[0].flat_state(), 1)
      self.assertLen(t2.states[0].flat_state(), 0)

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
        self.assertIsInstance(t1.graphdef, nnx.graph.NodeDef)
        self.assertIsInstance(t2.graphdef, nnx.graph.NodeRef)
        self.assertLen(t1.states[0].flat_state(), 1)
        self.assertLen(t2.states[0].flat_state(), 0)

        return pure_tree2

      pure_tree2 = f(pure_tree)

      impure_tree2 = nnx.from_tree(pure_tree2, ctxtag=ctxtag, is_inner=False)

      m1_out = impure_tree2[0]
      m2_out = impure_tree2[2]['b']

      self.assertIs(m, m1_out)
      self.assertIs(m, m2_out)
      self.assertEqual(m.a, 2)
      self.assertEqual(m.b.value, 1)  # type: ignore
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

  def test_fingerprint_basic(self):
    m = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    fp1 = nnx.graph.fingerprint(m)
    fp2 = nnx.graph.fingerprint(m)

    self.assertEqual(fp1, fp2)
    self.assertTrue(nnx.graph.check_fingerprint(m, fp1))
    self.assertTrue(nnx.graph.check_fingerprint(m, fp2))

  def test_fingerprint_variable_id_sensitive(self):
    m1 = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    fp1 = nnx.graph.fingerprint(m1)

    m2 = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    fp2 = nnx.graph.fingerprint(m2)

    self.assertNotEqual(fp1, fp2)
    self.assertTrue(nnx.graph.check_fingerprint(m1, fp1))
    self.assertTrue(nnx.graph.check_fingerprint(m2, fp2))
    self.assertFalse(nnx.graph.check_fingerprint(m1, fp2))
    self.assertFalse(nnx.graph.check_fingerprint(m2, fp1))

  def test_fingerprint_module_id_insensitive(self):
    m1 = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    m2 = nnx.Linear(2, 3, rngs=nnx.Rngs(0))

    m1.kernel = m2.kernel
    m1.bias = m2.bias

    fp1 = nnx.graph.fingerprint(m1)
    fp2 = nnx.graph.fingerprint(m2)

    self.assertNotEqual(fp1, fp2)
    self.assertTrue(nnx.graph.check_fingerprint(m1, fp1))
    self.assertTrue(nnx.graph.check_fingerprint(m2, fp2))
    self.assertFalse(nnx.graph.check_fingerprint(m1, fp2))
    self.assertFalse(nnx.graph.check_fingerprint(m2, fp1))


class SimpleModule(nnx.Module):
  pass


class SimplePyTreeModule(nnx.Module, experimental_pytree=True):
  pass


class TestThreading(parameterized.TestCase):

  @parameterized.parameters(
      (SimpleModule,),
      (SimplePyTreeModule,),
  )
  def test_threading(self, module_fn: Callable[[], nnx.Module]):
    x = module_fn()

    class MyThread(Thread):

      def run(self) -> None:
        nnx.graph.split(x)

    thread = MyThread()
    thread.start()
    thread.join()


if __name__ == '__main__':
  absltest.main()
