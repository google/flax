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

import collections.abc
from absl.testing import absltest
from flax import linen
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax


class TrainState(nnx.TrainState):
  batch_stats: nnx.State


class TestHelpers(absltest.TestCase):
  def test_train_state(self):
    m = nnx.Dict(a=nnx.Param(1), b=nnx.BatchStat(2))

    graphdef, params, batch_stats = nnx.split(m, nnx.Param, nnx.BatchStat)

    state = TrainState.create(
      graphdef,
      params=params,
      tx=optax.sgd(1.0),
      batch_stats=batch_stats,
    )

    leaves = jax.tree_util.tree_leaves(state)

  def test_train_state_methods(self):
    class Foo(nnx.Module):
      def __init__(self, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(2, 4, rngs=rngs)
        self.batch_norm = nnx.BatchNorm(4, rngs=rngs)

      def __call__(self, x: jax.Array, train: bool) -> jax.Array:
        x = self.linear(x)
        x = self.batch_norm(x, use_running_average=not train)
        return x

    module = Foo(rngs=nnx.Rngs(0))
    graphdef, params, batch_stats = nnx.split(module, nnx.Param, nnx.BatchStat)

    state = TrainState.create(
      graphdef,
      params=params,
      tx=optax.sgd(1.0),
      batch_stats=batch_stats,
    )

    x = jax.numpy.ones((1, 2))
    y, _updates = state.apply('params', 'batch_stats')(x, train=True)

    assert y.shape == (1, 4)

    # fake gradient
    grads = jax.tree.map(jnp.ones_like, state.params)
    # test apply_gradients
    state = state.apply_gradients(grads)

  def test_nnx_linen_sequential_equivalence(self):
    key1, key2 = jax.random.split(jax.random.key(0), 2)
    rngs = nnx.Rngs(0)
    x = jax.random.uniform(key1, (3, 1, 5))

    model_nnx = nnx.Sequential(
      nnx.Linear(5, 4, rngs=rngs), nnx.Linear(4, 2, rngs=rngs)
    )
    model = linen.Sequential([linen.Dense(4), linen.Dense(2)])

    variables = model.init(key2, x)
    for layer_index in range(2):
      for param in ('kernel', 'bias'):
        variables['params'][f'layers_{layer_index}'][param] = getattr(
          model_nnx.layers[layer_index], param
        )[...]
    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    np.testing.assert_array_equal(out, out_nnx)

    variables = model.init(key2, x)
    for layer_index in range(2):
      for param in ('kernel', 'bias'):
        getattr(model_nnx.layers[layer_index], param)[...] = variables[
          'params'
        ][f'layers_{layer_index}'][param]
    out_nnx = model_nnx(x)
    out = model.apply(variables, x)
    np.testing.assert_array_equal(out, out_nnx)

  def test_nnx_empty_sequential_is_identity(self):
    iden = nnx.Sequential()
    assert iden(12) == 12
    assert iden(12, 23) == (12, 23)
    assert iden() is None
    assert iden(k=2) == {'k': 2}

  def test_dict_mutable_mapping(self):
    d = nnx.Dict({'a': 1, 'b': 2})
    self.assertEqual(d['a'], 1)
    self.assertEqual(d['b'], 2)
    self.assertEqual(len(d), 2)

    d['c'] = 3
    self.assertEqual(d['c'], 3)
    self.assertEqual(len(d), 3)

    del d['a']
    self.assertEqual(len(d), 2)
    with self.assertRaises(KeyError):
      _ = d['a']

    self.assertSetEqual(set(d), {'b', 'c'})

  def test_dict_setdefault(self):
    d = nnx.Dict({'a': 1, 'b': 2})
    self.assertEqual(d.setdefault('a', 10), 1)
    self.assertEqual(d['a'], 1)

    self.assertEqual(d.setdefault('c', 3), 3)
    self.assertEqual(d['c'], 3)
    self.assertEqual(len(d), 3)

  def test_dict_contains(self):
    d = nnx.Dict({'a': 1, 'b': 2})
    self.assertIn('a', d)
    self.assertIn('b', d)
    self.assertNotIn('c', d)

    d['c'] = 3
    self.assertIn('c', d)

    del d['a']
    self.assertNotIn('a', d)

  def test_list_mutable_sequence(self):
    l = nnx.List([1, 2, 3])
    self.assertEqual(len(l), 3)
    self.assertEqual(l[0], 1)
    self.assertEqual(l[1], 2)
    self.assertEqual(l[2], 3)

    l.append(4)
    self.assertEqual(len(l), 4)
    self.assertEqual(l[3], 4)

    l.insert(1, 5)
    self.assertEqual(len(l), 5)
    self.assertEqual(l[0], 1)
    self.assertEqual(l[1], 5)
    self.assertEqual(l[2], 2)
    self.assertEqual(l[3], 3)
    self.assertEqual(l[4], 4)

    del l[2]
    self.assertEqual(len(l), 4)
    self.assertEqual(l[0], 1)
    self.assertEqual(l[1], 5)
    self.assertEqual(l[2], 3)
    self.assertEqual(l[3], 4)

    l[1:3] = [6, 7]
    self.assertEqual(l[1], 6)
    self.assertEqual(l[2], 7)

    self.assertEqual(l[1:3], [6, 7])

  def test_list_fori_loop(self):
    class Foo(nnx.Module):
      def __init__(self):
        self.layers = nnx.List([
            nnx.Linear(1, 1, rngs=nnx.Rngs(0)),
            nnx.Linear(1, 1, rngs=nnx.Rngs(0)),
        ])

    def batch_loop_body(i, carry):
      return carry

    net = Foo()
    jax.lax.fori_loop(0, 2, batch_loop_body, net)

  def test_list_pytree_default_behavior(self):
    ls = nnx.List([jnp.array(1), jnp.array(2), jnp.array(3)])
    leaves = jax.tree_util.tree_leaves(ls)
    self.assertLen(leaves, 3)
    np.testing.assert_array_equal(leaves[0], jnp.array(1))
    np.testing.assert_array_equal(leaves[1], jnp.array(2))
    np.testing.assert_array_equal(leaves[2], jnp.array(3))

  def test_list_pytree_static_elements(self):
    ls = nnx.List([nnx.static(10), nnx.static(20), nnx.static(30)])
    leaves = jax.tree_util.tree_leaves(ls)
    self.assertEmpty(leaves)

  def test_list_pytree_data_elements(self):
    ls = nnx.List([nnx.data(1), nnx.data(2), nnx.data(3)])
    leaves = jax.tree_util.tree_leaves(ls)
    self.assertLen(leaves, 3)
    self.assertEqual(leaves[0], 1)
    self.assertEqual(leaves[1], 2)
    self.assertEqual(leaves[2], 3)

  def test_list_pytree_mixed_static_data(self):
    ls = nnx.List([
        nnx.data(jnp.array(1)),
        nnx.static(100),
        nnx.data(jnp.array(2)),
        nnx.static(200),
    ])
    leaves = jax.tree_util.tree_leaves(ls)
    self.assertLen(leaves, 2)
    np.testing.assert_array_equal(leaves[0], jnp.array(1))
    np.testing.assert_array_equal(leaves[1], jnp.array(2))

  def test_list_pytree_flatten_unflatten(self):
    ls = nnx.List([nnx.data(10), nnx.static('hello'), nnx.data(20)])
    leaves, treedef = jax.tree_util.tree_flatten(ls)
    self.assertLen(leaves, 2)
    self.assertEqual(leaves[0], 10)
    self.assertEqual(leaves[1], 20)

    new_leaves = [x * 2 for x in leaves]
    new_ls = jax.tree_util.tree_unflatten(treedef, new_leaves)
    self.assertEqual(new_ls[0], 20)
    self.assertEqual(new_ls[1], 'hello')
    self.assertEqual(new_ls[2], 40)

  def test_list_pytree_jit(self):
    ls = nnx.List([nnx.data(jnp.array(1.0)), nnx.static(999)])

    @jax.jit
    def double(ls):
      return jax.tree.map(lambda x: x * 2, ls)

    result = double(ls)
    np.testing.assert_array_equal(result[0], jnp.array(2.0))
    self.assertEqual(result[1], 999)

  def test_sequential_iterable(self):
    rngs = nnx.Rngs(0)
    l1 = nnx.Linear(4, 3, rngs=rngs)
    l2 = nnx.Linear(3, 2, rngs=rngs)

    seq = nnx.Sequential(l1, nnx.relu, l2)
    self.assertIsInstance(seq, collections.abc.Iterable)
    self.assertEqual(list(seq), [l1, nnx.relu, l2])
    for idx, layer in enumerate(seq):
      self.assertIs(layer, seq.layers[idx])

    # Nested Sequential remains a single callable layer.
    outer = nnx.Sequential(seq)
    self.assertEqual(list(outer), [seq])
    x = jnp.ones((2, 4))
    np.testing.assert_allclose(outer(x), seq(x))

  def test_sequential_heterogeneous_kwargs_forwarding(self):
    rngs = nnx.Rngs(params=0)
    l1 = nnx.Linear(4, 4, rngs=rngs)
    bn = nnx.BatchNorm(4, rngs=rngs)
    drop = nnx.Dropout(rate=0.5)
    l2 = nnx.Linear(4, 2, rngs=rngs)

    seq = nnx.Sequential(l1, bn, nnx.relu, drop, l2)
    x = jnp.ones((3, 4))

    # Eval call: BatchNorm receives use_running_average=True, Dropout receives
    # deterministic=True, while Linear and relu ignore both kwargs cleanly.
    mean_before = bn.mean[...].copy()
    y_eval_1 = seq(x, use_running_average=True, deterministic=True)
    y_eval_2 = seq(x, use_running_average=True, deterministic=True)
    np.testing.assert_allclose(y_eval_1, y_eval_2)
    np.testing.assert_array_equal(bn.mean[...], mean_before)

    # Train call with explicit call-time rngs (Dropout has no init-time rngs):
    # BatchNorm updates stats and Dropout uses the forwarded call-time rngs.
    y_train_1 = seq(
        x,
        use_running_average=False,
        deterministic=False,
        rngs=nnx.Rngs(dropout=42),
    )
    y_train_2 = seq(
        x,
        use_running_average=True,
        deterministic=False,
        rngs=nnx.Rngs(dropout=42),
    )
    self.assertEqual(y_train_1.shape, (3, 2))
    self.assertFalse(jnp.array_equal(bn.mean[...], mean_before))
    self.assertFalse(jnp.allclose(y_train_2, y_eval_1))

    # Layers with **kwargs receive all shared kwargs; dict/tuple chaining still
    # works alongside shared sideband kwargs.
    seen_kwargs = []

    def dict_producer(h, *, scale=1.0):
      return {'a': h * scale, 'b': jnp.ones_like(h)}

    def dict_consumer(a, b, **extra):
      seen_kwargs.append(extra)
      return a + b

    chained = nnx.Sequential(l1, dict_producer, dict_consumer)
    out = chained(x, scale=2.0, tag='meta')
    np.testing.assert_allclose(out, l1(x) * 2.0 + 1.0)
    self.assertEqual(seen_kwargs, [{'scale': 2.0, 'tag': 'meta'}])

    # Passing Layer 0's input by keyword does not collide on Layer 1.
    two_linears = nnx.Sequential(l1, l2)
    np.testing.assert_allclose(two_linears(inputs=x), l2(l1(x)))

  def test_sequential_unused_and_conflicting_kwargs(self):
    rngs = nnx.Rngs(0)
    lin_seq = nnx.Sequential(
        nnx.Linear(4, 4, rngs=rngs),
        nnx.relu,
        nnx.Linear(4, 2, rngs=rngs),
    )
    x = jnp.ones((2, 4))

    # Unused kwargs (e.g. deterministic / use_running_average / rngs when
    # optional Dropout or BatchNorm layers are omitted) are ignored cleanly.
    out = lin_seq(
        x,
        deterministic=True,
        use_running_average=True,
        rngs=nnx.Rngs(dropout=0),
    )
    np.testing.assert_allclose(out, lin_seq(x))

    # Collision between a returned dict key and an accepted call kwarg raises
    # TypeError rather than silently shadowing either value.
    def dict_out(h):
      return {'a': h, 'scale': 3.0}

    def consumer(a, *, scale=1.0):
      return a * scale

    conflict_seq = nnx.Sequential(dict_out, consumer)
    with self.assertRaisesRegex(
        TypeError, r'Got multiple values for arguments .* in Sequential\[1\]'
    ):
      conflict_seq(x, scale=2.0)

    # Unaccepted keys in a returned dict still raise TypeError on the consumer.
    def bad_dict_out(h):
      return {'a': h, 'unexpected_key': 1.0}

    bad_seq = nnx.Sequential(bad_dict_out, consumer)
    with self.assertRaisesRegex(
        TypeError, "unexpected keyword argument 'unexpected_key'"
    ):
      bad_seq(x)

  def test_sequential_repeated_blocks_shared_kwargs_and_scan(self):
    class DecoderBlock(nnx.Module):

      def __init__(self, dim: int, *, rngs: nnx.Rngs):
        self.norm = nnx.RMSNorm(dim, rngs=rngs)
        self.proj_in = nnx.Linear(dim, dim, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.25)
        self.proj_out = nnx.Linear(dim, dim, rngs=rngs)

      def __call__(
          self,
          x: jax.Array,
          *,
          mask: jax.Array | None = None,
          positions: jax.Array | None = None,
          deterministic: bool = False,
          rngs: nnx.Rngs | None = None,
      ) -> jax.Array:
        h = self.norm(x)
        if positions is not None:
          h = h + positions[..., None].astype(h.dtype) * 0.1
        h = nnx.silu(self.proj_in(h))
        h = self.dropout(h, deterministic=deterministic, rngs=rngs)
        h = self.proj_out(h)
        if mask is not None:
          h = jnp.where(mask[..., None], h, 0.0)
        return x + h

    class Decoder(nnx.Module):

      def __init__(self, dim: int, num_layers: int, *, rngs: nnx.Rngs):
        self.layers = nnx.Sequential(
            *(DecoderBlock(dim, rngs=rngs) for _ in range(num_layers))
        )

      def __call__(
          self,
          x: jax.Array,
          *,
          mask: jax.Array | None = None,
          positions: jax.Array | None = None,
          deterministic: bool = False,
          rngs: nnx.Rngs | None = None,
      ) -> jax.Array:
        return self.layers(
            x,
            mask=mask,
            positions=positions,
            deterministic=deterministic,
            rngs=rngs,
        )

    num_layers = 4
    batch, seq_len, dim = 2, 5, 8
    rngs = nnx.Rngs(params=0, dropout=1)
    decoder = Decoder(dim=dim, num_layers=num_layers, rngs=rngs)

    self.assertLen(list(decoder.layers), num_layers)
    for idx, block in enumerate(decoder.layers):
      self.assertIs(block, decoder.layers.layers[idx])

    x = jax.random.normal(jax.random.key(10), (batch, seq_len, dim))
    mask = jnp.array(
        [[True, True, True, False, False], [True, True, True, True, False]]
    )
    positions = jnp.broadcast_to(jnp.arange(seq_len), (batch, seq_len))

    # 1. Unscanned forward pass forwards shared kwargs to every block.
    expected = x
    for block in decoder.layers:
      expected = block(
          expected, mask=mask, positions=positions, deterministic=True
      )
    out_unscanned = decoder(
        x, mask=mask, positions=positions, deterministic=True
    )
    np.testing.assert_allclose(out_unscanned, expected, rtol=1e-6, atol=1e-6)

    # 2. Stochastic forward pass with call-time rngs works across all blocks.
    out_stoch = decoder(
        x,
        mask=mask,
        positions=positions,
        deterministic=False,
        rngs=nnx.Rngs(dropout=99),
    )
    self.assertEqual(out_stoch.shape, (batch, seq_len, dim))
    self.assertFalse(jnp.allclose(out_stoch, out_unscanned))

    # 3. Post-construction layer scan transformation over decoder.layers matches
    #    both forward output and gradients of unscanned Sequential execution.
    def loss_unscanned(model: Decoder, inputs: jax.Array) -> jax.Array:
      y = model(inputs, mask=mask, positions=positions, deterministic=True)
      return jnp.mean(y**2)

    val_unscanned, grads_unscanned = nnx.jit(
        nnx.value_and_grad(loss_unscanned)
    )(decoder, x)

    graphdefs, states = zip(*(nnx.split(b) for b in decoder.layers))
    stacked_state = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    stacked_blocks = nnx.merge(graphdefs[0], stacked_state)

    def loss_scanned(blocks: DecoderBlock, inputs: jax.Array) -> jax.Array:
      @nnx.scan(in_axes=(0, nnx.Carry), out_axes=nnx.Carry)
      def scan_body(block: DecoderBlock, carry: jax.Array) -> jax.Array:
        return block(carry, mask=mask, positions=positions, deterministic=True)

      y = scan_body(blocks, inputs)
      return jnp.mean(y**2)

    val_scanned, grads_scanned = nnx.jit(nnx.value_and_grad(loss_scanned))(
        stacked_blocks, x
    )
    np.testing.assert_allclose(val_scanned, val_unscanned, rtol=1e-6, atol=1e-6)
    for i in range(num_layers):
      layer_grad_from_scan = jax.tree.map(
          lambda g, idx=i: g[idx], grads_scanned
      )
      layer_grad_unscanned = grads_unscanned['layers']['layers'][i]
      jax.tree.map(
          lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5),
          layer_grad_from_scan,
          layer_grad_unscanned,
      )


if __name__ == '__main__':
  absltest.main()
