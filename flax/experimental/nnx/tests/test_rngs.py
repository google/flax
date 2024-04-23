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
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from flax.experimental import nnx


class TestRngs:
  def test_call(self):
    rngs = nnx.Rngs(0)
    key = rngs()

  def test_fallback(self):
    rngs = nnx.Rngs(0)
    key = rngs.dropout()

  def test_fallback_error_no_default(self):
    rngs = nnx.Rngs(some_name=0)
    with pytest.raises(AttributeError, match='No RNG named'):
      key = rngs.dropout()

  def test_rng_stream(self):
    key0 = jax.random.key(0)
    rngs = nnx.Rngs(params=key0)
    assert rngs.params.count.value == 0

    key1 = rngs.params()
    assert rngs.params.count.value == 1
    assert rngs.params.key.value is key0
    assert not jnp.allclose(key0, key1)

    key2 = rngs.params()
    assert rngs.params.count.value == 2
    assert rngs.params.key.value is key0
    assert not jnp.allclose(key1, key2)


  def test_rng_trace_level_constraints(self):
    rngs = nnx.Rngs(0)

    @jax.jit
    def f():
      with pytest.raises(
        nnx.errors.TraceContextError,
        match='Cannot call RngStream from a different trace level',
      ):
        rngs.params()

    f()

    rngs1: Any = None

    @jax.jit
    def h():
      nonlocal rngs1
      rngs1 = nnx.Rngs(1)

    h()

    assert isinstance(rngs1, nnx.Rngs)
    with pytest.raises(
      nnx.errors.TraceContextError,
      match='Cannot call RngStream from a different trace level',
    ):
      rngs1.params()

  def test_jit_updates(self):
    class Foo(nnx.Module):
      def __init__(self, not_rngs):
        rngs = not_rngs
        self.linear = nnx.Linear(2, 2, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, deterministic=False)

      def __call__(self, x, rngs):
        x = self.linear(x)
        x = self.dropout(x, rngs=rngs)
        return x

    rngs = nnx.Rngs(0)
    m = Foo(rngs)

    # +1 for the Linear kernel, +1 for the Linear bias
    assert rngs['default'].count.value == 2

    @nnx.jit
    def f(m: Foo, x: jax.Array, not_rngs: nnx.Rngs):
      rngs = not_rngs
      x = m(x, rngs)
      x = m(x, rngs)
      return x

    x = jnp.ones((2, 2))
    x = f(m, x, rngs)

    # +1 for the Dropout mask
    assert rngs['default'].count.value == 4

  def test_lifting_rng_state(self):
    class Foo(nnx.Module):
      def __init__(self, rngs):
        self.rngs = rngs
        self.dropout = nnx.Dropout(0.5, deterministic=False)
        self.linear = nnx.Linear(2, 3, rngs=rngs)

      def __call__(self, x):
        x = self.linear(x)
        x = self.dropout(x, rngs=self.rngs)
        return x

    rngs = nnx.Rngs(params=0, dropout=1)
    m = Foo(rngs)
    _, params, dropout_keys, param_keys, rng_counts = nnx.split(
      m, nnx.Param, 'dropout', 'params', nnx.RngCount
    )

    assert m.rngs.params.count.value == 2
    assert m.rngs['dropout'].count.value == 0
    assert len(dropout_keys.flat_state()) == 1
    assert len(param_keys.flat_state()) == 1
    assert len(rng_counts.flat_state()) == 2

    # split dropout keys
    split_dropout_keys = jax.tree_util.tree_map(
      lambda x: jax.random.split(x, 4), dropout_keys
    )
    # replicate params
    params = jax.tree_util.tree_map(
      lambda x: jnp.stack([x] * 4, axis=0), params
    )

    @partial(
      jax.vmap,
      in_axes=(0, 0, None, None, 0),
      out_axes=(0, 0, None),
    )
    def f(params, dropout_keys, param_keys, rng_counts, x):
      nnx.update(m, params, dropout_keys, param_keys, rng_counts)
      y = m(x)
      _, params, dropout_keys, param_keys, rng_counts = nnx.split(
        m, nnx.Param, 'dropout', 'params', nnx.RngCount
      )
      return y, params, rng_counts

    x = jnp.ones((4, 1, 2))
    y, params, rng_counts = f(
      params,
      split_dropout_keys,
      param_keys,
      rng_counts,
      x,
    )

    nnx.update(m, params, dropout_keys, param_keys, rng_counts)

    assert y.shape == (4, 1, 3)
    assert m.rngs.params.count.value == 2
    assert m.rngs['dropout'].count.value == 1

  def test_state_fork_split(self):
    rngs = nnx.Rngs(params=0, dropout=1)
    graphdef, state = nnx.split(rngs, nnx.RngState)
    split, broadcast = nnx.fork(state, ..., 4)

    assert len(jax.tree.leaves(split)) == 2
    assert len(jax.tree.leaves(broadcast)) == 2
    assert split.params.key.value.shape == (4,)
    assert split.dropout.key.value.shape == (4,)
    assert broadcast.params.count.value == 0
    assert broadcast.dropout.count.value == 0

  def test_state_fork_split_and_broadcast(self):
    rngs = nnx.Rngs(params=0, dropout=1)
    graphdef, state = nnx.split(rngs, nnx.RngState)
    split, broadcast = nnx.fork(state, 'params', 4)

    assert len(jax.tree.leaves(split)) == 1
    assert len(jax.tree.leaves(broadcast)) == 3
    assert split.params.key.value.shape == (4,)
    assert broadcast.dropout.key.value.shape == ()
    assert broadcast.params.count.value == 0
    assert broadcast.dropout.count.value == 0


  def test_state_fork_multidimensional_split(self):
    rngs = nnx.Rngs(params=0, dropout=1)
    graphdef, state = nnx.split(rngs, nnx.RngState)
    split, broadcast = nnx.fork(state, ..., (4, None, 3))

    assert len(jax.tree.leaves(split)) == 2
    assert len(jax.tree.leaves(broadcast)) == 2
    assert split.params.key.value.shape == (4, 1, 3)
    assert split.dropout.key.value.shape == (4, 1, 3)
    assert broadcast.params.count.value == 0
    assert broadcast.dropout.count.value == 0

  def test_state_fork_multidimensional_split_mixed(self):
    rngs = nnx.Rngs(params=0, dropout=1)
    graphdef, state = nnx.split(rngs, nnx.RngState)
    split, broadcast = nnx.fork(state, 'params', (4, None, 3))

    assert len(jax.tree.leaves(split)) == 1
    assert len(jax.tree.leaves(broadcast)) == 3
    assert split.params.key.value.shape == (4, 1, 3)
    assert broadcast.dropout.key.value.shape == ()
    assert broadcast.params.count.value == 0
    assert broadcast.dropout.count.value == 0
