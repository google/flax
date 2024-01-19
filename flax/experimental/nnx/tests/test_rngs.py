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

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from flax.experimental import nnx
from flax.experimental.nnx.nnx.rnglib import _stable_hash


class TestRngs:
  def test_hash(self):
    _hash = _stable_hash('hi')
    assert isinstance(_hash, int)

  def test_call(self):
    ctx = nnx.Ctx(0)
    key = ctx()

  def test_fallback(self):
    ctx = nnx.Ctx(0)
    key = ctx.dropout()

  def test_fallback_error_no_default(self):
    ctx = nnx.Ctx(some_name=0)
    with pytest.raises(AttributeError, match='No RNG named'):
      key = ctx.dropout()

  def test_rng_stream(self):
    key0 = jax.random.key(0)
    ctx = nnx.Ctx(params=key0)
    assert ctx.rngs._rngs['params'].counts[-1] == 0

    key1 = ctx.params()
    assert ctx.rngs._rngs['params'].counts[-1] == 1
    assert ctx.rngs._rngs['params'].key is key0
    assert not jnp.allclose(key0, key1)

    key2 = ctx.params()
    assert ctx.rngs._rngs['params'].counts[-1] == 2
    assert ctx.rngs._rngs['params'].key is key0
    assert not jnp.allclose(key1, key2)

  def test_rng_fork(self):
    key0 = jax.random.key(0)
    rngs1 = nnx.Ctx(params=key0)
    rngs2 = nnx.Ctx(rngs1.fork())

    assert rngs2.rngs._rngs['params'].counts == [0, 0]

    key1 = rngs1.params()
    key2 = rngs2.params()

    assert not jnp.allclose(key1, key2)

  def test_rng_trace_level_constraints(self):
    ctx = nnx.Ctx(0)

    @jax.jit
    def f():
      with pytest.raises(
        nnx.TraceContextError,
        match='Cannot use Rngs from a different trace level',
      ):
        ctx.params()

    f()

    @jax.jit
    def f():
      with pytest.raises(
        nnx.TraceContextError,
        match='Cannot use Rngs from a different trace level',
      ):
        ctx.fork()

    f()

    rngs1: Any = None

    @jax.jit
    def g():
      nonlocal rngs1
      rngs1 = nnx.Ctx(1)

    g()

    assert isinstance(rngs1, nnx.Ctx)
    with pytest.raises(
      nnx.TraceContextError,
      match='Cannot use Rngs from a different trace level',
    ):
      rngs1.params()

  def test_partition_merge(self):
    ctx = nnx.Ctx(dropout=0)

    keys, flags = ctx.fork()

    assert flags == {}
    assert 'dropout' in keys
    assert keys['dropout'].counts == [0, 0]

    rngs2 = nnx.Ctx(keys)

    key1 = ctx.dropout()
    key2 = rngs2.dropout()
    assert not jnp.allclose(key1, key2)

    rngs3 = nnx.Ctx(keys)
    key3 = rngs3.dropout()
    assert jnp.allclose(key2, key3)

  def test_fork_broadcast(self):
    ctx = nnx.Ctx(params=0, dropout=1)
    jax.random.key

    keys, flags = ctx.fork()  # all broadcast

    assert flags == {}
    assert keys['params'].key.shape == ()
    assert keys['dropout'].key.shape == ()
    assert jnp.allclose(keys['params'].key, jax.random.key(0))
    assert jnp.allclose(keys['dropout'].key, jax.random.key(1))

  def test_fork_split(self):
    ctx = nnx.Ctx(params=0, dropout=1)
    keys, flags = ctx.fork(4)  # split all

    assert flags == {}
    assert keys['params'].key.shape == (4,)
    assert keys['dropout'].key.shape == (4,)

  def test_fork_split_and_broadcast(self):
    ctx = nnx.Ctx(params=0, dropout=1)
    splits, broadcasts, flags = ctx.fork(params=4, dropout=None)

    assert flags == {}
    assert splits['params'].key.shape == (4,)
    assert broadcasts['dropout'].key.shape == ()

  def test_fork_filters(self):
    ctx = nnx.Ctx(params=0, dropout=1)
    splits, broadcasts, flags = ctx.fork({'params': 4})

    assert splits['params'].key.shape == (4,)
    assert broadcasts['dropout'].key.shape == ()

  def test_fork_multidimensional_split(self):
    ctx = nnx.Ctx(params=0, dropout=1)
    keys, flags = ctx.fork((4, None, 3))  # split all

    assert flags == {}
    assert keys['params'].key.shape == (4, 1, 3)
    assert keys['dropout'].key.shape == (4, 1, 3)

  def test_fork_multidimensional_split_mixed(self):
    ctx = nnx.Ctx(params=0, dropout=1)
    splits, broadcasts, flags = ctx.fork(params=(4, None, 3))  # split all

    assert splits['params'].key.shape == (4, 1, 3)
    assert broadcasts['dropout'].key.shape == ()

  def test_rng_stream_pytree(self):
    ctx = nnx.Ctx(params=0, dropout=1)
    stream = ctx.fork()[0]['params']

    stream2 = jax.tree_map(lambda x: x, stream)

    assert stream.key is stream2.key
