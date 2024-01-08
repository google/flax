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
    assert rngs._rngs['params'].counts[-1] == 0

    key1 = rngs.params()
    assert rngs._rngs['params'].counts[-1] == 1
    assert rngs._rngs['params'].key is key0
    assert not jnp.allclose(key0, key1)

    key2 = rngs.params()
    assert rngs._rngs['params'].counts[-1] == 2
    assert rngs._rngs['params'].key is key0
    assert not jnp.allclose(key1, key2)

  def test_rng_fork(self):
    key0 = jax.random.key(0)
    rngs1 = nnx.Rngs(params=key0)
    rngs2 = nnx.Rngs(rngs1.fork())

    assert rngs2._rngs['params'].counts == [0, 0]

    key1 = rngs1.params()
    key2 = rngs2.params()

    assert not jnp.allclose(key1, key2)

  def test_rng_trace_level_constraints(self):
    rngs = nnx.Rngs(0)

    @jax.jit
    def f():
      with pytest.raises(
        nnx.TraceContextError,
        match='Cannot use Rngs from a different trace level',
      ):
        rngs.params()

    f()

    @jax.jit
    def f():
      with pytest.raises(
        nnx.TraceContextError,
        match='Cannot use Rngs from a different trace level',
      ):
        rngs.fork()

    f()

    rngs1: Any = None

    @jax.jit
    def g():
      nonlocal rngs1
      rngs1 = nnx.Rngs(1)

    g()

    assert isinstance(rngs1, nnx.Rngs)
    with pytest.raises(
      nnx.TraceContextError,
      match='Cannot use Rngs from a different trace level',
    ):
      rngs1.params()

  def test_partition_merge(self):
    rngs = nnx.Rngs(dropout=0)

    keys = rngs.fork()

    assert 'dropout' in keys
    assert keys['dropout'].counts == [0, 0]

    rngs2 = nnx.Rngs(keys)

    key1 = rngs.dropout()
    key2 = rngs2.dropout()
    assert not jnp.allclose(key1, key2)

    rngs3 = nnx.Rngs(keys)
    key3 = rngs3.dropout()
    assert jnp.allclose(key2, key3)

  def test_fork_broadcast(self):
    rngs = nnx.Rngs(params=0, dropout=1)
    jax.random.key

    keys = rngs.fork()  # all broadcast

    assert keys['params'].key.shape == ()
    assert keys['dropout'].key.shape == ()
    assert jnp.allclose(keys['params'].key, jax.random.key(0))
    assert jnp.allclose(keys['dropout'].key, jax.random.key(1))

  def test_fork_split(self):
    rngs = nnx.Rngs(params=0, dropout=1)
    keys = rngs.fork(4)  # split all

    assert keys['params'].key.shape == (4,)
    assert keys['dropout'].key.shape == (4,)

  def test_fork_split_and_broadcast(self):
    rngs = nnx.Rngs(params=0, dropout=1)
    splits, broadcasts = rngs.fork(params=4, dropout=None)

    assert splits['params'].key.shape == (4,)
    assert broadcasts['dropout'].key.shape == ()

  def test_fork_filters(self):
    rngs = nnx.Rngs(params=0, dropout=1)
    splits, broadcasts = rngs.fork({'params': 4})

    assert splits['params'].key.shape == (4,)
    assert broadcasts['dropout'].key.shape == ()

  def test_fork_multidimensional_split(self):
    rngs = nnx.Rngs(params=0, dropout=1)
    keys = rngs.fork((4, None, 3))  # split all

    assert keys['params'].key.shape == (4, 1, 3)
    assert keys['dropout'].key.shape == (4, 1, 3)

  def test_fork_multidimensional_split_mixed(self):
    rngs = nnx.Rngs(params=0, dropout=1)
    splits, broadcasts = rngs.fork(params=(4, None, 3))  # split all

    assert splits['params'].key.shape == (4, 1, 3)
    assert broadcasts['dropout'].key.shape == ()

  def test_rng_stream_pytree(self):
    rngs = nnx.Rngs(params=0, dropout=1)
    stream = rngs.fork()['params']

    stream2 = jax.tree_map(lambda x: x, stream)

    assert stream.key is stream2.key
