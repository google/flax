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
import numpy as np
from absl.testing import absltest

from flax import nnx
from flax import errors


class TestRngs(absltest.TestCase):
  def test_call(self):
    rngs = nnx.Rngs(0)
    key = rngs()

  def test_fallback(self):
    rngs = nnx.Rngs(0)
    key = rngs.dropout()

  def test_fallback_error_no_default(self):
    rngs = nnx.Rngs(some_name=0)
    with self.assertRaisesRegex(AttributeError, 'No RngStream named'):
      key = rngs.dropout()

  def test_rng_stream(self):
    key0 = jax.random.key(0)
    rngs = nnx.Rngs(params=key0)
    self.assertEqual(rngs.params.count[...], 0)

    key1 = rngs.params()
    self.assertEqual(rngs.params.count[...], 1)
    self.assertIs(rngs.params.key[...], key0)
    self.assertFalse(jnp.allclose(key0, key1))

    key2 = rngs.params()
    self.assertEqual(rngs.params.count[...], 2)
    self.assertIs(rngs.params.key[...], key0)
    self.assertFalse(jnp.allclose(key1, key2))

  def test_rng_trace_level_constraints(self):
    rngs = nnx.Rngs(0)

    @jax.jit
    def f():
      with self.assertRaisesRegex(
        errors.TraceContextError,
        'Cannot mutate RngCount from a different trace level',
      ):
        rngs.params()

    f()

    rngs1: Any = None

    @jax.jit
    def h():
      nonlocal rngs1
      rngs1 = nnx.Rngs(1)

    h()

    self.assertIsInstance(rngs1, nnx.Rngs)
    with self.assertRaisesRegex(
      errors.TraceContextError,
      'Cannot mutate RngCount from a different trace level',
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
    self.assertEqual(rngs['default'].count[...], 2)

    @nnx.jit
    def f(m: Foo, x: jax.Array, not_rngs: nnx.Rngs):
      rngs = not_rngs
      x = m(x, rngs)
      x = m(x, rngs)
      return x

    x = jnp.ones((2, 2))
    x = f(m, x, rngs)

    # +1 for the Dropout mask
    self.assertEqual(rngs['default'].count[...], 4)

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
    graphdef, params, rng_counts, dropout_keys, param_keys = nnx.split(
      m, nnx.Param, nnx.RngCount, 'dropout', 'params'
    )

    self.assertEqual(m.rngs.params.count[...], 2)
    self.assertEqual(m.rngs['dropout'].count[...], 0)
    self.assertLen(nnx.to_flat_state(dropout_keys), 1)
    self.assertLen(nnx.to_flat_state(param_keys), 1)
    self.assertLen(nnx.to_flat_state(rng_counts), 2)

    # split dropout keys
    split_dropout_keys = jax.tree.map(
      lambda x: jax.random.split(x, 4), dropout_keys
    )
    # replicate params
    params = jax.tree.map(lambda x: jnp.stack([x] * 4, axis=0), params)

    @partial(
      jax.vmap,
      in_axes=(0, 0, None, None, 0),
      out_axes=(0, 0, None),
    )
    def f(params, dropout_keys, param_keys, rng_counts, x):
      m = nnx.merge(graphdef, params, dropout_keys, param_keys, rng_counts)
      y = m(x)
      _, params, rng_counts, dropout_keys, param_keys = nnx.split(
        m, nnx.Param, nnx.RngCount, 'dropout', 'params'
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

    self.assertEqual(y.shape, (4, 1, 3))
    self.assertEqual(m.rngs.params.count[...], 2)
    self.assertEqual(m.rngs['dropout'].count[...], 1)

  def test_reseed(self):
    class Model(nnx.Module):
      def __init__(self, rngs):
        self.linear = nnx.Linear(2, 3, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

      def __call__(self, x):
        return self.dropout(self.linear(x))

    model = Model(nnx.Rngs(params=0, dropout=42))
    x = jnp.ones((1, 2))

    y1 = model(x)

    # reset the ``dropout`` stream key to 42
    nnx.reseed(model, dropout=42)
    y2 = model(x)

    np.testing.assert_allclose(y1, y2)

  def test_random_helpers(self):
    rngs = nnx.Rngs(0, params=1)

    x_nnx = rngs.normal((2, 3))
    x_jax = jax.random.normal(jax.random.fold_in(jax.random.key(0), 0), (2, 3))
    np.testing.assert_allclose(x_nnx, x_jax)

    x_nnx = rngs.params.uniform((2, 3))
    x_jax = jax.random.uniform(jax.random.fold_in(jax.random.key(1), 0), (2, 3))
    np.testing.assert_allclose(x_nnx, x_jax)

    x_nnx = rngs.lecun_normal()((2, 3))
    x_jax = jax.nn.initializers.lecun_normal()(
      jax.random.fold_in(jax.random.key(0), 1), (2, 3)
    )
    np.testing.assert_allclose(x_nnx, x_jax)

    x_nnx = rngs.params.lecun_uniform()((2, 3))
    x_jax = jax.nn.initializers.lecun_uniform()(
      jax.random.fold_in(jax.random.key(1), 1), (2, 3)
    )
    np.testing.assert_allclose(x_nnx, x_jax)

if __name__ == '__main__':
  absltest.main()
