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

"""Tests for flax.jax_utils."""

from functools import partial
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import jax_utils
import jax
import jax.numpy as jnp
import numpy as np

NDEV = 4


class PadShardUnpadTest(chex.TestCase):
  BATCH_SIZES = [NDEV, NDEV + 1, NDEV - 1, 5 * NDEV, 5 * NDEV + 1, 5 * NDEV - 1]
  DTYPES = [np.float32, np.uint8, jax.numpy.bfloat16, np.int32]

  def tearDown(self):
    chex.clear_trace_counter()
    super().tearDown()

  @parameterized.product(dtype=DTYPES, bs=BATCH_SIZES)
  def test_basics(self, dtype, bs):
    # Just tests that basic calling works without exploring caveats.
    @partial(jax_utils.pad_shard_unpad, static_argnums=())
    def add(a, b):
      b = jnp.asarray(b, dtype=dtype)
      return a + b

    x = np.arange(bs, dtype=dtype)
    y = add(x, 10 * x)
    chex.assert_type(y.dtype, x.dtype)
    np.testing.assert_allclose(np.float64(y), np.float64(x + 10 * x))

  @parameterized.product(dtype=DTYPES, bs=BATCH_SIZES)
  def test_trees(self, dtype, bs):
    # Just tests that basic calling works without exploring caveats.
    @partial(jax_utils.pad_shard_unpad, static_argnums=())
    def add(a, b):
      return a['a'] + b[0]

    x = jnp.arange(bs, dtype=dtype)
    y = add(dict(a=x), (10 * x,))
    chex.assert_type(y.dtype, x.dtype)
    np.testing.assert_allclose(np.float64(y), np.float64(x + 10 * x))

  @parameterized.parameters(DTYPES)
  def test_min_device_batch_avoids_recompile(self, dtype):
    @partial(jax_utils.pad_shard_unpad, static_argnums=())
    @jax.jit
    @chex.assert_max_traces(n=1)
    def add(a, b):
      b = jnp.asarray(b, dtype=dtype)
      return a + b

    chex.clear_trace_counter()

    for bs in self.BATCH_SIZES:
      x = jnp.arange(bs, dtype=dtype)
      y = add(x, 10 * x, min_device_batch=9)  # pylint: disable=unexpected-keyword-arg
      chex.assert_type(y.dtype, x.dtype)
      np.testing.assert_allclose(np.float64(y), np.float64(x + 10 * x))

  @parameterized.product(dtype=DTYPES, bs=BATCH_SIZES)
  def test_static_argnum(self, dtype, bs):
    @partial(jax_utils.pad_shard_unpad, static_argnums=(1,))
    def add(a, b):
      return a + jnp.asarray(b, dtype=dtype)

    x = jnp.arange(bs, dtype=dtype)
    y = add(x, 10)
    chex.assert_type(y.dtype, x.dtype)
    np.testing.assert_allclose(np.float64(y), np.float64(x + 10))

  @parameterized.product(dtype=DTYPES, bs=BATCH_SIZES)
  def test_static_argnames(self, dtype, bs):
    # In this test, leave static_argnums at the default value too, in order to
    # test the default/most canonical path where `params` are the first arg.
    @partial(jax_utils.pad_shard_unpad, static_argnames=('b',))
    def add(params, a, *, b):
      params = jnp.asarray(params, dtype=dtype)
      b = jnp.asarray(b, dtype=dtype)
      return params * a + b

    x = jnp.arange(bs, dtype=dtype)
    y = add(5, x, b=10)
    chex.assert_type(y.dtype, x.dtype)
    np.testing.assert_allclose(np.float64(y), np.float64(5 * x + 10))


class DataShardingTest(parameterized.TestCase):
  def setUp(self):
    if jax.device_count() < 4:
      self.skipTest('At least 4 devices required')

  @parameterized.product(num_devices= ["all", 2])
  def test_prefetch_to_device(self, num_devices):
    devices = jax.local_devices()
    if isinstance(num_devices, int):
      devices = devices[:num_devices]
    shape = (len(devices), 4, 16, 16, 3)
    iterator = (jnp.ones(shape) for _ in range(4))

    data_iter = jax_utils.prefetch_to_device(iterator, size=3, devices=devices)
    for _ in range(4):
      data = next(data_iter)
      self.assertEqual(data.shape, shape)
      self.assertIsNotNone(data.sharding)
      sharding_slices_per_device = data.sharding.devices_indices_map(tuple(data.shape))
      self.assertEqual(len(sharding_slices_per_device), len(devices))
      # Here we check that sharding_slices_per_device is like
      # Device(id=2): (slice(2, 3, None), slice(None, None, None), ..., slice(None, None, None))
      for i, dev in enumerate(devices):
        sharding_slice = sharding_slices_per_device[dev]
        self.assertEqual(sharding_slice[0], slice(i + 0, i + 1, None))
        for sharding_slice_j in sharding_slice[1:]:
          self.assertEqual(sharding_slice_j, slice(None, None, None))

  @parameterized.product(num_devices= ["all", 2])
  def test_replicate(self, num_devices):
    devices = jax.local_devices()
    if isinstance(num_devices, int):
      devices = devices[:num_devices]
    num_batches = 5
    shape = (2, 3)
    data_tree = [
      i * jnp.ones((2, 3)) for i in range(num_batches - 2)
    ] + [4, 5 * np.ones(shape)]
    out_tree = jax_utils.replicate(data_tree, devices=devices)

    def check_sharding(p):
      if p.ndim == 1:
        self.assertEqual(p.shape, (len(devices),))
      else:
        self.assertEqual(p.shape, (len(devices), *shape))
      self.assertIsNotNone(p.sharding)
      sharding_slices_per_device = p.sharding.devices_indices_map(tuple(p.shape))
      self.assertEqual(len(sharding_slices_per_device), len(devices))
      # Here we check that sharding_slices_per_device is like
      # Device(id=2): (slice(2, 3, None), slice(None, None, None), slice(None, None, None))
      for i, dev in enumerate(devices):
        sharding_slice = sharding_slices_per_device[dev]
        self.assertEqual(sharding_slice[0], slice(i + 0, i + 1, None))
        for sharding_slice_j in sharding_slice[1:]:
          self.assertEqual(sharding_slice_j, slice(None, None, None))

    jax.tree.map(check_sharding, out_tree)


if __name__ == '__main__':
  absltest.main()
