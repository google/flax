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
import re

from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import jax_utils
import jax
import jax.numpy as jnp
import numpy as np

NDEV = 4

_xla_device_count_flag_regexp = (
  r'[-]{0,2}xla_force_host_platform_device_count=(\d+)?(\s|$)'
)


def set_n_cpu_devices(n: int):
  xla_flags = os.getenv('XLA_FLAGS', '')
  xla_flags = re.sub(_xla_device_count_flag_regexp, '', xla_flags)
  os.environ['XLA_FLAGS'] = ' '.join(
    [f'--xla_force_host_platform_device_count={n}'] + xla_flags.split()
  )


def setUpModule():
  set_n_cpu_devices(NDEV)


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


if __name__ == '__main__':
  absltest.main()
