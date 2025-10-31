import unittest
from flax.pooling import pool,avg_pool, max_pool, min_pool
import numpy as np
import jax.numpy as jnp
from absl.testing import absltest, parameterized
import jax

jax.config.parse_flags_with_absl()


class PoolTest(parameterized.TestCase):
  def test_pool_custom_reduce(self):
    x = jnp.full((1, 3, 3, 1), 2.0)
    mul_reduce = lambda x, y: x * y
    y = pool(x, 1.0, mul_reduce, (2, 2), (1, 1), 'VALID')
    np.testing.assert_allclose(y, np.full((1, 2, 2, 1), 2.0**4))

  @parameterized.parameters(
    {'count_include_pad': True}, {'count_include_pad': False}
  )
  def test_avg_pool(self, count_include_pad):
    x = jnp.full((1, 3, 3, 1), 2.0)
    pool = lambda x: avg_pool(x, (2, 2), count_include_pad=count_include_pad)
    y = pool(x)
    np.testing.assert_allclose(y, np.full((1, 2, 2, 1), 2.0))
    y_grad = jax.grad(lambda x: pool(x).sum())(x)
    expected_grad = jnp.array(
      [
        [0.25, 0.5, 0.25],
        [0.5, 1.0, 0.5],
        [0.25, 0.5, 0.25],
      ]
    ).reshape((1, 3, 3, 1))
    np.testing.assert_allclose(y_grad, expected_grad)

  @parameterized.parameters(
    {'count_include_pad': True}, {'count_include_pad': False}
  )
  def test_avg_pool_no_batch(self, count_include_pad):
    x = jnp.full((3, 3, 1), 2.0)
    pool = lambda x: avg_pool(x, (2, 2), count_include_pad=count_include_pad)
    y = pool(x)
    np.testing.assert_allclose(y, np.full((2, 2, 1), 2.0))
    y_grad = jax.grad(lambda x: pool(x).sum())(x)
    expected_grad = jnp.array(
      [
        [0.25, 0.5, 0.25],
        [0.5, 1.0, 0.5],
        [0.25, 0.5, 0.25],
      ]
    ).reshape((3, 3, 1))
    np.testing.assert_allclose(y_grad, expected_grad)

  def test_max_pool(self):
    x = jnp.arange(9).reshape((1, 3, 3, 1)).astype(jnp.float32)
    pool = lambda x: max_pool(x, (2, 2))
    expected_y = jnp.array(
      [
        [4.0, 5.0],
        [7.0, 8.0],
      ]
    ).reshape((1, 2, 2, 1))
    y = pool(x)
    np.testing.assert_allclose(y, expected_y)
    y_grad = jax.grad(lambda x: pool(x).sum())(x)
    expected_grad = jnp.array(
      [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
      ]
    ).reshape((1, 3, 3, 1))
    np.testing.assert_allclose(y_grad, expected_grad)

  @parameterized.parameters(
    {'count_include_pad': True}, {'count_include_pad': False}
  )
  def test_avg_pool_padding_same(self, count_include_pad):
    x = jnp.array([1.0, 2.0, 3.0, 4.0]).reshape((1, 2, 2, 1))
    pool = lambda x: avg_pool(
      x, (2, 2), padding='SAME', count_include_pad=count_include_pad
    )
    y = pool(x)
    if count_include_pad:
      expected_y = jnp.array([10.0 / 4, 6.0 / 4, 7.0 / 4, 4.0 / 4]).reshape(
        (1, 2, 2, 1)
      )
    else:
      expected_y = jnp.array([10.0 / 4, 6.0 / 2, 7.0 / 2, 4.0 / 1]).reshape(
        (1, 2, 2, 1)
      )
    np.testing.assert_allclose(y, expected_y)

  def test_pooling_variable_batch_dims(self):
    x = jnp.zeros((1, 8, 32, 32, 3), dtype=jnp.float32)
    y = max_pool(x, (2, 2), (2, 2))

    assert y.shape == (1, 8, 16, 16, 3)

  def test_pooling_no_batch_dims(self):
    x = jnp.zeros((32, 32, 3), dtype=jnp.float32)
    y = max_pool(x, (2, 2), (2, 2))

    assert y.shape == (16, 16, 3)

if __name__ == '__main__':
    unittest.main()
