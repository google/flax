from absl.testing import absltest
import jax
import jax.numpy as jnp
from flax.nnx import initializers

class InitializersTest(absltest.TestCase):
    def test_zeros_init_shape(self):
        shape = (2, 3)
        initializer = initializers.zeros_init()
        result = initializer(jax.random.PRNGKey(0), shape)
        self.assertEqual(result.shape, shape, "zeros_init did not produce the expected shape")

    def test_zeros_init_values(self):
        shape = (2, 3)
        initializer = initializers.zeros_init()
        result = initializer(jax.random.PRNGKey(0), shape)
        expected = jnp.zeros(shape)
        self.assertTrue(jnp.array_equal(result, expected), "zeros_init did not produce the expected values")

    def test_ones_init_shape(self):
        shape = (2, 3)
        initializer = initializers.ones_init()
        result = initializer(jax.random.PRNGKey(0), shape)
        self.assertEqual(result.shape, shape, "ones_init did not produce the expected shape")

    def test_ones_init_values(self):
        shape = (2, 3)
        initializer = initializers.ones_init()
        result = initializer(jax.random.PRNGKey(0), shape)
        expected = jnp.ones(shape)
        self.assertTrue(jnp.array_equal(result, expected), "ones_init did not produce the expected values")

if __name__ == '__main__':
    absltest.main()