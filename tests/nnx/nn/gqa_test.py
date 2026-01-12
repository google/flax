import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np

class TestGQA:
    def test_gqa_shapes(self):
        B, T, S = 2, 4, 5
        D = 8
        num_heads_q = 6
        num_heads_kv = 3

        k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
        query = jax.random.normal(k1, (B, T, num_heads_q, D))
        key   = jax.random.normal(k2, (B, S, num_heads_kv, D))
        value = jax.random.normal(k3, (B, S, num_heads_kv, D))

        output = nnx.dot_product_attention(query, key, value)
        expected_shape = (B, T, num_heads_q, D)
        assert output.shape == expected_shape

    def test_gqa_invalid_heads(self):
        B, T, D = 1, 4, 8
        query = jnp.ones((B, T, 5, D))
        key   = jnp.ones((B, T, 2, D))
        value = key

        try:
            nnx.dot_product_attention(query, key, value)
            assert False, "Should have raised ValueError"
        except ValueError as e:

            assert "must be a multiple" in str(e)

    def test_gqa_parity_with_jax(self):
        class DummyModule(nnx.Module):
            pass

        dummy_module = DummyModule()

        B, T, S, D = 2, 8, 8, 16
        num_heads_q = 4
        num_heads_kv = 2

        rng = jax.random.key(42)
        k1, k2, k3 = jax.random.split(rng, 3)

        query = jax.random.normal(k1, (B, T, num_heads_q, D))
        key   = jax.random.normal(k2, (B, S, num_heads_kv, D))
        value = jax.random.normal(k3, (B, S, num_heads_kv, D))

        jax_out = jax.nn.dot_product_attention(query, key, value)

        # NNX should handle broadcasting internally
        nnx_out = nnx.dot_product_attention(
            query, key, value,
            module=dummy_module
        )

        np.testing.assert_allclose(nnx_out, jax_out, atol=1e-3, rtol=1e-3)