from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
import pytest
import numpy as np

class TestTechnicalSupport:
  def test_mha_preferred_element_type(self):
    rngs = nnx.Rngs(0)
    # Testing that it doesn't crash and respects preferred_element_type
    model = nnx.MultiHeadAttention(
        num_heads=2,
        in_features=4,
        preferred_element_type=jnp.float16,
        decode=False,
        rngs=rngs,
    )
    x = jnp.ones((1, 3, 4), dtype=jnp.float32)
    y = model(x)
    assert y.shape == (1, 3, 4)
    assert model.preferred_element_type == jnp.float16

  def test_mha_out_sharding_signature(self):
    rngs = nnx.Rngs(0)
    model = nnx.MultiHeadAttention(
        num_heads=2,
        in_features=4,
        decode=False,
        rngs=rngs,
    )
    x = jnp.ones((1, 3, 4))
    # Just verify it accepts out_sharding=None without error
    y = model(x, out_sharding=None)
    assert y.shape == (1, 3, 4)

  def test_lstm_preferred_element_type(self):
    rngs = nnx.Rngs(0)
    model = nnx.LSTMCell(
        in_features=4,
        hidden_features=4,
        preferred_element_type=jnp.float16,
        rngs=rngs,
    )
    carry = model.initialize_carry((1, 4), rngs=rngs)
    x = jnp.ones((1, 4), dtype=jnp.float32)
    (new_c, new_h), out = model(carry, x)
    assert out.shape == (1, 4)
    assert model.preferred_element_type == jnp.float16
    assert model.ii.preferred_element_type == jnp.float16

  def test_gru_preferred_element_type(self):
    rngs = nnx.Rngs(0)
    model = nnx.GRUCell(
        in_features=4,
        hidden_features=4,
        preferred_element_type=jnp.float16,
        rngs=rngs,
    )
    carry = model.initialize_carry((1, 4), rngs=rngs)
    x = jnp.ones((1, 4), dtype=jnp.float32)
    new_h, out = model(carry, x)
    assert out.shape == (1, 4)
    assert model.preferred_element_type == jnp.float16
    assert model.dense_i.preferred_element_type == jnp.float16

  def test_simple_cell_out_sharding_signature(self):
    rngs = nnx.Rngs(0)
    model = nnx.SimpleCell(
        in_features=4,
        hidden_features=4,
        rngs=rngs,
    )
    carry = model.initialize_carry((1, 4), rngs=rngs)
    x = jnp.ones((1, 4))
    # Just verify it accepts out_sharding=None without error
    new_h, out = model(carry, x, out_sharding=None)
    assert out.shape == (1, 4)
