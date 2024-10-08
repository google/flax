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

import jax, jax.numpy as jnp
from jax import random

from flax import linen
from flax import nnx
from flax.nnx.nn import initializers

import numpy as np

from absl.testing import absltest

# @nnx.jit
# def run_layer(layer, inputs):
#     carry = layer.initialize_carry(None, inputs.shape)
#     carry, _ = layer(carry, inputs)
#     return carry


# @nnx.jit
# def run_model(model, inputs):
#     out = model(inputs)
#     return out


# if __name__ == "__main__":
#     rngs = rnglib.Rngs(0)
#     batch_size, seq_len, feature_size, hidden_size = 48, 32, 16, 64
#     x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, feature_size))
#     # layer = SimpleCell(
#     #     in_features=feature_size, hidden_features=hidden_size, rngs=rngs
#     # )
#     layer = LSTMCell(in_features=feature_size, hidden_features=hidden_size, rngs=rngs)
#     # layer = OptimizedLSTMCell(in_features=feature_size, hidden_features=hidden_size, rngs=rngs)
#     # layer = GRUCell(in_features=feature_size, hidden_features=hidden_size, rngs=rngs)
#     rnn = RNN(
#         layer,
#         time_major=False,
#         reverse=False,
#         keep_order=False,
#         unroll=1,
#         return_carry=False,
#     )
#     from timeit import timeit

#     output = run_model(rnn, x)
#     print(output.shape)

#     print(timeit(lambda: run_model(rnn, x), number=100))

#     bidirectional = Bidirectional(
#         forward_rnn=rnn, backward_rnn=rnn, time_major=False, return_carry=True
#     )
#     ((c1, h1), (c2, h2)), output = run_model(bidirectional, x)  # for lstm
#     print(output.shape)
#     print(c1.shape)

class TestLSTMCell(absltest.TestCase):
    def test_basic(self):
        module = nnx.LSTMCell(
            in_features=3,
            hidden_features=4,
            rngs=nnx.Rngs(0),
        )
        x = jnp.ones((2, 3))
        carry = module.initialize_carry(module.rngs, x.shape)
        new_carry, y = module(carry, x)
        self.assertEqual(y.shape, (2, 4))

    def test_lstm_sequence(self):
        """Test LSTMCell over a sequence of inputs."""
        module = nnx.LSTMCell(
            in_features=3,
            hidden_features=4,
            rngs=nnx.Rngs(0),
        )
        x = random.normal(random.PRNGKey(1), (5, 2, 3))  # seq_len, batch, feature
        carry = module.initialize_carry(module.rngs, x.shape[1:])
        outputs = []
        for t in range(x.shape[0]):
            carry, y = module(carry, x[t])
            outputs.append(y)
        outputs = jnp.stack(outputs)
        self.assertEqual(outputs.shape, (5, 2, 4))

    def test_lstm_with_different_dtypes(self):
        """Test LSTMCell with different data types."""
        module = nnx.LSTMCell(
            in_features=3,
            hidden_features=4,
            dtype=jnp.bfloat16,
            param_dtype=jnp.bfloat16,
            rngs=nnx.Rngs(0),
        )
        x = jnp.ones((2, 3), dtype=jnp.bfloat16)
        carry = module.initialize_carry(module.rngs, x.shape)
        new_carry, y = module(carry, x)
        self.assertEqual(y.dtype, jnp.bfloat16)
        self.assertEqual(y.shape, (2, 4))

    def test_lstm_with_custom_activations(self):
        """Test LSTMCell with custom activation functions."""
        module = nnx.LSTMCell(
            in_features=3,
            hidden_features=4,
            gate_fn=jax.nn.relu,
            activation_fn=jax.nn.elu,
            rngs=nnx.Rngs(0),
        )
        x = jnp.ones((1, 3))
        carry = module.initialize_carry(module.rngs, x.shape)
        new_carry, y = module(carry, x)
        self.assertEqual(y.shape, (1, 4))

    def test_lstm_initialize_carry(self):
        """Test the initialize_carry method."""
        module = nnx.LSTMCell(
            in_features=3,
            hidden_features=4,
            carry_init=initializers.ones,
            rngs=nnx.Rngs(0),
        )
        x_shape = (1, 3)
        carry = module.initialize_carry(module.rngs, x_shape)
        c, h = carry
        self.assertTrue(jnp.all(c == 1.0))
        self.assertTrue(jnp.all(h == 1.0))
        self.assertEqual(c.shape, (1, 4))
        self.assertEqual(h.shape, (1, 4))

    def test_lstm_with_variable_sequence_length(self):
        """Test LSTMCell with variable sequence lengths."""
        module = nnx.LSTMCell(
            in_features=3,
            hidden_features=4,
            rngs=nnx.Rngs(0)
        )

        # Simulate a batch with variable sequence lengths
        x = jnp.array([
            [[1, 2, 3], [4, 5, 6], [0, 0, 0]],  # Sequence length 2
            [[7, 8, 9], [10, 11, 12], [13, 14, 15]],  # Sequence length 3
        ])  # Shape: (batch_size=2, max_seq_length=3, features=3)

        seq_lengths = jnp.array([2, 3])  # Actual lengths for each sequence
        batch_size = x.shape[0]
        max_seq_length = x.shape[1]
        carry = module.initialize_carry(module.rngs, (batch_size, 3))
        outputs = []
        for t in range(max_seq_length):
            input_t = x[:, t, :]
            carry, y = module(carry, input_t)
            outputs.append(y)
        outputs = jnp.stack(outputs, axis=1)  # Shape: (batch_size, max_seq_length, hidden_features)

        # Zero out outputs beyond the actual sequence lengths
        mask = (jnp.arange(max_seq_length)[None, :] < seq_lengths[:, None])
        outputs = outputs * mask[:, :, None]
        self.assertEqual(outputs.shape, (2, 3, 4))

    def test_lstm_stateful(self):
        """Test that LSTMCell maintains state across calls."""
        module = nnx.LSTMCell(
            in_features=3,
            hidden_features=4,
            rngs=nnx.Rngs(0),
        )
        x1 = jnp.ones((1, 3))
        x2 = jnp.ones((1, 3)) * 2
        carry = module.initialize_carry(module.rngs, x1.shape)
        carry, y1 = module(carry, x1)
        carry, y2 = module(carry, x2)
        self.assertEqual(y1.shape, (1, 4))
        self.assertEqual(y2.shape, (1, 4))

    def test_lstm_equivalence_with_flax_linen(self):
        """Test that nnx.LSTMCell produces the same outputs as flax.linen.LSTMCell."""
        in_features = 3
        hidden_features = 4
        key = random.PRNGKey(42)
        x = random.normal(key, (1, in_features))

        # Initialize nnx.LSTMCell
        rngs_nnx = nnx.Rngs(0)
        module_nnx = nnx.LSTMCell(
            in_features=in_features,
            hidden_features=hidden_features,
            rngs=rngs_nnx,
        )
        carry_nnx = module_nnx.initialize_carry(rngs_nnx, x.shape)
        # Initialize flax.linen.LSTMCell
        module_linen = linen.LSTMCell(
            features=hidden_features,
        )
        carry_linen = module_linen.initialize_carry(random.PRNGKey(0), x.shape)
        variables_linen = module_linen.init(random.PRNGKey(1), carry_linen, x)

        # Copy parameters from flax.linen.LSTMCell to nnx.LSTMCell
        params_linen = variables_linen['params']
        # Map the parameters from linen to nnx
        # Assuming the parameter names and shapes are compatible
        # For a precise mapping, you might need to adjust parameter names
        # Get the parameters from nnx module
        nnx_params = module_nnx.__dict__

        # Map parameters from linen to nnx
        for gate in ['i', 'f', 'g', 'o']:
            # Input kernels (input to gate)
            if gate == 'f':
                nnx_layer = getattr(module_nnx, f'if_')
            else:
                nnx_layer = getattr(module_nnx, f'i{gate}')
            linen_params = params_linen[f'i{gate}']
            nnx_layer.kernel.value = linen_params['kernel']
            if nnx_layer.use_bias:
                nnx_layer.bias.value = linen_params['bias']
            # Hidden kernels (hidden state to gate)
            nnx_layer = getattr(module_nnx, f'h{gate}')
            linen_params = params_linen[f'h{gate}']
            nnx_layer.kernel.value = linen_params['kernel']
            if nnx_layer.use_bias:
                nnx_layer.bias.value = linen_params['bias']

        # Run both modules
        new_carry_nnx, y_nnx = module_nnx(carry_nnx, x)
        new_carry_linen, y_linen = module_linen.apply(variables_linen, carry_linen, x)

        # Compare outputs
        np.testing.assert_allclose(y_nnx, y_linen, atol=1e-5)
        # Compare carries
        for c_nnx, c_linen in zip(new_carry_nnx, new_carry_linen):
            np.testing.assert_allclose(c_nnx, c_linen, atol=1e-5)

if __name__ == '__main__':
  absltest.main()
