# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for helpers."""

from __future__ import annotations


from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import flax.linen as nn
import helpers
import jax
import jax.numpy as jnp
import numpy as np


class ModuleFromLinenVariablesTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          inputs_shape=(1, 2, 3, 4),
          num_features=10,
          use_bias=True,
      ),
      dict(
          inputs_shape=(10, 5),
          num_features=4,
          use_bias=False,
      ),
  )
  def test_same_structure(self, inputs_shape, num_features, use_bias):
    rng_key = jax.random.PRNGKey(0)
    rng_inputs, rng_params = jax.random.split(rng_key)
    inputs = jax.random.normal(rng_inputs, inputs_shape)

    linen_mdl = nn.Dense(
        features=num_features,
        use_bias=use_bias,
    )
    linen_init_vars = linen_mdl.init(rng_params, jnp.zeros(inputs_shape))
    linen_output = linen_mdl.apply(
        linen_init_vars,
        inputs,
    )

    mdl = helpers.module_from_linen_variables(
        module_factory=lambda: nnx.Linear(
            in_features=inputs_shape[-1],
            out_features=num_features,
            use_bias=use_bias,
            rngs=nnx.Rngs(params=rng_params),
        ),
        variables=linen_init_vars,
    )
    output = mdl(inputs)

    np.testing.assert_array_equal(output, linen_output)

  @parameterized.parameters(
      dict(
          inputs_shape=(1, 2, 3, 4),
          num_features=(10, 20, 7),
          use_bias=(False, True, False),
      ),
  )
  def test_different_structure(self, inputs_shape, num_features, use_bias):
    rng_key = jax.random.PRNGKey(0)
    rng_inputs, rng_params = jax.random.split(rng_key)
    inputs = jax.random.normal(rng_inputs, inputs_shape)

    linen_mdl = nn.Sequential([
        nn.Sequential([
            nn.BatchNorm(use_running_average=False),
            nn.Dense(features=f, use_bias=b),
        ])
        for f, b in zip(num_features, use_bias)
    ])
    linen_init_vars = linen_mdl.init(rng_key, jnp.zeros(inputs_shape))
    linen_output, linen_vars = linen_mdl.apply(
        linen_init_vars,
        inputs,
        mutable=['batch_stats'],
    )

    module_factory = lambda: nnx.Sequential(*[
        nnx.Sequential(
            nnx.BatchNorm(
                num_features=in_f,
                use_running_average=False,
                rngs=nnx.Rngs(params=rng_params),
            ),
            nnx.Linear(
                in_features=in_f,
                out_features=out_f,
                use_bias=b,
                rngs=nnx.Rngs(params=rng_params),
            ),
        )
        for in_f, out_f, b in zip(in_features, out_features, use_bias)
    ])

    def _map_key_fn(key: tuple[str, ...]) -> tuple[str | int, ...]:
      new_key = []
      for k in key[1:]:
        if k.startswith('layers_'):
          prefix, suffix = k.split('layers_')
          assert not prefix, prefix
          new_key.append('layers')
          new_key.append(int(suffix))
        else:
          new_key.append(k)

      return tuple(new_key)

    in_features = (inputs_shape[-1], *num_features[:-1])
    out_features = num_features
    mdl = helpers.module_from_linen_variables(
        module_factory=module_factory,
        variables=linen_init_vars,
        map_key_fn=_map_key_fn,
    )
    output = mdl(inputs)

    np.testing.assert_array_equal(output, linen_output)
    for i in range(len(num_features)):
      np.testing.assert_array_equal(
          mdl.layers[i].layers[0].mean.value,
          linen_vars['batch_stats'][f'layers_{i}']['layers_0']['mean'],
      )
      np.testing.assert_array_equal(
          mdl.layers[i].layers[0].var.value,
          linen_vars['batch_stats'][f'layers_{i}']['layers_0']['var'],
      )


if __name__ == '__main__':
  absltest.main()
