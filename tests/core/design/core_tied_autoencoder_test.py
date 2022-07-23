# Copyright 2022 The Flax Authors.
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

from dataclasses import dataclass

from absl.testing import absltest

from jax import numpy as jnp, random
import jax


from flax.core import init, unfreeze, lift, nn


def transpose(fn):
  def trans(variables):
    return jax.tree_util.tree_map(lambda x: x.T, variables)

  return lift.map_variables(
      fn, "params", map_in_fn=trans, map_out_fn=trans,
      mutable=True)


@dataclass
class TiedAutoEncoder:

  latents: int
  features: int

  def __call__(self, scope, x):
    z = self.encode(scope, x)
    return self.decode(scope, z)

  def encode(self, scope, x):
    return nn.dense(scope, x, self.latents, bias=False)

  def decode(self, scope, z):
    return transpose(nn.dense)(
        scope, z, self.features, bias=False)


class TiedAutoEncoderTest(absltest.TestCase):

  def test_tied_auto_encoder(self):
    ae = TiedAutoEncoder(latents=2, features=4)
    x = jnp.ones((1, ae.features))
    x_r, variables = init(ae)(random.PRNGKey(0), x)

    param_shapes = unfreeze(
        jax.tree_util.tree_map(jnp.shape, variables['params']))
    self.assertEqual(param_shapes, {
        'kernel': (4, 2),
    })
    self.assertEqual(x.shape, x_r.shape)

  def test_init_from_decoder(self):
    ae = TiedAutoEncoder(latents=2, features=4)
    z = jnp.ones((1, ae.latents))
    x_r, variables = init(ae.decode)(random.PRNGKey(0), z)

    param_shapes = unfreeze(
        jax.tree_util.tree_map(jnp.shape, variables['params']))
    self.assertEqual(param_shapes, {
        'kernel': (4, 2),
    })
    self.assertEqual(x_r.shape, (1, 4))


if __name__ == '__main__':
  absltest.main()
