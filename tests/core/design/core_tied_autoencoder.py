# Copyright 2020 The Flax Authors.
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


from flax.core import Scope, Array, init, unfreeze, lift, nn


def mlp(scope: Scope, x: Array, hidden: int, out: int):
  x = scope.child(nn.dense, 'hidden')(x, hidden)
  x = nn.relu(x)
  return scope.child(nn.dense, 'out')(x, out)


@dataclass
class TiedAutoEncoder:

  latents: int
  features: int

  def __call__(self, scope, x):
    z = self.encode(scope, x)
    return self.decode(scope, z)

  def encode(self, scope, x):
    assert x.shape[-1] == self.features
    return self._tied(nn.dense)(scope, x, self.latents, bias=False)

  def decode(self, scope, z):
    assert z.shape[-1] == self.latents
    return self._tied(nn.dense, transpose=True)(
        scope, z, self.features, bias=False)

  def _tied(self, fn, transpose=False):
    if not transpose:
      return fn

    def trans(variables):
      if 'params' not in variables:
        return variables
      params = variables['params']
      params['kernel'] = params['kernel'].T
      return variables

    return lift.transform_module(
        fn, trans_in_fn=trans, trans_out_fn=trans)


class TiedAutoEncoderTest(absltest.TestCase):

  def test_tied_auto_encoder(self):
    ae = TiedAutoEncoder(latents=2, features=4)
    x = jnp.ones((1, ae.features))
    x_r, variables = init(ae)(random.PRNGKey(0), x)

    param_shapes = unfreeze(
        jax.tree_map(jnp.shape, variables['params']))
    self.assertEqual(param_shapes, {
        'kernel': (4, 2),
    })
    self.assertEqual(x.shape, x_r.shape)

  def test_init_from_decoder(self):
    ae = TiedAutoEncoder(latents=2, features=4)
    z = jnp.ones((1, ae.latents))
    x_r, variables = init(ae.decode)(random.PRNGKey(0), z)

    param_shapes = unfreeze(
        jax.tree_map(jnp.shape, variables['params']))
    self.assertEqual(param_shapes, {
        'kernel': (4, 2),
    })
    self.assertEqual(x_r.shape, (1, 4))
