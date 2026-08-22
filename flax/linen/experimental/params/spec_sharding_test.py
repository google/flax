# Copyright 2023 The Flax Authors.
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


from absl.testing import absltest
import fiddle as fdl
from fiddle.experimental import auto_config
from fiddle.experimental import dataclasses as fdl_dc
import jax
from flax import linen as nn
from jax import numpy as jnp
from flax.linen.experimental.params import spec_sharding
from flax.linen.experimental.params import param as ps


class SimpleLinear(nn.Module):
  features: int
  kernel: ps.Param = ps.param_schema(input_features=ps.Given, features=ps.Attr)

  @nn.compact
  def __call__(self, x):
    kernel = self.kernel(input_features=x.shape[-1])
    return x @ kernel


# Test both styles of default factory.
class FFN(nn.Module):
  first: SimpleLinear = fdl_dc.field(
      default_factory=auto_config.auto_config(lambda: SimpleLinear(features=10)))
  second: SimpleLinear = fdl_dc.field(default_factory=SimpleLinear,
                                      configurable_factory=True)

  def __call__(self, x):
    x = self.first(x)
    x = jax.nn.relu(x)
    x = self.second(x)
    return x


class TestSharding(spec_sharding.ShardingPolicy):
  """Shards model weights across a `model` axis."""


@TestSharding.policy_for(SimpleLinear)
def _(c: fdl.Config[SimpleLinear]):
  c.kernel.sharding.features = 'model'
  c.kernel.sharding.input_features = 'data'


@TestSharding.policy_for(FFN)
def _(c: fdl.Config[FFN]):
  # Make `second` inverse of `first` to ensure post-order traversal works
  # correctly.
  c.second.kernel.sharding.features = 'data'
  c.second.kernel.sharding.input_features = 'model'


class SpecShardingTest(absltest.TestCase):

  def test_simple_sharding(self):
    config = fdl.Config(FFN)
    TestSharding.shard(config)

    self.assertEqual('model', config.first.kernel.sharding.features)
    self.assertEqual('data', config.first.kernel.sharding.input_features)
    self.assertEqual('data', config.second.kernel.sharding.features)
    self.assertEqual('model', config.second.kernel.sharding.input_features)

    config.second.features = 1
    model = fdl.build(config)

    self.assertEqual(10, model.first.features)
    self.assertEqual(1, model.second.features)

    batch_size = 7
    variables = model.init(jax.random.PRNGKey(42), jnp.zeros((batch_size, 100)))
    params = variables['params']

    self.assertEqual((100, 10), params['first']['kernel']['w'].shape)
    self.assertEqual((10, 1), params['second']['kernel']['w'].shape)


if __name__ == '__main__':
  absltest.main()
