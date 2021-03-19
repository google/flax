# Copyright 2021 The Flax Authors.
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


from flax import linen as nn
import jax
from jax import  random
import numpy as np

from typing import Sequence,List


# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()
# Require JAX omnistaging mode.
jax.config.enable_omnistaging()


class SequentialTest(absltest.TestCase):

	def test_sequential(self):

		class MLP(nn.Module):

			features: Sequence[int]

			def setup(self):
			    # we automatically know what to do with lists, dicts of submodules
				self.layers = [nn.Dense(feat) for feat in self.features]

			def __call__(self, inputs):
				x = inputs
				for i, lyr in enumerate(self.layers):
				  x = lyr(x)
				return x


		model1 = nn.Sequential([nn.Dense(3),nn.Dense(4), nn.Dense(5)])
		model2 = MLP(features=[3,4,5])

		key1, key2 = random.split(random.PRNGKey(0), 2)
		x = random.uniform(key1, (4,4))

		init_variables1 = model1.init(key2, x)
		init_variables2 = model2.init(key2, x)

		y1 = model1.apply(init_variables1, x)
		y2 = model2.apply(init_variables2, x)

		self.assertTrue(np.all(y1 == y2))

if __name__ == '__main__':
  absltest.main()

