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

from flax.core import Scope, init, apply, nn

from jax import random

# batch norm is in nn/normalization.py

if __name__ == "__main__":
  x = random.normal(random.PRNGKey(0), (2, 3))
  y, params = init(nn.batch_norm)(random.PRNGKey(1), x)
  print(y)
  print(params)
