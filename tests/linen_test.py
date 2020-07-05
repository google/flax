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

"""Tests for flax.linen."""

from absl.testing import absltest

from flax import linen as nn

import jax
from jax import random
from jax.nn import initializers
import jax.numpy as jnp

import numpy as onp

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class DummyModule(nn.Module):

  def apply(self, x):
    bias = self.param('bias', x.shape, initializers.ones)
    return x + bias
