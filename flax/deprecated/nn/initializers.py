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

"""Initializers for Flax.
"""

# pylint: disable=unused-import
# re-export initializer functions from jax.nn
from jax.nn.initializers import kaiming_normal
from jax.nn.initializers import kaiming_uniform
from jax.nn.initializers import lecun_normal
from jax.nn.initializers import lecun_uniform
from jax.nn.initializers import normal
from jax.nn.initializers import ones
from jax.nn.initializers import orthogonal
from jax.nn.initializers import delta_orthogonal
from jax.nn.initializers import uniform
from jax.nn.initializers import variance_scaling
from jax.nn.initializers import xavier_normal
from jax.nn.initializers import xavier_uniform
from jax.nn.initializers import zeros
# pylint: enable=unused-import
