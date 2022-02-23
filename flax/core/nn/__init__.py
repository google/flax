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

"""Flax Neural Network api."""

# pylint: disable=g-multiple-import
# re-export commonly used modules and functions
from .attention import (dot_product_attention, multi_head_dot_product_attention)
from flax.linen import activation
from flax.linen import initializers
from flax.linen.activation import (celu, elu, gelu, glu, leaky_relu,
                                   log_sigmoid, log_softmax, relu, sigmoid,
                                   silu, soft_sign, softmax, softplus, swish,
                                   tanh)
from flax.linen.pooling import avg_pool, max_pool
from .linear import Embedding, conv, conv_transpose, dense, dense_general, embedding
from .normalization import batch_norm, group_norm, layer_norm
from .stochastic import dropout

# pylint: enable=g-multiple-import
