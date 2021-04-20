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

"""The Flax Module system."""


# pylint: disable=g-multiple-import
# re-export commonly used modules and functions
from .activation import (celu, elu, gelu, glu, leaky_relu, log_sigmoid,
                         log_softmax, relu, sigmoid, soft_sign, softmax,
                         softplus, swish, silu, tanh)
from .attention import (MultiHeadDotProductAttention, SelfAttention,
                        dot_product_attention, make_attention_mask,
                        make_causal_mask, combine_masks)
from ..core import broadcast, DenyList
from .linear import Conv, ConvTranspose, Dense, DenseGeneral, Embed
from .module import Module, compact, enable_named_call, disable_named_call, Variable, init, init_with_output, apply
from .normalization import BatchNorm, GroupNorm, LayerNorm
from .pooling import avg_pool, max_pool
from .recurrent import GRUCell, LSTMCell, ConvLSTM, OptimizedLSTMCell
from .stochastic import Dropout
from .transforms import jit, named_call, remat, scan, vmap
from .initializers import zeros, ones

# pylint: enable=g-multiple-import
