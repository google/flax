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

"""The Flax Module system."""


# pylint: disable=g-multiple-import
# re-export commonly used modules and functions
from .activation import (PReLU, celu, elu, gelu, glu, leaky_relu, log_sigmoid,
                         log_softmax, relu, sigmoid, silu, soft_sign, softmax,
                         softplus, swish, tanh)
from .attention import (MultiHeadDotProductAttention, SelfAttention,
                        combine_masks, dot_product_attention,
                        dot_product_attention_weights, make_attention_mask,
                        make_causal_mask)
from .combinators import Sequential
from ..core import DenyList, FrozenDict, broadcast
from .initializers import ones, zeros
from .linear import Conv, ConvLocal, ConvTranspose, Dense, DenseGeneral, Embed
from .module import (Module, Variable, apply, compact,
                     disable_named_call, enable_named_call, init,
                     init_with_output, merge_param, nowrap, override_named_call)
from .normalization import BatchNorm, GroupNorm, LayerNorm
from .pooling import avg_pool, max_pool, pool
from .recurrent import ConvLSTM, GRUCell, LSTMCell, OptimizedLSTMCell
from .stochastic import Dropout
from .transforms import (checkpoint, custom_vjp, jit, jvp, map_variables,
                         named_call, remat, remat_scan, scan, vjp, vmap,
                         while_loop)

# pylint: enable=g-multiple-import
