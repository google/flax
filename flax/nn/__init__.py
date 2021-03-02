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

"""Flax Neural Network api."""

# pylint: disable=g-multiple-import
# re-export commonly used modules and functions
from .activation import (celu, elu, gelu, glu, leaky_relu, log_sigmoid,
                         log_softmax, relu, sigmoid, soft_sign, softmax,
                         softplus, swish, silu, tanh)
from .attention import (dot_product_attention, MultiHeadDotProductAttention,
                        SelfAttention)
from .base import (Module, Model, Collection, capture_module_outputs,
                   module, stateful, get_state, module_method)
from .linear import Dense, DenseGeneral, Conv, ConvTranspose, Embed
from .normalization import BatchNorm, LayerNorm, GroupNorm
from .pooling import max_pool, avg_pool
from .recurrent import LSTMCell, GRUCell, ConvLSTM, OptimizedLSTMCell
from .stochastic import make_rng, stochastic, dropout, is_stochastic
# pylint: enable=g-multiple-import
import warnings
# Makes sure the user sees the warning, as deprecation warnings are silent by default
warnings.filterwarnings("default", category=DeprecationWarning, module=__name__)
