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
from .activation import (
  PReLU as PReLU,
  celu as celu,
  elu as elu,
  gelu as gelu,
  glu as glu,
  leaky_relu as leaky_relu,
  log_sigmoid as log_sigmoid,
  log_softmax as log_softmax,
  relu as relu,
  sigmoid as sigmoid,
  silu as silu,
  soft_sign as soft_sign,
  softmax as softmax,
  softplus as softplus,
  swish as swish,
  tanh as tanh
)
from .attention import (
  MultiHeadDotProductAttention as MultiHeadDotProductAttention,
  SelfAttention as SelfAttention,
  combine_masks as combine_masks,
  dot_product_attention as dot_product_attention,
  dot_product_attention_weights as dot_product_attention_weights,
  make_attention_mask as make_attention_mask,
  make_causal_mask as make_causal_mask
)
from .combinators import Sequential as Sequential
from ..core import (
  DenyList as DenyList,
  FrozenDict as FrozenDict,
  broadcast as broadcast
)
from .initializers import (
  ones as ones,
  zeros as zeros
)
from .linear import (
  Conv as Conv,
  ConvLocal as ConvLocal,
  ConvTranspose as ConvTranspose,
  Dense as Dense,
  DenseGeneral as DenseGeneral,
  Embed as Embed
)
from .module import (
  Module as Module,
  Variable as Variable,
  apply as apply,
  compact as compact,
  disable_named_call as disable_named_call,
  enable_named_call as enable_named_call,
  init as init,
  init_with_output as init_with_output,
  merge_param as merge_param,
  nowrap as nowrap,
  override_named_call as override_named_call
)
from .normalization import (
  BatchNorm as BatchNorm,
  GroupNorm as GroupNorm,
  LayerNorm as LayerNorm
)
from .pooling import (
  avg_pool as avg_pool,
  max_pool as max_pool,
  pool as pool
)
from .recurrent import (
  ConvLSTM as ConvLSTM,
  GRUCell as GRUCell,
  LSTMCell as LSTMCell,
  OptimizedLSTMCell as OptimizedLSTMCell
)
from .stochastic import Dropout as Dropout
from .transforms import (
  checkpoint as checkpoint,
  custom_vjp as custom_vjp,
  jit as jit,
  jvp as jvp,
  map_variables as map_variables,
  named_call as named_call,
  remat as remat,
  remat_scan as remat_scan,
  scan as scan,
  vjp as vjp,
  vmap as vmap,
  while_loop as while_loop,
  cond as cond
)
# pylint: enable=g-multiple-import
