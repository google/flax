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


# pylint: disable=g-multiple-import,useless-import-alias
# re-export commonly used modules and functions
from .activation import (
  PReLU as PReLU,
  celu as celu,
  elu as elu,
  gelu as gelu,
  glu as glu,
  hard_sigmoid as hard_sigmoid,
  hard_silu as hard_silu,
  hard_swish as hard_swish,
  hard_tanh as hard_tanh,
  leaky_relu as leaky_relu,
  log_sigmoid as log_sigmoid,
  log_softmax as log_softmax,
  logsumexp as logsumexp,
  normalize as normalize,
  one_hot as one_hot,
  relu as relu,
  relu6 as relu6,
  selu as selu,
  sigmoid as sigmoid,
  silu as silu,
  soft_sign as soft_sign,
  softmax as softmax,
  softplus as softplus,
  standardize as standardize,
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
  broadcast as broadcast,
  meta as meta,
)
from ..core.meta import (
    Partitioned as Partitioned,
    with_partitioning as with_partitioning,
    get_partition_spec as get_partition_spec,
    unbox as unbox,
    PARTITION_NAME as PARTITION_NAME,
)
from .spmd import (
    logical_axis_rules as logical_axis_rules,
    set_logical_axis_rules as set_logical_axis_rules,
    get_logical_axis_rules as get_logical_axis_rules,
    logical_to_mesh_axes,
    logical_to_mesh,
    with_logical_constraint,
    LogicallyPartitioned as LogicallyPartitioned,
    with_logical_partitioning as with_logical_partitioning,
)
from .initializers import (
  ones as ones,
  ones_init as ones_init,
  zeros as zeros,
  zeros_init as zeros_init
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
  ConvLSTMCell as ConvLSTMCell,
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
  cond as cond,
  switch as switch,
  add_metadata_axis,
)
from .summary import tabulate
# pylint: enable=g-multiple-import
