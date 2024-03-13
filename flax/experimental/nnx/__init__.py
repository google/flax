# Copyright 2024 The Flax Authors.
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

from flax.linen.pooling import avg_pool as avg_pool
from flax.linen.pooling import max_pool as max_pool
from flax.linen.pooling import min_pool as min_pool
from flax.linen.pooling import pool as pool
from flax.typing import Initializer as Initializer

from .nnx import compatibility as compatibility
from .nnx import graph_utils as graph_utils
from .nnx.errors import TraceContextError as TraceContextError
from .nnx.filterlib import All as All
from .nnx.filterlib import Not as Not
from .nnx.graph_utils import GraphDef as GraphDef
from .nnx.helpers import Dict as Dict
from .nnx.helpers import Sequence as Sequence
from .nnx.helpers import TrainState as TrainState
from .nnx.module import GraphDef as GraphDef
from .nnx.module import M as M
from .nnx.module import Module as Module
from .nnx.module import merge as merge
from .nnx.nn import initializers as initializers
from .nnx.nn.activations import celu as celu
from .nnx.nn.activations import elu as elu
from .nnx.nn.activations import gelu as gelu
from .nnx.nn.activations import glu as glu
from .nnx.nn.activations import hard_sigmoid as hard_sigmoid
from .nnx.nn.activations import hard_silu as hard_silu
from .nnx.nn.activations import hard_swish as hard_swish
from .nnx.nn.activations import hard_tanh as hard_tanh
from .nnx.nn.activations import leaky_relu as leaky_relu
from .nnx.nn.activations import log_sigmoid as log_sigmoid
from .nnx.nn.activations import log_softmax as log_softmax
from .nnx.nn.activations import logsumexp as logsumexp
from .nnx.nn.activations import one_hot as one_hot
from .nnx.nn.activations import relu as relu
from .nnx.nn.activations import relu6 as relu6
from .nnx.nn.activations import selu as selu
from .nnx.nn.activations import sigmoid as sigmoid
from .nnx.nn.activations import silu as silu
from .nnx.nn.activations import soft_sign as soft_sign
from .nnx.nn.activations import softmax as softmax
from .nnx.nn.activations import softplus as softplus
from .nnx.nn.activations import standardize as standardize
from .nnx.nn.activations import swish as swish
from .nnx.nn.activations import tanh as tanh
from .nnx.nn.attention import MultiHeadAttention as MultiHeadAttention
from .nnx.nn.attention import combine_masks as combine_masks
from .nnx.nn.attention import dot_product_attention as dot_product_attention
from .nnx.nn.attention import make_attention_mask as make_attention_mask
from .nnx.nn.attention import make_causal_mask as make_causal_mask
from .nnx.nn.linear import Conv as Conv
from .nnx.nn.linear import Embed as Embed
from .nnx.nn.linear import Linear as Linear
from .nnx.nn.linear import LinearGeneral as LinearGeneral
from .nnx.nn.normalization import BatchNorm as BatchNorm
from .nnx.nn.normalization import LayerNorm as LayerNorm
from .nnx.nn.normalization import RMSNorm as RMSNorm
from .nnx.nn.stochastic import Dropout as Dropout
from .nnx.pytreelib import Pytree as Pytree
from .nnx.pytreelib import TreeNode as TreeNode
from .nnx.rnglib import Rngs as Rngs
from .nnx.rnglib import RngStream as RngStream
from .nnx.spmd import PARTITION_NAME as PARTITION_NAME
from .nnx.spmd import get_partition_spec as get_partition_spec
from .nnx.spmd import get_named_sharding as get_named_sharding
from .nnx.spmd import with_partitioning as with_partitioning
from .nnx.spmd import with_sharding_constraint as with_sharding_constraint
from .nnx.state import State as State
from .nnx.transforms import JIT as JIT
from .nnx.transforms import Remat as Remat
from .nnx.transforms import Scan as Scan
from .nnx.transforms import Vmap as Vmap
from .nnx.transforms import grad as grad
from .nnx.transforms import jit as jit
from .nnx.transforms import remat as remat
from .nnx.transforms import scan as scan
from .nnx.transforms import value_and_grad as value_and_grad
from .nnx.transforms import vmap as vmap
from .nnx.variables import EMPTY as EMPTY
from .nnx.variables import A as A
from .nnx.variables import BatchStat as BatchStat
from .nnx.variables import Cache as Cache
from .nnx.variables import Empty as Empty
from .nnx.variables import Intermediate as Intermediate
from .nnx.variables import Param as Param
from .nnx.variables import Rng as Rng
from .nnx.variables import Variable as Variable
from .nnx.variables import VariableMetadata as VariableMetadata
from .nnx.variables import with_metadata as with_metadata
