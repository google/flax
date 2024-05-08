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

from flax.experimental.nnx.nnx import compatibility as compatibility
from flax.experimental.nnx.nnx import graph as graph
from flax.experimental.nnx.nnx import errors as errors
from flax.experimental.nnx.nnx import errors as helpers
from flax.experimental.nnx.nnx.filterlib import All as All
from flax.experimental.nnx.nnx.filterlib import Not as Not
from flax.experimental.nnx.nnx.graph import GraphDef as GraphDef
from flax.experimental.nnx.nnx.graph import GraphNode as GraphNode
from flax.experimental.nnx.nnx.helpers import Dict as Dict
from flax.experimental.nnx.nnx.helpers import List as List
from flax.experimental.nnx.nnx.helpers import Sequential as Sequential
from flax.experimental.nnx.nnx.helpers import TrainState as TrainState
from flax.experimental.nnx.nnx.module import M as M
from flax.experimental.nnx.nnx.module import Module as Module
from flax.experimental.nnx.nnx.graph import merge as merge
from flax.experimental.nnx.nnx.graph import UpdateContext as UpdateContext
from flax.experimental.nnx.nnx.graph import split as split
from flax.experimental.nnx.nnx.graph import update as update
from flax.experimental.nnx.nnx.graph import clone as clone
from flax.experimental.nnx.nnx.graph import pop as pop
from flax.experimental.nnx.nnx.graph import state as state
from flax.experimental.nnx.nnx.graph import graphdef as graphdef
from flax.experimental.nnx.nnx.nn import initializers as initializers
from flax.experimental.nnx.nnx.nn.activations import celu as celu
from flax.experimental.nnx.nnx.nn.activations import elu as elu
from flax.experimental.nnx.nnx.nn.activations import gelu as gelu
from flax.experimental.nnx.nnx.nn.activations import glu as glu
from flax.experimental.nnx.nnx.nn.activations import hard_sigmoid as hard_sigmoid
from flax.experimental.nnx.nnx.nn.activations import hard_silu as hard_silu
from flax.experimental.nnx.nnx.nn.activations import hard_swish as hard_swish
from flax.experimental.nnx.nnx.nn.activations import hard_tanh as hard_tanh
from flax.experimental.nnx.nnx.nn.activations import leaky_relu as leaky_relu
from flax.experimental.nnx.nnx.nn.activations import log_sigmoid as log_sigmoid
from flax.experimental.nnx.nnx.nn.activations import log_softmax as log_softmax
from flax.experimental.nnx.nnx.nn.activations import logsumexp as logsumexp
from flax.experimental.nnx.nnx.nn.activations import one_hot as one_hot
from flax.experimental.nnx.nnx.nn.activations import relu as relu
from flax.experimental.nnx.nnx.nn.activations import relu6 as relu6
from flax.experimental.nnx.nnx.nn.activations import selu as selu
from flax.experimental.nnx.nnx.nn.activations import sigmoid as sigmoid
from flax.experimental.nnx.nnx.nn.activations import silu as silu
from flax.experimental.nnx.nnx.nn.activations import soft_sign as soft_sign
from flax.experimental.nnx.nnx.nn.activations import softmax as softmax
from flax.experimental.nnx.nnx.nn.activations import softplus as softplus
from flax.experimental.nnx.nnx.nn.activations import standardize as standardize
from flax.experimental.nnx.nnx.nn.activations import swish as swish
from flax.experimental.nnx.nnx.nn.activations import tanh as tanh
from flax.experimental.nnx.nnx.nn.attention import MultiHeadAttention as MultiHeadAttention
from flax.experimental.nnx.nnx.nn.attention import combine_masks as combine_masks
from flax.experimental.nnx.nnx.nn.attention import dot_product_attention as dot_product_attention
from flax.experimental.nnx.nnx.nn.attention import make_attention_mask as make_attention_mask
from flax.experimental.nnx.nnx.nn.attention import make_causal_mask as make_causal_mask
from flax.experimental.nnx.nnx.nn.linear import Conv as Conv
from flax.experimental.nnx.nnx.nn.linear import Embed as Embed
from flax.experimental.nnx.nnx.nn.linear import Linear as Linear
from flax.experimental.nnx.nnx.nn.linear import LinearGeneral as LinearGeneral
from flax.experimental.nnx.nnx.nn.linear import Einsum as Einsum
from flax.experimental.nnx.nnx.nn.normalization import BatchNorm as BatchNorm
from flax.experimental.nnx.nnx.nn.normalization import LayerNorm as LayerNorm
from flax.experimental.nnx.nnx.nn.normalization import RMSNorm as RMSNorm
from flax.experimental.nnx.nnx.nn.stochastic import Dropout as Dropout
from flax.experimental.nnx.nnx.rnglib import Rngs as Rngs
from flax.experimental.nnx.nnx.rnglib import RngStream as RngStream
from flax.experimental.nnx.nnx.rnglib import RngState as RngState
from flax.experimental.nnx.nnx.rnglib import RngKey as RngKey
from flax.experimental.nnx.nnx.rnglib import RngCount as RngCount
from flax.experimental.nnx.nnx.rnglib import fork as fork
from flax.experimental.nnx.nnx.spmd import PARTITION_NAME as PARTITION_NAME
from flax.experimental.nnx.nnx.spmd import get_partition_spec as get_partition_spec
from flax.experimental.nnx.nnx.spmd import get_named_sharding as get_named_sharding
from flax.experimental.nnx.nnx.spmd import with_partitioning as with_partitioning
from flax.experimental.nnx.nnx.spmd import with_sharding_constraint as with_sharding_constraint
from flax.experimental.nnx.nnx.state import State as State
from flax.experimental.nnx.nnx.training import metrics as metrics
from flax.experimental.nnx.nnx.training import optimizer as optimizer
from flax.experimental.nnx.nnx.training.metrics import Metric as Metric
from flax.experimental.nnx.nnx.training.metrics import MultiMetric as MultiMetric
from flax.experimental.nnx.nnx.training.optimizer import Optimizer as Optimizer
from flax.experimental.nnx.nnx.transforms import Jit as Jit
from flax.experimental.nnx.nnx.transforms import jit as jit
from flax.experimental.nnx.nnx.transforms import Remat as Remat
from flax.experimental.nnx.nnx.transforms import Scan as Scan
from flax.experimental.nnx.nnx.transforms import Vmap as Vmap
from flax.experimental.nnx.nnx.transforms import grad as grad
from flax.experimental.nnx.nnx.transforms import remat as remat
from flax.experimental.nnx.nnx.transforms import scan as scan
from flax.experimental.nnx.nnx.transforms import value_and_grad as value_and_grad
from flax.experimental.nnx.nnx.transforms import vmap as vmap
from flax.experimental.nnx.nnx.transforms import eval_shape as eval_shape
from flax.experimental.nnx.nnx.variables import EMPTY as EMPTY
from flax.experimental.nnx.nnx.variables import A as A
from flax.experimental.nnx.nnx.variables import BatchStat as BatchStat
from flax.experimental.nnx.nnx.variables import Cache as Cache
from flax.experimental.nnx.nnx.variables import Empty as Empty
from flax.experimental.nnx.nnx.variables import Intermediate as Intermediate
from flax.experimental.nnx.nnx.variables import Param as Param
from flax.experimental.nnx.nnx.variables import Variable as Variable
from flax.experimental.nnx.nnx.variables import VariableState as VariableState
from flax.experimental.nnx.nnx.variables import VariableMetadata as VariableMetadata
from flax.experimental.nnx.nnx.variables import with_metadata as with_metadata
from flax.experimental.nnx.nnx.visualization import display as display
