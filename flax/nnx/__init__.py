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

from .nnx.bridge import wrappers as wrappers
from .nnx.bridge.variables import (
  register_variable_name_type_pair as register_variable_name_type_pair,
)
from .nnx import graph as graph
from .nnx import errors as errors
from .nnx import helpers as helpers
from .nnx import bridge as bridge
from .nnx import traversals as traversals
from .nnx import filterlib as filterlib
from .nnx import transforms as transforms
from .nnx import extract as extract
from .nnx.filterlib import WithTag as WithTag
from .nnx.filterlib import PathContains as PathContains
from .nnx.filterlib import OfType as OfType
from .nnx.filterlib import Any as Any
from .nnx.filterlib import All as All
from .nnx.filterlib import Not as Not
from .nnx.filterlib import Everything as Everything
from .nnx.filterlib import Nothing as Nothing
from .nnx.graph import GraphDef as GraphDef
from .nnx.graph import GraphState as GraphState
from .nnx.graph import PureState as PureState
from .nnx.object import Object as Object
from .nnx.object import pytree as pytree
from .nnx.helpers import Dict as Dict
from .nnx.helpers import List as List
from .nnx.helpers import Sequential as Sequential
from .nnx.helpers import TrainState as TrainState
from .nnx.module import M as M
from .nnx.module import Module as Module
from .nnx.graph import merge as merge
from .nnx.graph import UpdateContext as UpdateContext
from .nnx.graph import update_context as update_context
from .nnx.graph import current_update_context as current_update_context
from .nnx.graph import split as split
from .nnx.graph import update as update
from .nnx.graph import clone as clone
from .nnx.graph import pop as pop
from .nnx.graph import state as state
from .nnx.graph import graphdef as graphdef
from .nnx.graph import iter_graph as iter_graph
from .nnx.graph import call as call
from .nnx.graph import SplitContext as SplitContext
from .nnx.graph import split_context as split_context
from .nnx.graph import MergeContext as MergeContext
from .nnx.graph import merge_context as merge_context
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
from .nnx.nn.linear import ConvTranspose as ConvTranspose
from .nnx.nn.linear import Embed as Embed
from .nnx.nn.linear import Linear as Linear
from .nnx.nn.linear import LinearGeneral as LinearGeneral
from .nnx.nn.linear import Einsum as Einsum
from .nnx.nn.lora import LoRA as LoRA
from .nnx.nn.lora import LoRALinear as LoRALinear
from .nnx.nn.lora import LoRAParam as LoRAParam
from .nnx.nn.normalization import BatchNorm as BatchNorm
from .nnx.nn.normalization import LayerNorm as LayerNorm
from .nnx.nn.normalization import RMSNorm as RMSNorm
from .nnx.nn.normalization import GroupNorm as GroupNorm
from .nnx.nn.stochastic import Dropout as Dropout
from .nnx.rnglib import Rngs as Rngs
from .nnx.rnglib import RngStream as RngStream
from .nnx.rnglib import RngState as RngState
from .nnx.rnglib import RngKey as RngKey
from .nnx.rnglib import RngCount as RngCount
from .nnx.rnglib import ForkStates as ForkStates
from .nnx.rnglib import fork as fork
from .nnx.rnglib import reseed as reseed
from .nnx.rnglib import split_rngs as split_rngs
from .nnx.rnglib import restore_rngs as restore_rngs
from .nnx.spmd import PARTITION_NAME as PARTITION_NAME
from .nnx.spmd import get_partition_spec as get_partition_spec
from .nnx.spmd import get_named_sharding as get_named_sharding
from .nnx.spmd import with_partitioning as with_partitioning
from .nnx.spmd import with_sharding_constraint as with_sharding_constraint
from .nnx.state import State as State
from .nnx.training import metrics as metrics
from .nnx.variables import (
  Param as Param,
)
# this needs to be imported before optimizer to prevent circular import
from .nnx.training import optimizer as optimizer
from .nnx.training.metrics import Metric as Metric
from .nnx.training.metrics import MultiMetric as MultiMetric
from .nnx.training.optimizer import Optimizer as Optimizer
from .nnx.transforms.deprecated import Jit as Jit
from .nnx.transforms.deprecated import Remat as Remat
from .nnx.transforms.deprecated import Scan as Scan
from .nnx.transforms.deprecated import Vmap as Vmap
from .nnx.transforms.deprecated import Pmap as Pmap
from .nnx.transforms.autodiff import DiffState as DiffState
from .nnx.transforms.autodiff import grad as grad
from .nnx.transforms.autodiff import value_and_grad as value_and_grad
from .nnx.transforms.autodiff import custom_vjp as custom_vjp
from .nnx.transforms.autodiff import remat as remat
from .nnx.transforms.compilation import jit as jit
from .nnx.transforms.compilation import StateSharding as StateSharding
from .nnx.transforms.iteration import Carry as Carry
from .nnx.transforms.iteration import scan as scan
from .nnx.transforms.iteration import vmap as vmap
from .nnx.transforms.iteration import pmap as pmap
from .nnx.transforms.transforms import eval_shape as eval_shape
from .nnx.transforms.transforms import cond as cond
from .nnx.transforms.iteration import StateAxes as StateAxes
from .nnx.variables import A as A
from .nnx.variables import BatchStat as BatchStat
from .nnx.variables import Cache as Cache
from .nnx.variables import Intermediate as Intermediate
from .nnx.variables import Variable as Variable
from .nnx.variables import VariableState as VariableState
from .nnx.variables import VariableMetadata as VariableMetadata
from .nnx.variables import with_metadata as with_metadata
from .nnx.visualization import display as display
from .nnx.extract import to_tree as to_tree
from .nnx.extract import from_tree as from_tree
from .nnx.extract import TreeNode as TreeNode
