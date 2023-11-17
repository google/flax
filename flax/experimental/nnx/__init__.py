# Copyright 2023 The Flax Authors.
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

from flax.experimental.nnx.nnx import compatibility
from flax.experimental.nnx.nnx import graph_utils
from flax.experimental.nnx.nnx.dataclasses import dataclass
from flax.experimental.nnx.nnx.dataclasses import field
from flax.experimental.nnx.nnx.dataclasses import param_field
from flax.experimental.nnx.nnx.dataclasses import treenode_field
from flax.experimental.nnx.nnx.dataclasses import variable_field
from flax.experimental.nnx.nnx.errors import TraceContextError
from flax.experimental.nnx.nnx.filterlib import All
from flax.experimental.nnx.nnx.filterlib import Not
from flax.experimental.nnx.nnx.flaglib import flags
from flax.experimental.nnx.nnx.graph_utils import GraphDef
from flax.experimental.nnx.nnx.helpers import Dict
from flax.experimental.nnx.nnx.helpers import Sequence
from flax.experimental.nnx.nnx.helpers import TrainState
from flax.experimental.nnx.nnx.module import GraphDef
from flax.experimental.nnx.nnx.module import M
from flax.experimental.nnx.nnx.module import merge
from flax.experimental.nnx.nnx.module import Module
from flax.experimental.nnx.nnx.nn import initializers
from flax.experimental.nnx.nnx.nn.activations import celu
from flax.experimental.nnx.nnx.nn.activations import elu
from flax.experimental.nnx.nnx.nn.activations import gelu
from flax.experimental.nnx.nnx.nn.activations import glu
from flax.experimental.nnx.nnx.nn.activations import hard_sigmoid
from flax.experimental.nnx.nnx.nn.activations import hard_silu
from flax.experimental.nnx.nnx.nn.activations import hard_swish
from flax.experimental.nnx.nnx.nn.activations import hard_tanh
from flax.experimental.nnx.nnx.nn.activations import leaky_relu
from flax.experimental.nnx.nnx.nn.activations import log_sigmoid
from flax.experimental.nnx.nnx.nn.activations import log_softmax
from flax.experimental.nnx.nnx.nn.activations import logsumexp
from flax.experimental.nnx.nnx.nn.activations import normalize
from flax.experimental.nnx.nnx.nn.activations import one_hot
from flax.experimental.nnx.nnx.nn.activations import relu
from flax.experimental.nnx.nnx.nn.activations import relu6
from flax.experimental.nnx.nnx.nn.activations import selu
from flax.experimental.nnx.nnx.nn.activations import sigmoid
from flax.experimental.nnx.nnx.nn.activations import silu
from flax.experimental.nnx.nnx.nn.activations import soft_sign
from flax.experimental.nnx.nnx.nn.activations import softmax
from flax.experimental.nnx.nnx.nn.activations import softplus
from flax.experimental.nnx.nnx.nn.activations import standardize
from flax.experimental.nnx.nnx.nn.activations import swish
from flax.experimental.nnx.nnx.nn.activations import tanh
from flax.experimental.nnx.nnx.nn.linear import Conv
from flax.experimental.nnx.nnx.nn.linear import Embed
from flax.experimental.nnx.nnx.nn.linear import Linear
from flax.experimental.nnx.nnx.nn.normalization import BatchNorm
from flax.experimental.nnx.nnx.nn.normalization import LayerNorm
from flax.experimental.nnx.nnx.nn.stochastic import Dropout
from flax.experimental.nnx.nnx.pytreelib import Pytree
from flax.experimental.nnx.nnx.pytreelib import TreeNode
from flax.experimental.nnx.nnx.rnglib import Rngs
from flax.experimental.nnx.nnx.rnglib import RngStream
from flax.experimental.nnx.nnx.spmd import get_partition_spec
from flax.experimental.nnx.nnx.spmd import PARTITION_NAME
from flax.experimental.nnx.nnx.spmd import with_partitioning
from flax.experimental.nnx.nnx.spmd import with_sharding_constraint
from flax.experimental.nnx.nnx.state import State
from flax.experimental.nnx.nnx.transforms import grad
from flax.experimental.nnx.nnx.transforms import JIT
from flax.experimental.nnx.nnx.transforms import jit
from flax.experimental.nnx.nnx.transforms import Remat
from flax.experimental.nnx.nnx.transforms import remat
from flax.experimental.nnx.nnx.transforms import Scan
from flax.experimental.nnx.nnx.transforms import scan
from flax.experimental.nnx.nnx.transforms import value_and_grad
from flax.experimental.nnx.nnx.transforms import Vmap
from flax.experimental.nnx.nnx.transforms import vmap
from flax.experimental.nnx.nnx.variables import A
from flax.experimental.nnx.nnx.variables import BatchStat
from flax.experimental.nnx.nnx.variables import Cache
from flax.experimental.nnx.nnx.variables import EMPTY
from flax.experimental.nnx.nnx.variables import Empty
from flax.experimental.nnx.nnx.variables import Intermediate
from flax.experimental.nnx.nnx.variables import Param
from flax.experimental.nnx.nnx.variables import Rng
from flax.experimental.nnx.nnx.variables import Variable
from flax.experimental.nnx.nnx.variables import VariableMetadata
from flax.experimental.nnx.nnx.variables import with_metadata
from flax.linen.pooling import avg_pool
from flax.linen.pooling import max_pool
from flax.linen.pooling import min_pool
from flax.linen.pooling import pool
