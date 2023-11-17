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

"""The Flax Module system."""


# pylint: disable=g-multiple-import,useless-import-alias
# re-export commonly used modules and functions
from flax.core import (
    DenyList,
    FrozenDict,
    broadcast,
    meta,
)
from flax.core.meta import (
    PARTITION_NAME,
    Partitioned,
    get_partition_spec,
    get_sharding,
    unbox,
    with_partitioning,
)
from flax.linen.activation import (
    PReLU,
    celu,
    elu,
    gelu,
    glu,
    hard_sigmoid,
    hard_silu,
    hard_swish,
    hard_tanh,
    leaky_relu,
    log_sigmoid,
    log_softmax,
    logsumexp,
    normalize,
    one_hot,
    relu,
    relu6,
    selu,
    sigmoid,
    silu,
    soft_sign,
    softmax,
    softplus,
    standardize,
    swish,
    tanh,
)
from flax.linen.attention import (
    MultiHeadDotProductAttention,
    SelfAttention,
    combine_masks,
    dot_product_attention,
    dot_product_attention_weights,
    make_attention_mask,
    make_causal_mask,
)
from flax.linen.combinators import Sequential
from flax.linen.fp8_ops import Fp8DotGeneralOp
from flax.linen.initializers import (
    ones,
    ones_init,
    zeros,
    zeros_init,
)
from flax.linen.linear import (
    Conv,
    ConvLocal,
    ConvTranspose,
    Dense,
    DenseGeneral,
    Embed,
)
from flax.linen.module import (
    Module,
    Variable,
    apply,
    compact,
    disable_named_call,
    enable_named_call,
    init,
    init_with_output,
    intercept_methods,
    merge_param,
    nowrap,
    override_named_call,
)
from flax.linen.normalization import (
    BatchNorm,
    GroupNorm,
    LayerNorm,
    RMSNorm,
    SpectralNorm,
    WeightNorm,
)
from flax.linen.pooling import (avg_pool, max_pool, pool)
from flax.linen.recurrent import (
    Bidirectional,
    ConvLSTMCell,
    GRUCell,
    LSTMCell,
    MGUCell,
    OptimizedLSTMCell,
    RNN,
    RNNCellBase,
)
from flax.linen.spmd import (
    LogicallyPartitioned,
    get_logical_axis_rules,
    logical_axis_rules,
    logical_to_mesh,
    logical_to_mesh_axes,
    logical_to_mesh_sharding,
    set_logical_axis_rules,
    with_logical_constraint,
    with_logical_partitioning,
)
from flax.linen.stochastic import Dropout
from flax.linen.summary import tabulate
from flax.linen.transforms import (
    add_metadata_axis,
    checkpoint,
    cond,
    custom_vjp,
    grad,
    jit,
    jvp,
    map_variables,
    named_call,
    remat,
    remat_scan,
    scan,
    switch,
    value_and_grad,
    vjp,
    vmap,
    while_loop,
)
# pylint: enable=g-multiple-import
