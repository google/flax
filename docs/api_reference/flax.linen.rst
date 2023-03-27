
flax.linen package
==================

.. currentmodule:: flax.linen

Linen is the Flax Module system. Read more about our design goals in the `Linen README <https://github.com/google/flax/blob/main/flax/linen/README.md>`_.



Module
------------------------

.. autoclass:: Module
   :members: setup, variable, param, bind, unbind, apply, init, init_with_output, make_rng, sow, variables, Variable, __setattr__, tabulate, is_initializing, perturb

Init/Apply
------------------------

.. currentmodule:: flax.linen
.. autofunction:: apply
.. autofunction:: init
.. autofunction:: init_with_output

Variable dictionary
----------------------

.. automodule:: flax.core.variables
.. autoclass:: Variable


Compact methods
----------------------

.. currentmodule:: flax.linen
.. autofunction:: compact


No wrap methods
----------------------

.. currentmodule:: flax.linen
.. autofunction:: nowrap


Profiling
----------------------

.. automodule:: flax.linen
.. currentmodule:: flax.linen

.. autosummary::
  :toctree: _autosummary

    enable_named_call
    disable_named_call
    override_named_call


Inspection
----------------------

.. automodule:: flax.linen
.. currentmodule:: flax.linen

.. autosummary::
  :toctree: _autosummary

    tabulate


Transformations
----------------------

.. automodule:: flax.linen.transforms
.. currentmodule:: flax.linen

.. autosummary::
  :toctree: _autosummary

    vmap
    scan
    jit
    remat
    remat_scan
    map_variables
    jvp
    vjp
    custom_vjp
    while_loop
    cond
    switch


SPMD
----------------------

.. automodule:: flax.linen.spmd
.. currentmodule:: flax.linen

.. autosummary::
  :toctree: _autosummary

    Partitioned
    with_partitioning
    get_partition_spec
    LogicallyPartitioned
    logical_axis_rules
    set_logical_axis_rules
    get_logical_axis_rules
    logical_to_mesh_axes
    logical_to_mesh
    with_logical_constraint
    with_logical_partitioning


Linear Modules
------------------------

.. autosummary::
  :toctree: _autosummary
  :template: flax_module

    Dense
    DenseGeneral
    Conv
    ConvTranspose
    ConvLocal
    Embed


Normalization
------------------------

.. autosummary::
  :toctree: _autosummary
  :template: flax_module

    BatchNorm
    LayerNorm
    GroupNorm


Pooling
------------------------

.. autosummary::
  :toctree: _autosummary

    max_pool
    avg_pool
    pool


Activation functions
------------------------

.. automodule:: flax.linen.activation
.. currentmodule:: flax.linen.activation

.. autosummary::
  :toctree: _autosummary

    PReLU
    celu
    elu
    gelu
    glu
    hard_sigmoid
    hard_silu
    hard_swish
    hard_tanh
    leaky_relu
    log_sigmoid
    log_softmax
    logsumexp
    one_hot
    relu
    relu6 as relu6,
    selu
    sigmoid
    silu
    soft_sign
    softmax
    softplus
    standardize
    swish
    tanh


Initializers
------------------------

.. automodule:: flax.linen.initializers
.. currentmodule:: flax.linen.initializers

.. autosummary::
  :toctree: _autosummary

    constant
    delta_orthogonal
    glorot_normal
    glorot_uniform
    he_normal
    he_uniform
    kaiming_normal
    kaiming_uniform
    lecun_normal
    lecun_uniform
    normal
    ones
    ones_init
    orthogonal
    uniform
    standardize
    variance_scaling
    xavier_normal
    xavier_uniform
    zeros
    zeros_init


Combinators
------------------------

.. currentmodule:: flax.linen

.. autosummary::
  :toctree: _autosummary
  :template: flax_module

    Sequential


Attention primitives
------------------------

.. autosummary::
  :toctree: _autosummary

    dot_product_attention_weights
    dot_product_attention
    make_attention_mask
    make_causal_mask

.. autosummary::
  :toctree: _autosummary
  :template: flax_module

    SelfAttention
    MultiHeadDotProductAttention


Stochastic
------------------------

.. autosummary::
  :toctree: _autosummary
  :template: flax_module

    Dropout


RNN primitives
------------------------

.. autosummary::
  :toctree: _autosummary
  :template: flax_module

    LSTMCell
    OptimizedLSTMCell
    GRUCell
    RNNCellBase
    RNN
