
flax.nn package
=================

.. currentmodule:: flax.nn
.. automodule:: flax.nn


Core: Module abstraction
------------------------

.. autoclass:: Module
   :members: create, create_by_shape, init, init_by_shape, partial, shared, apply, param, get_param, state, is_stateful, is_initializing

Core: Additional
------------------------

.. autosummary::
  :toctree: _autosummary

    Model
    Collection
    capture_module_outputs
    stateful
    get_state
    module_method



Linear modules
------------------------

.. autosummary::
  :toctree: _autosummary

    Dense
    DenseGeneral
    Conv
    Embed


Normalization
------------------------

.. autosummary::
  :toctree: _autosummary

    BatchNorm
    LayerNorm
    GroupNorm


Pooling
------------------------

.. autosummary::
  :toctree: _autosummary

    max_pool
    avg_pool


Activation functions
------------------------

.. autosummary::
  :toctree: _autosummary

    celu
    elu
    gelu
    glu
    log_sigmoid
    log_softmax
    relu
    sigmoid
    soft_sign
    softmax
    softplus
    swish


Stochastic functions
------------------------

.. autosummary::
  :toctree: _autosummary

    make_rng
    stochastic
    is_stochastic
    dropout


Attention primitives
------------------------

.. autosummary::
  :toctree: _autosummary

    dot_product_attention
    SelfAttention


RNN primitives
------------------------

.. autosummary::
  :toctree: _autosummary

    LSTMCell
    GRUCell
