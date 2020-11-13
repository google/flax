
flax.linen package
==================

.. currentmodule:: flax.linen
.. automodule:: flax.linen


Module
------------------------

.. autoclass:: Module
   :members: setup, variable, param, apply, init, init_with_output, variables, Variable, __setattr__

Variables
----------------------

.. automodule:: flax.core.variables
.. autoclass:: Variable


Compact methods
----------------------

.. currentmodule:: flax.linen
.. autofunction:: compact

	     
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
