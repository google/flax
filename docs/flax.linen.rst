
flax.linen package
==================

.. currentmodule:: flax.linen

Linen is the Flax Module system. Read more about our design goals in the `Linen README <https://github.com/google/flax/blob/main/flax/linen/README.md>`_.



Module
------------------------

.. autoclass:: Module
   :members: setup, variable, param, bind, apply, init, init_with_output, make_rng, sow, variables, Variable, __setattr__

Init/Apply
------------------------

.. currentmodule:: flax.linen
.. autofunction:: apply
.. autofunction:: init
.. autofunction:: init_with_output

Variables
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


Linear modules
------------------------

.. autosummary::
  :toctree: _autosummary

    Dense
    DenseGeneral
    Conv
    ConvTranspose
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

    dot_product_attention_weights
    dot_product_attention
    SelfAttention


Stochastic
------------------------

.. autosummary::
  :toctree: _autosummary
  
    Dropout
    

RNN primitives
------------------------

.. autosummary::
  :toctree: _autosummary

    LSTMCell
    OptimizedLSTMCell
    GRUCell
