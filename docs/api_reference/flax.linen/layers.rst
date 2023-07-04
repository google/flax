Layers
======

.. currentmodule:: flax.linen

Linear Modules
------------------------

.. flax_module::
  :module: flax.linen
  :class: Dense

.. flax_module::
  :module: flax.linen
  :class: DenseGeneral

.. flax_module::
  :module: flax.linen
  :class: Conv

.. flax_module::
  :module: flax.linen
  :class: ConvTranspose

.. flax_module::
  :module: flax.linen
  :class: ConvLocal

.. flax_module::
  :module: flax.linen
  :class: Embed

Pooling
------------------------

.. autofunction:: max_pool
.. autofunction:: avg_pool
.. autofunction:: pool

Normalization
------------------------

.. flax_module::
  :module: flax.linen
  :class: BatchNorm

.. flax_module::
  :module: flax.linen
  :class: LayerNorm

.. flax_module::
  :module: flax.linen
  :class: GroupNorm


Combinators
------------------------

.. flax_module::
  :module: flax.linen
  :class: Sequential

Stochastic
------------------------

.. flax_module::
  :module: flax.linen
  :class: Dropout

Attention
------------------------

.. flax_module::
  :module: flax.linen
  :class: SelfAttention

.. flax_module::
  :module: flax.linen
  :class: MultiHeadDotProductAttention

.. autofunction:: dot_product_attention_weights
.. autofunction:: dot_product_attention
.. autofunction:: make_attention_mask
.. autofunction:: make_causal_mask

Recurrent
------------------------

.. flax_module::
  :module: flax.linen
  :class: RNNCellBase

.. flax_module::
  :module: flax.linen
  :class: LSTMCell

.. flax_module::
  :module: flax.linen
  :class: OptimizedLSTMCell

.. flax_module::
  :module: flax.linen
  :class: GRUCell

.. flax_module::
  :module: flax.linen
  :class: RNN

.. flax_module::
  :module: flax.linen
  :class: Bidirectional


**Summary**

.. autosummary::
  :toctree: _autosummary
  :template: flax_module

  Dense
  DenseGeneral
  Conv
  ConvTranspose
  ConvLocal
  Embed
  BatchNorm
  LayerNorm
  GroupNorm
  RMSNorm
  Sequential
  Dropout
  SelfAttention
  MultiHeadDotProductAttention
  RNNCellBase
  LSTMCell
  OptimizedLSTMCell
  GRUCell
  RNN
  Bidirectional

.. autosummary::
  :toctree: _autosummary

  max_pool
  avg_pool
  pool
  dot_product_attention_weights
  dot_product_attention
  make_attention_mask
  make_causal_mask