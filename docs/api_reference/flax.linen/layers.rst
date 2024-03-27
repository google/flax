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
  :class: Einsum

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

.. flax_module::
  :module: flax.linen
  :class: RMSNorm

.. flax_module::
  :module: flax.linen
  :class: InstanceNorm

.. flax_module::
  :module: flax.linen
  :class: SpectralNorm

.. flax_module::
  :module: flax.linen
  :class: WeightNorm


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
  :class: MultiHeadDotProductAttention

.. flax_module::
  :module: flax.linen
  :class: MultiHeadAttention

.. flax_module::
  :module: flax.linen
  :class: SelfAttention

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
  :class: ConvLSTMCell

.. flax_module::
  :module: flax.linen
  :class: SimpleCell

.. flax_module::
  :module: flax.linen
  :class: GRUCell

.. flax_module::
  :module: flax.linen
  :class: MGUCell

.. flax_module::
  :module: flax.linen
  :class: RNN

.. flax_module::
  :module: flax.linen
  :class: Bidirectional

BatchApply
------------------------

.. flax_module::
  :module: flax.linen
  :class: BatchApply
