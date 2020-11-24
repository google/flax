
flax.optim package
===================

Optimizers utilities
------------------------

.. currentmodule:: flax.optim

.. automodule:: flax.optim


.. autoclass:: Optimizer
   :members: apply_gradient, compute_gradients, optimize

.. autoclass:: OptimizerDef
   :members: apply_param_gradient, init_param_state, apply_gradient, init_state, update_hyper_params, create

.. autoclass:: MultiOptimizer

.. autoclass:: ModelParamTraversal


Available optimizers
------------------------

.. autoclass:: Adam

.. autoclass:: Adafactor

.. autoclass:: Adagrad

.. autoclass:: DynamicScale

.. autoclass:: GradientDescent

.. autoclass:: LAMB

.. autoclass:: LARS

.. autoclass:: Momentum

.. autoclass:: RMSProp

.. autoclass:: WeightNorm
