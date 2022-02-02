
flax.optim package
===================

.. automodule:: flax.optim

.. currentmodule:: flax.optim

Optimizer Base Classes
------------------------

.. autoclass:: Optimizer
   :members: apply_gradient

.. autoclass:: OptimizerDef
   :members: apply_param_gradient, init_param_state, apply_gradient, init_state, update_hyper_params, create

MultiOptimizer
------------------------

.. autoclass:: MultiOptimizer
    :members: update_hyper_params

Available optimizers
------------------------

.. autosummary::
  :toctree: _autosummary

    Adam
    AdaBelief
    Adafactor
    Adagrad
    Adadelta
    DynamicScale
    GradientDescent
    LAMB
    LARS
    Momentum
    RMSProp
    WeightNorm
