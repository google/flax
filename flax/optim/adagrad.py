# Copyright 2021 The Flax Authors.
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

import jax.numpy as jnp
import numpy as np
from .. import struct
from .base import OptimizerDef


@struct.dataclass
class _AdagradHyperParams:
  """Adagrad hyper parameters"""

  learning_rate: float
  eps: float


@struct.dataclass
class _AdagradParamState:
  """Adagrad parameter state"""

  G: np.ndarray


class Adagrad(OptimizerDef):
  """Adagrad optimizer"""
  def __init__(self, learning_rate: float = None, eps=1e-8):
    """Constructor for the Adagrad optimizer.
        
    Args:
      learning_rate: the step size used to update the parameters.
    """
    hyper_params = _AdagradHyperParams(learning_rate, eps)
    super().__init__(hyper_params)

  def init_param_state(self, param):
    """Initialize parameter state"""
    return _AdagradParamState(jnp.zeros_like(param))

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    """Apply per-parameter gradients"""

    assert hyper_params.learning_rate is not None, 'no learning rate provided.'
    new_G = state.G + jnp.square(grad)
    new_param = param - hyper_params.learning_rate * grad / (jnp.sqrt(new_G) +
                                                             hyper_params.eps)
    new_state = _AdagradParamState(new_G)

    return new_param, new_state
