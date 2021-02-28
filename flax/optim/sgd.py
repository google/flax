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

import numpy as np

from .. import struct

from .base import OptimizerDef


@struct.dataclass
class _GradientDescentHyperParams:
  learning_rate: np.ndarray


class GradientDescent(OptimizerDef):
  """Gradient descent optimizer."""

  def __init__(self, learning_rate=None):
    """Constructor for the GradientDescent optimizer.

    Args:
      learning_rate: the step size used to update the parameters.
    """
    hyper_params = _GradientDescentHyperParams(learning_rate)
    super().__init__(hyper_params)

  def init_param_state(self, param):
    return ()

  def apply_param_gradient(self, step, hyper_params, param, state, grad):
    del step
    assert hyper_params.learning_rate is not None, 'no learning rate provided.'
    new_param = param - hyper_params.learning_rate * grad
    return new_param, state
