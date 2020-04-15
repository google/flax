# Copyright 2020 The Flax Authors.
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

"""Flax Optimizer api."""

# pylint: disable=g-multiple-import
# re-export commonly used modules and functions
from .adam import Adam
from .adagrad import Adagrad
from .base import OptimizerState, OptimizerDef, Optimizer, MultiOptimizer, ModelParamTraversal
from .lamb import LAMB
from .lars import LARS
from .momentum import Momentum
from .rmsprop import RMSProp
from .sgd import GradientDescent
from .weight_norm import WeightNorm

__all__ = [
    "Adam",
    "Adagrad",
    "OptimizerState",
    "OptimizerDef",
    "Optimizer",
    "MultiOptimizer",
    "LAMB",
    "LARS",
    "Momentum",
    "RMSProp",
    "GradientDescent",
    "WeightNorm",
]
# pylint: enable=g-multiple-import
