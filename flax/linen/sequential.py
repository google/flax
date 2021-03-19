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

from typing import Sequence
from flax.linen.module import Module
from typing import Type


class Sequential(Module):

  """ A sequential module for stacking layers sequentially. Each layer has only one input and one output. Layers will be added to it in the order they are passed in.
    Attributes:
      layers: A sequence of the layers that needed to be stacked sequentially.
  """
  layers: Sequence[Type[Module]]

  def __call__(self, x):
  	"""Feed forward the input through the sequential layers.

    Args:
      x: The nd-array to be an input of the model.

    Returns:
      The output nd-array of the model.
    """

  	for layer in self.layers:
  	  x = layer(x)
  	return x
