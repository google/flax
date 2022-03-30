# Copyright 2022 The Flax Authors.
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

"""Combinators of modules, such as a Sequential."""

from typing import Any, Callable, Sequence

from flax.linen.module import Module


class Sequential(Module):
  """Applies a linear chain of Modules.

  Meant to be used only for the simple case of fusing together callables where
  the input of a particular module/op is the output of the previous one.

  Modules will be applied in the order that they are passed in the constructor.

  The apply() method of Sequential accepts any input and forwards it to the
  first module it contains. It chains the output sequentially to the input of
  the next module and returns the output of the final module.

  Example usage::

    class Foo(nn.Module):
      feature_sizes: Sequence[int]

      @nn.compact
      def __call__(self, x):
        return nn.Sequential([nn.Dense(4),
                              nn.relu,
                              nn.Dense(2),
                              nn.log_softmax])(x)
  """
  layers: Sequence[Callable[..., Any]]

  def __call__(self, *args, **kwargs):
    if not self.layers:
      raise ValueError(f'Empty Sequential module {self.name}.')

    outputs = self.layers[0](*args, **kwargs)
    for layer in self.layers[1:]:
      outputs = layer(outputs)
    return outputs
