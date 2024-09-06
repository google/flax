# Copyright 2024 The Flax Authors.
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


import jax

from flax import nnx


# lets pretend this function loads a pretrained model from a checkpoint
def load_pretrained():
  return nnx.Linear(784, 128, rngs=nnx.Rngs(0))


# create a simple linear classifier using a pretrained backbone
class Classifier(nnx.Module):
  def __init__(self, *, rngs: nnx.Rngs):
    self.backbone = nnx.Linear(784, 128, rngs=nnx.Rngs(0))
    self.head = nnx.Linear(128, 10, rngs=rngs)

  def __call__(self, x):
    x = self.backbone(x)
    x = nnx.relu(x)
    x = self.head(x)
    return x


# create the classifier using the pretrained backbone, here we are technically
# doing "parameter surgery", however, compared to Haiku/Flax where you must manually
# construct the parameter structure, in NNX this is done automatically
model = Classifier(rngs=nnx.Rngs(42))
model.backbone = load_pretrained()


# create a filter to select all the parameters that are not part of the
# backbone, i.e. the classifier parameters
is_trainable = lambda path, node: (
  'backbone' in path and isinstance(node, nnx.Param)
)

# split the parameters into trainable and non-trainable parameters
graphdef, trainable_params, non_trainable = nnx.split(model, is_trainable, ...)

print(
  'trainable_params =',
  jax.tree.map(jax.numpy.shape, trainable_params),
)
print('non_trainable = ', jax.tree.map(jax.numpy.shape, non_trainable))
