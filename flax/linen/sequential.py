from typing import Sequence
from flax.linen.module import Module


class Sequential(nn.Module):
  layers: Sequence[nn.Module]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
