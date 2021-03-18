from typing import Sequence
from flax.linen.module import Module


class Sequential(Module):
  layers: Sequence[Module]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
