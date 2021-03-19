from typing import Sequence
from flax.linen.module import Module
from typing import Type



class Sequential(Module):
  layers: Sequence[Type[Module]]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
