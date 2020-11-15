from typing import TypeVar, Dict
import jax

from .frozen_dict import FrozenDict


K = TypeVar('K')
V = TypeVar('V')


@jax.tree_util.register_pytree_node_class
class FrozenVariableDict(FrozenDict[K, V]):
  """A FrozenVariableDict stores module variables in an immutable dictionary."""

  def __getitem__(self, key):
    if key not in self._dict:
      raise KeyError(f"Variable {key} is not in the FrozenVariableDict. Perhaps the variable you are "
                     "accessing has not been defined yet. Please make sure you access variables after "
                     "shape inference and that you access them by the proper name.")
    v = self._dict[key]
    if isinstance(v, dict):
      return FrozenVariableDict(v)
    return v


def freeze_variables(xs: Dict[K, V]) -> FrozenVariableDict[K, V]:
  """Freeze a nested dict of variables.

  Makes a nested `dict` immutable by transforming it into `FrozenVariableDict`.
  """
  return FrozenVariableDict(xs)