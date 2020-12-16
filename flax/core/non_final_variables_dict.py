from typing import TypeVar, Dict, Any
import jax

from .frozen_dict import FrozenDict


K = TypeVar('K')
V = TypeVar('V')


@jax.tree_util.register_pytree_node_class
class NonFinalVariablesDict(FrozenDict[K, V]):
  """A NonFinalVariablesDict is used to store variables that are possibly being accessed before shape inference.

  If the user tries to access a variable that is not in the dict, this will display an error message explaining that
  some variables may be missing if they are accessed before shape inference.
  """

  def __getitem__(self, key):
    if key not in self._dict:
      raise KeyError(f"Variable {key} is not in the NonFinalVariablesDict. Perhaps the variable you are "
                     "accessing has not been defined yet. Please make sure you access variables after "
                     "shape inference and that you access them by the proper name.")
    v = self._dict[key]
    if isinstance(v, dict):
      return NonFinalVariablesDict(v)
    return v

  def __repr__(self):
    frozen_repr = super().__repr__()
    return frozen_repr.replace("FrozenDict", "NonFinalVariablesDict", 1)


def make_nonfinal(xs: Dict[Any, Any]) -> NonFinalVariablesDict[Any, Any]:
  """Make a NonFinalVariablesDict from a dictionary of variables."""
  return NonFinalVariablesDict(xs)
