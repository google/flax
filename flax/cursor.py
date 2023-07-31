import enum
from sqlite3 import Cursor
from typing import Any, Callable, Generic, Mapping, NamedTuple, Optional, Protocol, Sequence, TypeVar, Union, runtime_checkable
import jax
from flax.core import freeze, FrozenDict
import dataclasses
from flax.training.train_state import TrainState
import optax

A = TypeVar('A')
Key = Any


@runtime_checkable
class Indexable(Protocol):

  def __getitem__(self, key) -> Any:
    ...


class AccessType(enum.Enum):
  GETATTR = enum.auto()
  GETITEM = enum.auto()


# def flatten_until_found(tree: Any, targets: Sequence[Any]):
#   remaining = len(targets)

#   def is_leaf(x):
#     nonlocal remaining
#     leaf = False
#     if x in targets:
#       remaining -= 1
#       leaf = True
#     return leaf or remaining <= 0

#   value = jax.tree_util.tree_flatten(tree, is_leaf=is_leaf)

#   if remaining > 0:
#     return None
#   else:
#     return value


@dataclasses.dataclass
class ParentKey(Generic[A]):
  parent: 'Cursor[A]'
  key: Any


class Cursor(Generic[A]):
  obj: A
  parent_key: Optional[ParentKey[A]]
  changes: dict[Any, Union[Any, 'Cursor[A]']]

  def __init__(self, obj: A, parent_key: Optional[ParentKey[A]]):
    vars(self)['obj'] = obj
    vars(self)['parent_key'] = parent_key
    vars(self)['changes'] = {}

  @property
  def root(self) -> 'Cursor[A]':
    if self.parent_key is None:
      return self
    else:
      return self.parent_key.parent.root

  def __getitem__(self, key) -> 'Cursor[A]':
    if key in self.changes:
      return self.changes[key]

    if not isinstance(self.obj, Indexable):
      raise TypeError(f'Cannot index into {self.obj}')

    if isinstance(self.obj, Mapping) and key not in self.obj:
      raise KeyError(f'Key {key} not found in {self.obj}')

    child = Cursor(self.obj[key], ParentKey(self, key))
    self.changes[key] = child
    return child

  def __getattr__(self, name) -> 'Cursor[A]':
    if name in self.changes:
      return self.changes[name]

    if not hasattr(self.obj, name):
      raise AttributeError(f'Attribute {name} not found in {self.obj}')

    child = Cursor(getattr(self.obj, name), ParentKey(self, name))
    self.changes[name] = child
    return child

  def __setitem__(self, key, value):
    self.changes[key] = value

  def __setattr__(self, name, value):
    self.changes[name] = value

  def apply(self, change_fn: Callable[[str, Any], tuple[bool, Any]]):
    """
    def increment_ints_at_layer1(path: str, value):
      if 'layer1' in path and isinstance(value, int)
        return True, value + 1
      else:
        return False, value

    c = cursor(config)
    c.apply(increment_ints_at_layer1)
    config = c.build()
    """
    ...

  def build(self) -> A:
    changes = {
        key: child.build() if isinstance(child, Cursor) else child
        for key, child in self.changes.items()
    }
    if isinstance(self.obj, FrozenDict):
      obj = self.obj.copy(changes)
    elif isinstance(self.obj, (dict, list)):
      obj = self.obj.copy()
      for key, value in changes.items():
        obj[key] = value
    elif isinstance(self.obj, tuple):
      obj = list(self.obj)
      for key, value in changes.items():
        obj[key] = value
      obj = tuple(obj)
    elif dataclasses.is_dataclass(self.obj):
      obj = dataclasses.replace(self.obj, **changes)
    else:
      raise ValueError(f'Cannot build object of type {type(self.obj).__name__}')

      # NOTE: There is a way to try to do a general replace for pytrees, but it requires
      # the key of `changes` to store the type of access (getattr, getitem, etc.)
      # in order to access those value from the original object and try to replace them
      # with the new value. For simplicity, this is not implemented for now.
      # ----------------------
      # changed_values = tuple(changes.values())
      # result = flatten_until_found(self.obj, changed_values)

      # if result is None:
      #   raise ValueError('Cannot find object in parent')

      # leaves, treedef = result
      # leaves = [leaf if leaf is not self.obj else value for leaf in leaves]
      # obj = jax.tree_util.tree_unflatten(treedef, leaves)

    return obj  # type: ignore

  def set(self, value) -> A:
    if self.parent_key is None:
      return value
    parent, key = self.parent_key.parent, self.parent_key.key
    parent.changes[key] = value
    return parent.root.build()


def cursor(obj: A) -> Cursor[A]:
  return Cursor(obj, None)
