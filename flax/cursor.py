# Copyright 2023 The Flax Authors.
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

import enum
from typing import Any, Callable, Dict, Generator, Generic, Mapping, Optional, Protocol, Tuple, TypeVar, Union, runtime_checkable
from flax.core import FrozenDict
import dataclasses


A = TypeVar('A')
Key = Any


@runtime_checkable
class Indexable(Protocol):

  def __getitem__(self, key) -> Any:
    ...


@dataclasses.dataclass
class ParentKey(Generic[A]):
  parent: 'Cursor[A]'
  key: Any


class AccessType(enum.Enum):
  GETITEM = enum.auto()
  GETATTR = enum.auto()


def is_named_tuple(obj):
  return (
      isinstance(obj, tuple)
      and hasattr(obj, '_fields')
      and hasattr(obj, '_asdict')
      and hasattr(obj, '_replace')
  )


def _get_changes(path, obj, update_fn):
  """Helper function for ``Cursor.apply_update``. Returns a generator of
  Tuple[Tuple[Union[str, int], Any], ...], where the first element is a
  tuple key path where the change was applied from the ``update_fn``, and
  the second element is the newly modified value. If the generator is
  non-empty, then the tuple key path will always be non-empty as well."""
  if path:
    str_path = '/'.join(str(key) for key, _ in path)
    new_obj = update_fn(str_path, obj)
    if new_obj is not obj:
      yield path, new_obj
      return

  if isinstance(obj, (FrozenDict, dict)):
    items = obj.items()
    access_type = AccessType.GETITEM
  elif is_named_tuple(obj):
    items = ((name, getattr(obj, name)) for name in obj._fields)  # type: ignore
    access_type = AccessType.GETATTR
  elif isinstance(obj, (list, tuple)):
    items = enumerate(obj)
    access_type = AccessType.GETITEM
  elif dataclasses.is_dataclass(obj):
    items = (
        (f.name, getattr(obj, f.name))
        for f in dataclasses.fields(obj)
        if f.init
    )
    access_type = AccessType.GETATTR
  else:
    yield from ()  # empty generator
    return

  for key, value in items:
    yield from _get_changes(path + ((key, access_type),), value, update_fn)


class Cursor(Generic[A]):
  obj: A
  parent_key: Optional[ParentKey[A]]
  changes: Dict[Any, Union[Any, 'Cursor[A]']]

  def __init__(self, obj: A, parent_key: Optional[ParentKey[A]]):
    # NOTE: we use `vars` here to avoid calling `__setattr__`
    # vars(self) = self.__dict__
    vars(self)['obj'] = obj
    vars(self)['parent_key'] = parent_key
    vars(self)['changes'] = {}

  @property
  def root(self) -> 'Cursor[A]':
    if self.parent_key is None:
      return self
    else:
      return self.parent_key.parent.root  # type: ignore

  def __getitem__(self, key) -> 'Cursor[A]':
    if key in self.changes:
      return self.changes[key]

    if not isinstance(self.obj, Indexable):
      raise TypeError(f'Cannot index into {self.obj}')

    if isinstance(self.obj, Mapping) and key not in self.obj:
      raise KeyError(f'Key {key} not found in {self.obj}')

    if is_named_tuple(self.obj):
      return getattr(self, self.obj._fields[key])  # type: ignore

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
    if is_named_tuple(self.obj):
      return setattr(self, self.obj._fields[key], value)  # type: ignore
    self.changes[key] = Cursor(value, ParentKey(self, key))

  def __setattr__(self, name, value):
    self.changes[name] = Cursor(value, ParentKey(self, name))

  def set(self, value) -> A:
    """Set a new value for an attribute, property, element or entry
    in the Cursor object and return a copy of the original object,
    containing the new set value.

    Example::

      from flax.cursor import cursor
      from flax.training import train_state
      import optax

      dict_obj = {'a': 1, 'b': (2, 3), 'c': [4, 5]}
      modified_dict_obj = cursor(dict_obj)['b'][0].set(10)
      assert modified_dict_obj == {'a': 1, 'b': (10, 3), 'c': [4, 5]}

      state = train_state.TrainState.create(
          apply_fn=lambda x: x,
          params=dict_obj,
          tx=optax.adam(1e-3),
      )
      modified_state = cursor(state).params['b'][1].set(10)
      assert modified_state.params == {'a': 1, 'b': (2, 10), 'c': [4, 5]}

    Args:
      value: the value used to set an attribute, property, element or entry in the Cursor object
    Returns:
      A copy of the original object with the new set value.
    """
    if self.parent_key is None:
      return value
    parent, key = self.parent_key.parent, self.parent_key.key  # type: ignore
    parent.changes[key] = value
    return parent.root.build()

  def build(self) -> A:
    """Create and return a copy of the original object with accumulated changes.
    This method is to be called after making changes to the Cursor object.

    NOTE: The new object is built bottom-up, the changes will be first applied
    to the leaf nodes, and then its parent, all the way up to the root.

    Example::

      from flax.cursor import cursor
      from flax.training import train_state
      import optax

      dict_obj = {'a': 1, 'b': (2, 3), 'c': [4, 5]}
      c = cursor(dict_obj)
      c['b'][0] = 10
      c['a'] = (100, 200)
      modified_dict_obj = c.build()
      assert modified_dict_obj == {'a': (100, 200), 'b': (10, 3), 'c': [4, 5]}

      state = train_state.TrainState.create(
          apply_fn=lambda x: x,
          params=dict_obj,
          tx=optax.adam(1e-3),
      )
      new_fn = lambda x: x + 1
      c = cursor(state)
      c.params['b'][1] = 10
      c.apply_fn = new_fn
      modified_state = c.build()
      assert modified_state.params == {'a': 1, 'b': (2, 10), 'c': [4, 5]}
      assert modified_state.apply_fn == new_fn

    Returns:
      A copy of the original object with the accumulated changes.
    """
    changes = {
        key: child.build() if isinstance(child, Cursor) else child
        for key, child in self.changes.items()
    }
    if isinstance(self.obj, FrozenDict):
      obj = self.obj.copy(changes)  # type: ignore
    elif isinstance(self.obj, (dict, list)):
      obj = self.obj.copy()  # type: ignore
      for key, value in changes.items():
        obj[key] = value
    elif is_named_tuple(self.obj):
      obj = self.obj._replace(**changes)  # type: ignore
    elif isinstance(self.obj, tuple):
      obj = list(self.obj)  # type: ignore
      for key, value in changes.items():
        obj[key] = value
      obj = tuple(obj)  # type: ignore
    elif dataclasses.is_dataclass(self.obj):
      obj = dataclasses.replace(self.obj, **changes)  # type: ignore
    else:
      obj = self.obj  # type: ignore

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

  def apply_update(
      self,
      update_fn: Callable[[str, Any], Any],
  ) -> 'Cursor[A]':
    """Traverse the Cursor object and apply conditional changes recursively via an ``update_fn``.
    The ``update_fn`` has a function signature of ``(str, Any) -> Any``:

    - The input arguments are the current key path (in the form of a string delimited
      by '/') and value at that current key path
    - The output is the new value (either modified by the ``update_fn`` or same as the
      input value if the condition wasn't fulfilled)

    To generate a copy of the original object with the accumulated changes, call the ``.build`` method.

    NOTES:

    - If the ``update_fn`` returns a modified value, this function will not recurse any further
      down that branch to apply changes. For example, if we intend to replace an attribute that points
      to a dictionary with an int, we don't need to look for further changes inside the dictionary,
      since the dictionary will be replaced anyways.
    - The ``is`` operator is used to determine whether the return value is modified (by comparing it
      to the input value). Therefore if the ``update_fn`` modifies a mutable container (e.g. lists,
      dicts, etc.) and returns the same container, ``.apply_update`` will treat the returned value as
      unmodified as it contains the same ``id``. To avoid this, return a copy of the modified value.
    - The ``.apply_update`` WILL NOT apply the ``update_fn`` to the value at the top-most level of
      the pytree (i.e. the root node). The ``update_fn`` will be applied recursively, starting at the
      root node's children.

    Example::

      import flax.linen as nn
      from flax.cursor import cursor
      import jax
      import jax.numpy as jnp

      class Model(nn.Module):
        @nn.compact
        def __call__(self, x):
          x = nn.Dense(3)(x)
          x = nn.relu(x)
          x = nn.Dense(3)(x)
          x = nn.relu(x)
          x = nn.Dense(3)(x)
          x = nn.relu(x)
          return x

      params = Model().init(jax.random.PRNGKey(0), jnp.empty((1, 2)))['params']

      def update_fn(path, value):
        '''Multiply all dense kernel params by 2 and add 1.
        Subtract the Dense_1 bias param by 1.'''
        if 'kernel' in path:
          return value * 2 + 1
        elif 'Dense_1' in path and 'bias' in path:
          return value - 1
        return value

      c = cursor(params)
      new_params = c.apply_update(update_fn).build()
      for layer in ('Dense_0', 'Dense_1', 'Dense_2'):
        assert (new_params[layer]['kernel'] == 2 * params[layer]['kernel'] + 1).all()
        if layer == 'Dense_1':
          assert (new_params[layer]['bias'] == jnp.array([-1, -1, -1])).all()
        else:
          assert (new_params[layer]['bias'] == params[layer]['bias']).all()

      assert jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                lambda x, y: (x == y).all(),
                params,
                Model().init(jax.random.PRNGKey(0), jnp.empty((1, 2)))[
                    'params'
                ],
            )
        ) # make sure original params are unchanged

    Args:
      update_fn: the function that will conditionally apply changes to the Cursor object
    Returns:
      The current Cursor object with the updates applied by the ``update_fn``.
    """
    for path, value in _get_changes((), self.obj, update_fn):
      child = self
      for key, access_type in path[:-1]:
        if access_type is AccessType.GETITEM:
          child = child[key]
        else:  # access_type is AccessType.GETATTR
          child = getattr(child, key)
      key, access_type = path[-1]
      if access_type is AccessType.GETITEM:
        child[key] = value
      else:  # access_type is AccessType.GETATTR
        setattr(child, key, value)

    return self


def cursor(obj: A) -> Cursor[A]:
  """Wrap Cursor over obj and return it.
  Changes can then be applied to the Cursor object in the following ways:

  - single-line change via the ``.set`` method
  - multiple changes, and then calling the ``.build`` method
  - multiple changes conditioned on the tree path and node value, via the ``.apply_update`` method

  ``.set`` example::

    from flax.cursor import cursor

    dict_obj = {'a': 1, 'b': (2, 3), 'c': [4, 5]}
    modified_dict_obj = cursor(dict_obj)['b'][0].set(10)
    assert modified_dict_obj == {'a': 1, 'b': (10, 3), 'c': [4, 5]}

  ``.build`` example::

    from flax.cursor import cursor

    dict_obj = {'a': 1, 'b': (2, 3), 'c': [4, 5]}
    c = cursor(dict_obj)
    c['b'][0] = 10
    c['a'] = (100, 200)
    modified_dict_obj = c.build()
    assert modified_dict_obj == {'a': (100, 200), 'b': (10, 3), 'c': [4, 5]}

  ``.apply_update`` example::

    from flax.cursor import cursor
    from flax.training import train_state
    import optax

    def update_fn(path, value):
      '''Replace params with empty dictionary.'''
      if 'params' in path:
        return {}
      return value

    state = train_state.TrainState.create(
        apply_fn=lambda x: x,
        params={'a': 1, 'b': 2},
        tx=optax.adam(1e-3),
    )
    c = cursor(state)
    state2 = c.apply_update(update_fn).build()
    assert state2.params == {}
    assert state.params == {'a': 1, 'b': 2} # make sure original params are unchanged

  View the docstrings for each method to see more examples of their usage.

  Args:
    obj: the object you want to wrap the Cursor in
  Returns:
    A Cursor object wrapped around obj.
  """
  return Cursor(obj, None)
