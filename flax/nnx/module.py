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

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp

from flax.nnx import (
  filterlib,
  graph,
)
from flax.nnx import variablelib as variableslib
from flax.nnx.pytreelib import Pytree, PytreeMeta
from flax.nnx.graph import GraphState
from flax.typing import Key, Path, PathParts
import warnings

A = tp.TypeVar('A')
B = tp.TypeVar('B')
M = tp.TypeVar('M', bound='Module')
S = tp.TypeVar('S', bound=tp.Union[GraphState, tuple[GraphState, ...]])
V = tp.TypeVar('V', bound=variableslib.Variable[tp.Any])
F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])

StateMapping = tp.Mapping[Path, tp.Any]
tuple_reduce = lambda xs, x: xs + (x,)
tuple_init = lambda: ()


class ModuleMeta(PytreeMeta):
  # we keep a trivial derived class just in case we need to
  # add more functionality in the future
  pass


class Module(Pytree, metaclass=ModuleMeta):
  """Base class for all neural network modules.

  Layers and models should subclass this class.

  ``Module``'s can contain submodules, and in this way can be nested in a tree
  structure. Submodules can be assigned as regular attributes inside the
  ``__init__`` method.

  You can define arbitrary "forward pass" methods on your ``Module`` subclass.
  While no methods are special-cased, ``__call__`` is a popular choice since
  you can call the ``Module`` directly::

    >>> from flax import nnx
    >>> import jax.numpy as jnp

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.linear1 = nnx.Linear(2, 3, rngs=rngs)
    ...     self.linear2 = nnx.Linear(3, 4, rngs=rngs)
    ...   def __call__(self, x):
    ...     x = self.linear1(x)
    ...     x = nnx.relu(x)
    ...     x = self.linear2(x)
    ...     return x

    >>> x = jnp.ones((1, 2))
    >>> model = Model(rngs=nnx.Rngs(0))
    >>> y = model(x)
  """

  def sow(
      self,
      variable_type: type[variableslib.Variable[B]] | str,
      name: str,
      value: A,
      reduce_fn: tp.Callable[[B, A], B] = tuple_reduce,
      init_fn: tp.Callable[[], B] = tuple_init,  # type: ignore
  ) -> bool:
    """``sow()`` can be used to collect intermediate values without
    the overhead of explicitly passing a container through each Module call.
    ``sow()`` stores a value in a new ``Module`` attribute, denoted by ``name``.
    The value will be wrapped by a :class:`Variable` of type ``variable_type``,
    which can be useful to filter for in :func:`split`, :func:`state` and
    :func:`pop`.

    By default the values are stored in a tuple and each stored value
    is appended at the end. This way all intermediates can be tracked when
    the same module is called multiple times.

    Example usage::

      >>> from flax import nnx
      >>> import jax.numpy as jnp

      >>> class Model(nnx.Module):
      ...   def __init__(self, rngs):
      ...     self.linear1 = nnx.Linear(2, 3, rngs=rngs)
      ...     self.linear2 = nnx.Linear(3, 4, rngs=rngs)
      ...   def __call__(self, x, add=0):
      ...     x = self.linear1(x)
      ...     self.sow(nnx.Intermediate, 'i', x+add)
      ...     x = self.linear2(x)
      ...     return x

      >>> x = jnp.ones((1, 2))
      >>> model = Model(rngs=nnx.Rngs(0))
      >>> assert not hasattr(model, 'i')

      >>> y = model(x)
      >>> assert hasattr(model, 'i')
      >>> assert len(model.i) == 1 # tuple of length 1
      >>> assert model.i[0].shape == (1, 3)

      >>> y = model(x, add=1)
      >>> assert len(model.i) == 2 # tuple of length 2
      >>> assert (model.i[0] + 1 == model.i[1]).all()

    Alternatively, a custom init/reduce function can be passed::

      >>> class Model(nnx.Module):
      ...   def __init__(self, rngs):
      ...     self.linear1 = nnx.Linear(2, 3, rngs=rngs)
      ...     self.linear2 = nnx.Linear(3, 4, rngs=rngs)
      ...   def __call__(self, x):
      ...     x = self.linear1(x)
      ...     self.sow(nnx.Intermediate, 'sum', x,
      ...              init_fn=lambda: 0,
      ...              reduce_fn=lambda prev, curr: prev+curr)
      ...     self.sow(nnx.Intermediate, 'product', x,
      ...              init_fn=lambda: 1,
      ...              reduce_fn=lambda prev, curr: prev*curr)
      ...     x = self.linear2(x)
      ...     return x

      >>> x = jnp.ones((1, 2))
      >>> model = Model(rngs=nnx.Rngs(0))

      >>> y = model(x)
      >>> assert (model.sum[...] == model.product[...]).all()
      >>> intermediate = model.sum[...]

      >>> y = model(x)
      >>> assert (model.sum[...] == intermediate*2).all()
      >>> assert (model.product[...] == intermediate**2).all()

    Args:
      variable_type: The :class:`Variable` type for the stored value.
        Typically :class:`Intermediate` is used to indicate an
        intermediate value.
      name: A string denoting the ``Module`` attribute name, where
        the sowed value is stored.
      value: The value to be stored.
      reduce_fn: The function used to combine the existing value with the new
        value. The default is to append the value to a tuple.
      init_fn: For the first value stored, ``reduce_fn`` will be passed the result
        of ``init_fn`` together with the value to be stored. The default is an
        empty tuple.
    """
    if isinstance(variable_type, str):
      variable_type = variableslib.variable_type_from_name(
          variable_type, allow_register=True
      )

    if hasattr(self, name):
      variable = getattr(self, name)
      if not isinstance(variable, variableslib.Variable):
        raise ValueError(
          f"Expected '{name}' to be a Variable, got {type(variable).__name__}"
        )
      elif type(variable) != variable_type:
        raise ValueError(
          f"Expected '{name}' to be of type '{variable_type.__name__}', "
          f"got '{type(variable).__name__}'"
        )
      variable.raw_value = reduce_fn(variable.raw_value, value)
    else:
      reduced_value = reduce_fn(init_fn(), value)
      setattr(self, name, variable_type(reduced_value))

    return True

  def perturb(
      self,
      name: str,
      value: tp.Any,
      variable_type: (
          str | type[variableslib.Variable[tp.Any]]
      ) = variableslib.Perturbation,
  ):
    """Add an zero-value variable ("perturbation") to the intermediate value.

    The gradient of ``value`` would be the same as the gradient of this
    perturbation variable. Therefore, if you define your loss function with
    both params and perturbations as standalone arguments, you can get the
    intermediate gradients of ``value`` by running ``jax.grad`` on the
    perturbation variable.

    Since the shape of the perturbation value depends on the shape of the input,
    a perturbation variable is only created after you run a sample input through
    the model once.

    .. note::
      This creates extra dummy variables of the same size as ``value``, thus
      occupies more memory. Use it only to debug gradients in training.

    Example usage::

      >>> from flax import nnx
      >>> import jax.numpy as jnp

      >>> class Model(nnx.Module):
      ...   def __init__(self, rngs):
      ...     self.linear1 = nnx.Linear(2, 3, rngs=rngs)
      ...     self.linear2 = nnx.Linear(3, 4, rngs=rngs)
      ...   def __call__(self, x):
      ...     x = self.linear1(x)
      ...     x = self.perturb('xgrad', x)
      ...     x = self.linear2(x)
      ...     return x

      >>> x = jnp.ones((1, 2))
      >>> y = jnp.ones((1, 4))
      >>> model = Model(rngs=nnx.Rngs(0))
      >>> assert not hasattr(model, 'xgrad')  # perturbation requires a sample input run
      >>> _ = model(x)
      >>> assert model.xgrad.shape == (1, 3)   # same as the intermediate value
      >>> graphdef, params, perturbations = nnx.split(model, nnx.Param, nnx.Perturbation)

      >>> # Take gradients on the Param and Perturbation variables
      >>> @nnx.grad(argnums=(0, 1))
      ... def grad_loss(params, perturbations, inputs, targets):
      ...   model = nnx.merge(graphdef, params, perturbations)
      ...   return jnp.mean((model(inputs) - targets) ** 2)

      >>> (grads, perturbations) = grad_loss(params, perturbations, x, y)
      >>> # `perturbations.xgrad[...]` is the intermediate gradient
      >>> assert not jnp.array_equal(perturbations.xgrad[...], jnp.zeros((1, 3)))

    Args:
      name: A string denoting the ``Module`` attribute name for the
        perturbation value.
      value: The value to take intermediate gradient.
      variable_type: The :class:`Variable` type for the stored perturbation.
        Defaulted at :class:`nnx.Perturbation`.
    """
    if isinstance(variable_type, str):
      variable_type = variableslib.variable_type_from_name(
          variable_type, allow_register=True
      )
    if not hasattr(self, name):
      zeros = jax.tree.map(jnp.zeros_like, value)
      setattr(self, name, variable_type(zeros))
    old_value: variableslib.Variable[tp.Any] = getattr(self, name)
    if not isinstance(old_value, variable_type):
      raise ValueError(
        f"Expected '{name}' to be of type '{variable_type.__name__}', "
        f"got '{type(old_value).__name__}'"
      )
    return old_value[...] + value

  def iter_modules(self) -> tp.Iterator[tuple[PathParts, Module]]:
    """
    Warning: this method is method is deprecated; use :func:`iter_modules` instead.

    Recursively iterates over all nested :class:`Module`'s of the current Module, including
    the current Module. Alias of :func:`iter_modules`.
    """
    warnings.warn(
      "The 'm.iter_modules()' method is deprecated; use the 'nnx.iter_modules(m)' function instead.",
      DeprecationWarning,
      stacklevel=2,
    )
    yield from iter_modules(self)

  def iter_children(self) -> tp.Iterator[tuple[Key, Module]]:
    """
    Warning: this method is method is deprecated; use :func:`iter_children` instead.

    Iterates over all children :class:`Module`'s of the current Module. This
    method is similar to :func:`iter_modules`, except it only iterates over the
    immediate children, and does not recurse further down. Alias of :func:`iter_children`.
    """
    warnings.warn(
      "The 'm.iter_children()' method is deprecated; use the 'nnx.iter_children(m)' function instead.",
      DeprecationWarning,
      stacklevel=2,
    )
    yield from iter_children(self)

  def set_attributes(
    self,
    *filters: filterlib.Filter,
    raise_if_not_found: bool = True,
    **attributes: tp.Any,
  ) -> None:
    """Sets the attributes of nested Modules including the current Module.
    If the attribute is not found in the Module, it is ignored.

    Example::

      >>> from flax import nnx
      ...
      >>> class Block(nnx.Module):
      ...   def __init__(self, din, dout, *, rngs: nnx.Rngs):
      ...     self.linear = nnx.Linear(din, dout, rngs=rngs)
      ...     self.dropout = nnx.Dropout(0.5, deterministic=False)
      ...     self.batch_norm = nnx.BatchNorm(10, use_running_average=False, rngs=rngs)
      ...
      >>> block = Block(2, 5, rngs=nnx.Rngs(0))
      >>> block.dropout.deterministic, block.batch_norm.use_running_average
      (False, False)
      >>> block.set_attributes(deterministic=True, use_running_average=True)
      >>> block.dropout.deterministic, block.batch_norm.use_running_average
      (True, True)

    ``Filter``'s can be used to set the attributes of specific Modules::

      >>> block = Block(2, 5, rngs=nnx.Rngs(0))
      >>> block.set_attributes(nnx.Dropout, deterministic=True)
      >>> # Only the dropout will be modified
      >>> block.dropout.deterministic, block.batch_norm.use_running_average
      (True, False)

    Args:
      *filters: Filters to select the Modules to set the attributes of.
      raise_if_not_found: If True (default), raises a ValueError if at least one attribute
        instance is not found in one of the selected Modules.
      **attributes: The attributes to set.
    """
    remaining_attributes = set(attributes.keys())
    if not filters:
      filters = (True,)
    predicates = tuple(map(filterlib.to_predicate, filters))
    for path, module in iter_modules(self):
      for predicate in predicates:
        if predicate(path, module):
          for name, value in attributes.items():
            if hasattr(module, name):
              if name in remaining_attributes:
                remaining_attributes.remove(name)
              setattr(module, name, value)
          break

    if remaining_attributes and raise_if_not_found:
      raise ValueError(
        f'Could not find at least one instance of the following attributes: {remaining_attributes}'
      )

  def train(self, **attributes):
    """Sets the Module to training mode.

    ``train`` uses ``set_attributes`` to recursively set attributes ``deterministic=False``
    and ``use_running_average=False`` of all nested Modules that have these attributes.
    Its primarily used to control the runtime behavior of the ``Dropout`` and ``BatchNorm``
    Modules.

    Example::

      >>> from flax import nnx
      ...
      >>> class Block(nnx.Module):
      ...   def __init__(self, din, dout, *, rngs: nnx.Rngs):
      ...     self.linear = nnx.Linear(din, dout, rngs=rngs)
      ...     # initialize Dropout and BatchNorm in eval mode
      ...     self.dropout = nnx.Dropout(0.5, deterministic=True)
      ...     self.batch_norm = nnx.BatchNorm(10, use_running_average=True, rngs=rngs)
      ...
      >>> block = Block(2, 5, rngs=nnx.Rngs(0))
      >>> block.dropout.deterministic, block.batch_norm.use_running_average
      (True, True)
      >>> block.train()
      >>> block.dropout.deterministic, block.batch_norm.use_running_average
      (False, False)

    Args:
      **attributes: additional attributes passed to ``set_attributes``.
    """
    return self.set_attributes(
      deterministic=False,
      use_running_average=False,
      **attributes,
      raise_if_not_found=False,
    )

  def eval(self, **attributes):
    """Sets the Module to evaluation mode.

    ``eval`` uses ``set_attributes`` to recursively set attributes ``deterministic=True``
    and ``use_running_average=True`` of all nested Modules that have these attributes.
    Its primarily used to control the runtime behavior of the ``Dropout`` and ``BatchNorm``
    Modules.

    Example::

      >>> from flax import nnx
      ...
      >>> class Block(nnx.Module):
      ...   def __init__(self, din, dout, *, rngs: nnx.Rngs):
      ...     self.linear = nnx.Linear(din, dout, rngs=rngs)
      ...     self.dropout = nnx.Dropout(0.5)
      ...     self.batch_norm = nnx.BatchNorm(10, rngs=rngs)
      ...
      >>> block = Block(2, 5, rngs=nnx.Rngs(0))
      >>> block.dropout.deterministic, block.batch_norm.use_running_average
      (False, False)
      >>> block.eval()
      >>> block.dropout.deterministic, block.batch_norm.use_running_average
      (True, True)

    Args:
      **attributes: additional attributes passed to ``set_attributes``.
    """
    return self.set_attributes(
      deterministic=True,
      use_running_average=True,
      **attributes,
      raise_if_not_found=False,
    )


def set_mode(node: A, /, *, only: filterlib.Filter = ...,  **kwargs) -> A:
  """Creates a new node with static attributes updated according to **kwargs.
  The new node contains references to jax arrays in the original node.
  If a kwarg is not found in any module, this method raises a ValueError.

  Example::

    >>> from flax import nnx
    ...
    >>> class Block(nnx.Module):
    ...   def __init__(self, din, dout, *, rngs: nnx.Rngs):
    ...     self.linear = nnx.Linear(din, dout, rngs=rngs)
    ...     self.dropout = nnx.Dropout(0.5, deterministic=False)
    ...     self.batch_norm = nnx.BatchNorm(10, use_running_average=False, rngs=rngs)
    ...
    >>> block = Block(2, 5, rngs=nnx.Rngs(0))
    >>> block.dropout.deterministic, block.batch_norm.use_running_average
    (False, False)
    >>> new_block = nnx.set_mode(block, deterministic=True, use_running_average=True)
    >>> new_block.dropout.deterministic, new_block.batch_norm.use_running_average
    (True, True)

  ``Filter``'s can be used to set the attributes of specific Modules::

    >>> block = Block(2, 5, rngs=nnx.Rngs(0))
    >>> new_block = nnx.set_mode(block, only=nnx.Dropout, deterministic=True)
    >>> # Only the dropout will be modified
    >>> new_block.dropout.deterministic, new_block.batch_norm.use_running_average
    (True, False)

  Args:
    node: the object to create a copy of.
    only: Filters to select the Modules to set the attributes of.
    **kwargs: The attributes to set.
  """
  predicate = filterlib.to_predicate(only)

  counts = {k: 0 for k in kwargs}
  counts["_set_mode_calls"] = 0

  def _set_mode_fn(path, node):
    if hasattr(node, 'set_mode') and predicate(path, node):
      counts["_set_mode_calls"] += 1
      unused = node.set_mode(**kwargs)
      for k in unused:
        counts[k] += 1
    return node

  out = graph.recursive_map(_set_mode_fn, node)

  set_mode_calls = counts.pop("_set_mode_calls")
  unused_keys = [k for k, v in counts.items() if v == set_mode_calls]
  if unused_keys:
    raise ValueError(f"Unused keys found in set_mode: {unused_keys}")

  return out


def train_mode(node: A, /, *, only: filterlib.Filter = ..., **kwargs) -> A:
  """Creates a new node set to training mode.

  ``train`` uses ``set_mode`` to recursively set attributes ``deterministic=False``
  and ``use_running_average=False`` of all nested Modules that have these attributes.
  Its primarily used to control the runtime behavior of the ``Dropout`` and ``BatchNorm``
  Modules.

  Example::

    >>> from flax import nnx
    ...
    >>> class Block(nnx.Module):
    ...   def __init__(self, din, dout, *, rngs: nnx.Rngs):
    ...     self.linear = nnx.Linear(din, dout, rngs=rngs)
    ...     # initialize Dropout and BatchNorm in eval mode
    ...     self.dropout = nnx.Dropout(0.5, deterministic=True)
    ...     self.batch_norm = nnx.BatchNorm(10, use_running_average=True, rngs=rngs)
    ...
    >>> block = Block(2, 5, rngs=nnx.Rngs(0))
    >>> block.dropout.deterministic, block.batch_norm.use_running_average
    (True, True)
    >>> train_block = nnx.train_mode(block)
    >>> train_block.dropout.deterministic, train_block.batch_norm.use_running_average
    (False, False)

  Args:
    **kwargs: additional attributes passed to ``set_attributes``.
  """
  return set_mode(
      node,
      only=only,
      train=True,
      deterministic=False,
      use_running_average=False,
      **kwargs,
  )


def eval_mode(node: A, /, *, only: filterlib.Filter = ..., **kwargs) -> A:
  """Creates a new node set to evaluation mode.

  ``eval`` uses ``set_mode`` to recursively set attributes ``deterministic=True``
  and ``use_running_average=True`` of all nested Modules that have these attributes.
  Its primarily used to control the runtime behavior of the ``Dropout`` and ``BatchNorm``
  Modules.

  Example::

    >>> from flax import nnx
    ...
    >>> class Block(nnx.Module):
    ...   def __init__(self, din, dout, *, rngs: nnx.Rngs):
    ...     self.linear = nnx.Linear(din, dout, rngs=rngs)
    ...     self.dropout = nnx.Dropout(0.5)
    ...     self.batch_norm = nnx.BatchNorm(10, rngs=rngs)
    ...
    >>> block = Block(2, 5, rngs=nnx.Rngs(0))
    >>> block.dropout.deterministic, block.batch_norm.use_running_average
    (False, False)
    >>> eval_block = nnx.eval_mode(block)
    >>> eval_block.dropout.deterministic, eval_block.batch_norm.use_running_average
    (True, True)

  Args:
    **kwargs: additional attributes passed to ``set_mode``.
  """
  return set_mode(
      node,
      only=only,
      train=False,
      deterministic=True,
      use_running_average=True,
      **kwargs,
  )


def set_attributes(
    node: A, /, *, only: filterlib.Filter = ..., **attributes
) -> A:
  predicate = filterlib.to_predicate(only)

  def _set_attributes_fn(path, node):
    if predicate(path, node):
      for name, value in attributes.items():
        if hasattr(node, name):
          setattr(node, name, value)
    return node

  return graph.recursive_map(_set_attributes_fn, node)


def first_from(*args: tp.Optional[A], error_msg: str) -> A:
  """Return the first non-None argument.

  If all arguments are None, raise a ValueError with the given error message.

  Args:
    *args: the arguments to check
    error_msg: the error message to raise if all arguments are None
  Returns:
    The first non-None argument.
  """
  for arg in args:
    if arg is not None:
      return arg
  raise ValueError(error_msg)

def iter_modules(module: Module) -> tp.Iterator[tuple[PathParts, Module]]:
  """Recursively iterates over all nested :class:`Module`'s of the given Module, including
  the argument.

  Specifically, this function creates a generator that yields the path and the Module instance, where
  the path is a tuple of strings or integers representing the path to the Module from the
  root Module.

  Example::

    >>> from flax import nnx
    ...
    >>> class SubModule(nnx.Module):
    ...   def __init__(self, din, dout, rngs):
    ...     self.linear1 = nnx.Linear(din, dout, rngs=rngs)
    ...     self.linear2 = nnx.Linear(din, dout, rngs=rngs)
    ...
    >>> class Block(nnx.Module):
    ...   def __init__(self, din, dout, *, rngs: nnx.Rngs):
    ...     self.linear = nnx.Linear(din, dout, rngs=rngs)
    ...     self.submodule = SubModule(din, dout, rngs=rngs)
    ...     self.dropout = nnx.Dropout(0.5)
    ...     self.batch_norm = nnx.BatchNorm(10, rngs=rngs)
    ...
    >>> model = Block(2, 5, rngs=nnx.Rngs(0))
    >>> for path, module in nnx.iter_modules(model):
    ...   print(path, type(module).__name__)
    ...
    ('batch_norm',) BatchNorm
    ('dropout',) Dropout
    ('linear',) Linear
    ('submodule', 'linear1') Linear
    ('submodule', 'linear2') Linear
    ('submodule',) SubModule
    () Block
  """
  for path, value in graph.iter_graph(module):
    if isinstance(value, Module):
      yield path, value

def iter_children(module: Module) -> tp.Iterator[tuple[Key, Module]]:
  """Iterates over all children :class:`Module`'s of a given Module. This
  method is similar to :func:`iter_modules`, except it only iterates over the
  immediate children, and does not recurse further down.

  Specifically, this function creates a generator that yields the key and the Module instance,
  where the key is a string representing the attribute name of the Module to access
  the corresponding child Module.

  Example::

    >>> from flax import nnx
    ...
    >>> class SubModule(nnx.Module):
    ...   def __init__(self, din, dout, rngs):
    ...     self.linear1 = nnx.Linear(din, dout, rngs=rngs)
    ...     self.linear2 = nnx.Linear(din, dout, rngs=rngs)
    ...
    >>> class Block(nnx.Module):
    ...   def __init__(self, din, dout, *, rngs: nnx.Rngs):
    ...     self.linear = nnx.Linear(din, dout, rngs=rngs)
    ...     self.submodule = SubModule(din, dout, rngs=rngs)
    ...     self.dropout = nnx.Dropout(0.5)
    ...     self.batch_norm = nnx.BatchNorm(10, rngs=rngs)
    ...
    >>> model = Block(2, 5, rngs=nnx.Rngs(0))
    >>> for path, module in nnx.iter_children(model):
    ...  print(path, type(module).__name__)
    ...
    batch_norm BatchNorm
    dropout Dropout
    linear Linear
    submodule SubModule
  """
  node_impl = graph.get_node_impl(module)
  assert node_impl is not None
  node_dict = node_impl.node_dict(module)
  for key, value in node_dict.items():
    if isinstance(value, Module):
      yield key, value
