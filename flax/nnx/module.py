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

import inspect
import typing as tp

import jax
import jax.numpy as jnp

from flax.nnx import (
  filterlib,
  graphlib,
  pytreelib,
)
from flax.nnx import variablelib as variableslib
from flax.nnx.pytreelib import Pytree, PytreeMeta
from flax.nnx.graphlib import GraphState
from flax.nnx.statelib import split_state, State
import functools as ft
from flax.typing import Key, Path, PathParts
from collections.abc import MutableMapping
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
    """Store intermediate values during module execution for later extraction.

    Used with :func:`nnx.capture` decorator to collect intermediate values without
    explicitly passing containers through module calls. Values are stored under
    the specified ``name`` in a collection associated with ``variable_type``.

    By default, values are appended to a tuple, allowing multiple values to be
    tracked when the same module is called multiple times.

    Example usage::

      >>> from flax import nnx
      >>> import jax.numpy as jnp

      >>> class Model(nnx.Module):
      ...   def __init__(self, rngs):
      ...     self.linear1 = nnx.Linear(2, 3, rngs=rngs)
      ...     self.linear2 = nnx.Linear(3, 4, rngs=rngs)
      ...   def __call__(self, x):
      ...     x = self.linear1(x)
      ...     self.sow(nnx.Intermediate, 'features', x)
      ...     x = self.linear2(x)
      ...     return x

      >>> # With the capture decorator, sow returns intermediates
      >>> model = Model(rngs=nnx.Rngs(0))
      >>> @nnx.capture(nnx.Intermediate)
      ... def forward(model, x):
      ...   return model(x)
      >>> result, intermediates = forward(model, jnp.ones(2))
      >>> assert 'features' in intermediates

    Custom init/reduce functions can be passed to control accumulation::

      >>> class Model(nnx.Module):
      ...   def __init__(self, rngs):
      ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
      ...   def __call__(self, x):
      ...     x = self.linear(x)
      ...     self.sow(nnx.Intermediate, 'sum', x,
      ...              init_fn=lambda: 0,
      ...              reduce_fn=lambda prev, curr: prev+curr)
      ...     return x

    Args:
      variable_type: The :class:`Variable` type for the stored value.
        Typically :class:`Intermediate` or a subclass is used.
      name: A string key for storing the value in the collection.
      value: The value to be stored.
      reduce_fn: Function to combine existing and new values. Default appends
        to a tuple.
      init_fn: Function providing initial value for first ``reduce_fn`` call.
        Default is an empty tuple.
    """
    if isinstance(variable_type, str):
      variable_type = variableslib.variable_type_from_name(
          variable_type, allow_register=True
      )

    if hasattr(self, '__captures__'):
      for var in self.__captures__:
        if type(var) == variable_type:
          if name in var:
            var[name] = reduce_fn(var[name], value)
          else:
            var[name] = reduce_fn(init_fn(), value)
          return True
      else:
        return False
    elif hasattr(self, name):
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
        variable.set_value(reduce_fn(variable.get_value(), value))
    else:
      reduced_value = reduce_fn(init_fn(), value)
      setattr(self, name, variable_type(reduced_value))
    warnings.warn(
        """Using 'Module.sow()' outside of 'nnx.capture()' is deprecated; see
        https://flax.readthedocs.io/en/stable/capturing_intermediates.html for more information.
        """,
        DeprecationWarning,
        stacklevel=2,
      )
    return True

  def perturb(
      self,
      name: str,
      value: tp.Any,
      variable_type: (
          str | type[variableslib.Variable[tp.Any]]
      ) = variableslib.Perturbation,
  ):
    """Extract gradients of intermediate values during training.

    Used with :func:`nnx.capture` to record intermediate values in the forward pass
    and their gradients in the backward pass. Returns the value plus whatever perturbation
    is stored under ``name`` in the current capture context, allowing gradient computation via ``nnx.grad``.

    The workflow has four steps:
    1. Initialize perturbations with ``nnx.capture(model, nnx.Perturbation)``
    2. Run model with ``nnx.capture(model, nnx.Intermediate, init=perturbations)``
    3. Take gradients with respect to perturbations using ``nnx.grad``
    4. Combine results with ``nnx.merge_state(perturb_grads, intermediates)``

    .. note::
      This creates extra variables of the same size as ``value``, thus
      occupies more memory. Use it only to debug gradients in training.

    Example usage::

      >>> from flax import nnx
      >>> import jax.numpy as jnp

      >>> class Model(nnx.Module):
      ...   def __call__(self, x):
      ...     x2 = self.perturb('grad_of_x', x)
      ...     return 3 * x2

      >>> model = Model()
      >>> x = 1.0

      >>> # Step 1: Initialize perturbations
      >>> forward = nnx.capture(model, nnx.Perturbation)
      >>> _, perturbations = forward(x)

      >>> # Steps 2-4: Capture gradients
      >>> def train_step(model, perturbations, x):
      ...   def loss(model, perturbations, x):
      ...     return nnx.capture(model, nnx.Intermediate, init=perturbations)(x)
      ...   (grads, perturb_grads), sowed = nnx.grad(loss, argnums=(0, 1), has_aux=True)(model, perturbations, x)
      ...   return nnx.merge_state(perturb_grads, sowed)

      >>> metrics = train_step(model, perturbations, x)
      >>> # metrics contains gradients of intermediate values

    Args:
      name: A string key for storing the perturbation value.
      value: The intermediate value to capture gradients for. You must use
        the returned value (not the original) for gradient capturing to work.
      variable_type: The :class:`Variable` type for the stored perturbation.
        Default is :class:`nnx.Perturbation`.
    """
    if isinstance(variable_type, str):
      variable_type = variableslib.variable_type_from_name(
          variable_type, allow_register=True
      )

    if hasattr(self, '__captures__'):
      for var in self.__captures__:
        if type(var) == variable_type:
          if name not in var:
            zeros = jax.tree.map(jnp.zeros_like, value)
            var[name] = zeros
          old_value = var[name]
          return old_value + value
      else:
        return value
    elif hasattr(self, name):
      var = getattr(self, name)
      if not isinstance(var, variable_type):
        raise ValueError(
          f"Expected '{name}' to be of type '{variable_type.__name__}', "
          f"got '{type(var).__name__}'"
        )
      old_value = var.get_value()
    else:
      old_value = jax.tree.map(jnp.zeros_like, value)
      setattr(self, name, variable_type(old_value))
    warnings.warn("""
      Using 'Module.perturb()' outside of 'nnx.capture()' is deprecated; see
      https://flax.readthedocs.io/en/stable/capturing_intermediates.html for more information.
      """,
      DeprecationWarning,
      stacklevel=2)
    return old_value + value

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
    graph: bool | None = None,
    **attributes: tp.Any,
  ) -> None:
    """Sets the attributes of nested :class:`flax.nnx.Module`'s including the current
    ``nnx.Module``. If the attribute is not found in the ``nnx.Module``, it is ignored.

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

    ``Filter``'s (``flax.nnx.filterlib``) can be used to set the attributes of specific
    ``nnx.Module``'s::

      >>> block = Block(2, 5, rngs=nnx.Rngs(0))
      >>> block.set_attributes(nnx.Dropout, deterministic=True)
      >>> # Only the dropout will be modified
      >>> block.dropout.deterministic, block.batch_norm.use_running_average
      (True, False)

    Args:
      *filters: NNX ``Filter``'s to select the :class:`flax.nnx.Module`'s whose attributes will be to be set.
      raise_if_not_found: If ``True`` (default), raises a ValueError if at least one attribute
        instance is not found in one of the selected Modules.
      **attributes: The attributes to set.
    """
    remaining_attributes = set(attributes.keys())
    if not filters:
      filters = (True,)
    predicates = tuple(map(filterlib.to_predicate, filters))
    for path, module in iter_modules(self, graph=graph):
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
        'Could not find at least one instance of the following'
        f' attributes: {sorted(remaining_attributes)}'
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

def view(node: A, /, *, only: filterlib.Filter = ..., raise_if_not_found: bool = True, graph: bool | None = None, **kwargs) -> A:
  """Creates a new node with static attributes updated according to ``**kwargs``.

  The new node contains references to jax arrays in the original node. If a
  kwarg is not found in any module, this method raises a ValueError. Uses the
  ``set_view`` class method in nnx.Modules. ``set_view`` class methods should
  return any unused kwargs.

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
    >>> new_block = nnx.view(block, deterministic=True, use_running_average=True)
    >>> new_block.dropout.deterministic, new_block.batch_norm.use_running_average
    (True, True)

  ``Filter``'s can be used to set the attributes of specific Modules::
    >>> block = Block(2, 5, rngs=nnx.Rngs(0))
    >>> new_block = nnx.view(block, only=nnx.Dropout, deterministic=True)
    >>> # Only the dropout will be modified
    >>> new_block.dropout.deterministic, new_block.batch_norm.use_running_average
    (True, False)

  Args:
    node: the object to create a copy of.
    only: Filters to select the Modules to set the attributes of.
    graph: If ``True`` (default), uses graph-mode which supports the full
      NNX feature set including shared references. If ``False``, uses
      tree-mode which treats Modules as regular JAX pytrees, avoiding
      the overhead of the graph protocol.
    **kwargs: The attributes to set.
  """
  predicate = filterlib.to_predicate(only)

  remaining = set(kwargs)

  def _set_mode_fn(path, node):
    if hasattr(node, 'set_view') and predicate(path, node):
      sig = inspect.signature(node.set_view)
      has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
      )
      if has_var_keyword:
        node.set_view(**kwargs)
        remaining.clear()
      else:
        named_params = {
          name
          for name, p in sig.parameters.items()
          if p.kind
          in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
          )
        }
        filtered_kwargs = {
          k: v for k, v in kwargs.items() if k in named_params
        }
        node.set_view(**filtered_kwargs)
        remaining.difference_update(named_params)
    return node

  out = graphlib.recursive_map(_set_mode_fn, node, graph=graph)

  if raise_if_not_found and remaining:
    raise ValueError(f"Unused keys found in nnx.view: {sorted(remaining)}")

  return out

def with_attributes(
  node: A,
  /,
  *,
  only: filterlib.Filter = ...,
  raise_if_not_found: bool = True,
  graph: bool | None = None,
  **attributes: tp.Any,
) -> A:
  """Creates a new node with attributes updated according to ``**attributes``.

  The new node contains references to jax arrays in the original node. Unlike
  ``set_attributes``, this function does not modify the original node.

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
    >>> new_block = nnx.with_attributes(block, deterministic=True, use_running_average=True)
    >>> new_block.dropout.deterministic, new_block.batch_norm.use_running_average
    (True, True)
    >>> block.dropout.deterministic, block.batch_norm.use_running_average
    (False, False)

  ``Filter``'s can be used to set the attributes of specific Modules::
    >>> block = Block(2, 5, rngs=nnx.Rngs(0))
    >>> new_block = nnx.with_attributes(block, only=nnx.Dropout, deterministic=True)
    >>> # Only the dropout will be modified
    >>> new_block.dropout.deterministic, new_block.batch_norm.use_running_average
    (True, False)

  Args:
    node: the object to create a copy of.
    only: Filters to select the Modules to set the attributes of.
    raise_if_not_found: If True (default), raises a ValueError if at least one
      attribute instance is not found in one of the selected Modules.
    graph: If ``True`` (default), uses graph-mode which supports the full
      NNX feature set including shared references. If ``False``, uses
      tree-mode which treats Modules as regular JAX pytrees, avoiding
      the overhead of the graph protocol.
    **attributes: The attributes to set.
  """
  predicate = filterlib.to_predicate(only)
  remaining_attributes = set(attributes.keys())

  def _set_attributes_fn(path, node):
    if isinstance(node, Module) and predicate(path, node):
      for name, value in attributes.items():
        if hasattr(node, name):
          setattr(node, name, value)
          remaining_attributes.discard(name)
    return node

  out = graphlib.recursive_map(_set_attributes_fn, node, graph=graph)

  if remaining_attributes and raise_if_not_found:
    raise ValueError(
      'Could not find at least one instance of the '
      f'following attributes: {sorted(remaining_attributes)}'
    )

  return out

def _parse_docstring_args(doc_str: str) -> dict[str, str]:
  """Parses parameters from `Args:` section of a function docstring.
  Assumes Google style docstrings. Returns a dictionary with
  keys representing argument names and values representing descriptions.
  Each description has lines starting with 4 spaces.
  """
  lines = doc_str.split("\n")

  # Get lines with the parameter names
  inds = [i for i, l in enumerate(lines) if l.startswith("  ") and not l.startswith("    ")]
  inds.append(len(lines))
  out = dict()

  # Parse each argument
  for i in range(len(inds)-1):
    start, end = inds[i], inds[i+1]

    # Process first line for the description
    first_colon = lines[start].find(":")
    name = lines[start][:first_colon].strip()
    desc = [" "*4 + lines[start][first_colon+1:].strip()]

    # Append remaining description lines
    for j in range(start+1, end):
      desc.append(lines[j])

    out[name] = "\n".join(desc)
  return out



def view_info(node: Module, /, *, only: filterlib.Filter = ..., graph: bool | None = None) -> str:
  """Provides information about the ``view`` arguments for a module and all
  submodules. If no docstring is provided for a module's `set_view`, this function
  puts the `set_view` signature below the function.

  Example::
    >>> from flax import nnx
    ...
    >>> class CustomModel(nnx.Module):
    ...   def __init__(self, *, rngs):
    ...       self.mha = nnx.MultiHeadAttention(4, 8, 32, rngs=rngs)
    ...       self.drop = nnx.Dropout(0.5, rngs=rngs)
    ...       self.bn = nnx.BatchNorm(32, rngs=rngs)
    ...
    >>> model = CustomModel(rngs=nnx.Rngs(0))
    >>> print(nnx.view_info(model))
    BatchNorm:
      use_running_average: bool | None = None
        if True, the stored batch statistics will be
        used instead of computing the batch statistics on the input.
    Dropout:
      deterministic: bool | None = None
        if True, disables dropout masking.
    MultiHeadAttention:
      deterministic: bool | None = None
        if True, the module is set to deterministic mode.
      decode: bool | None = None
        if True, the module is set to decode mode.
      batch_size: int | Shape | None = None
        the batch size to use for the cache.
      max_length: int | None = None
        the max length to use for the cache.

  Args:
    node: the object to display ``view`` information for.
    only: Filters to select the Modules to display information for.
    graph: If ``True`` (default), uses graph-mode which supports the full
      NNX feature set including shared references. If ``False``, uses
      tree-mode which treats Modules as regular JAX pytrees, avoiding
      the overhead of the graph protocol.
  """
  predicate = filterlib.to_predicate(only)
  classes: set[Module] = set()

  def _set_mode_info_fn(path, node):
    if hasattr(node, 'set_view') and predicate(path, node):
      classes.add(node.__class__)
    return node

  graphlib.recursive_map(_set_mode_info_fn, node, graph=graph)

  class_list = sorted(list(classes), key=lambda x: x.__qualname__)
  out_str = []
  for c in class_list:
    out_str.append(f"{c.__qualname__}:")
    sig = inspect.signature(c.set_view)
    doc = inspect.getdoc(c.set_view)

    # Parse docstring
    if isinstance(doc, str):
      start, end = doc.find("Args:\n"), doc.find("Returns:\n")
      if end == -1:
        end = len(doc)
      doc = doc[start+6:end]
      parsed_docstring = _parse_docstring_args(doc)

      # Generate output from signature and docstring
      skip_names = {"self", "args", "kwargs"}
      for name, param in sig.parameters.items():
        if name in skip_names:
          continue

        if param.default is inspect.Parameter.empty:
          out_str.append(f"  {name}: {param.annotation}")
        else:
          out_str.append(f"  {name}: {param.annotation} = {param.default}")
        out_str.append(parsed_docstring[name])
    else:
      out_str.append(f"  set_view{sig}")


  return "\n".join(out_str)

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

def iter_modules(
  module: Module, /, *, graph: bool | None = None,
) -> tp.Iterator[tuple[PathParts, Module]]:
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

  Args:
    module: A :class:`Module` object.
    graph: If ``True`` (default), uses graph-mode which supports the full
      NNX feature set including shared references. If ``False``, uses
      tree-mode which treats Modules as regular JAX pytrees, avoiding
      the overhead of the graph protocol.
  """
  for path, value in graphlib.iter_graph(module, graph=graph):
    if isinstance(value, Module):
      yield path, value

iter_children = graphlib.iter_children

P = tp.ParamSpec("P")
R = tp.TypeVar("R")

@tp.overload
def capture(
  fn: tp.Callable[P, R],
  *var_types: type[variableslib.Variable],
  init: tp.Optional[State] = None,
  method_outputs: tp.Optional[type[variableslib.Variable]] = None
) -> tp.Callable[P, tuple[R, State]]: ...

@tp.overload
def capture(
  fn: type[variableslib.Variable],
  *var_types: type[variableslib.Variable],
  init: tp.Optional[State] = None,
  method_outputs: tp.Optional[type[variableslib.Variable]] = None
) -> tp.Callable[[tp.Callable[P, R]], tp.Callable[P, tuple[R, State]]]: ...

def capture(fn: tp.Callable[P, R] | type[variableslib.Variable], *var_types: type[variableslib.Variable],
  init : tp.Optional[State] = None,
  method_outputs : tp.Optional[type[variableslib.Variable]] = None
) -> tp.Callable[P, tuple[R, State]] | tp.Callable[[tp.Callable[P, R]], tp.Callable[P, tuple[R, State]]]:
    """Wraps a function to capture intermediate values from a module during execution.

    This function wraps a `Callable`, executing it while collecting intermediate values that were stored using
    ``Module.sow()`` or ``Module.perturb()``.

    The `fn` argument can be either a function, a Module instance, or a bound method.
    If `fn` is a function, its first argument should be the module in which intermediate values are to be recorded.
    If `fn` is a bound method, the module used for storage is inferred from the instance.
    If `fn` is a Module, its `__call__` method will be wrapped.

    Args:
      fn: The `Callable` to wrap.
      var_types: Variable types to capture. If None, defaults to [].
      init: MutableMapping used to initialize perturbation values. This is useful for gradient extraction.
      method_outputs: If provided, automatically sows the output of each method
        in the module and its submodules using this variable type.

    Returns:
      A wrapped function that returns
      a tuple of (result, *intermediates) where result is the output of the function
      and each intermediate is a State containing the captured values with the corresponding type in `var_types`.

    Example with manual sowing::

      class Foo(nnx.Module):
        def __call__(self, x):
          self.sow(nnx.Intermediate, 'features', x)
          return x

      model = Foo(rngs=nnx.Rngs(0))
      forward = nnx.capture(model, nnx.Intermediate)
      result, intermediates = forward(x)
      # intermediates['features'] contains the sowed value

    Example with method outputs::

      class Foo(nnx.Module):
        def features(self, x):
          return x
        def classifier(self, x):
          return x
        def __call__(self, x):
          return self.classifier(self.features(x))

      model = Foo(rngs=nnx.Rngs(0))
      result, intermediates = nnx.capture(
        model, method_output_type=nnx.Intermediate)(x)
      # intermediates contains outputs of features(), classifier(), and __call__()

    Example with gradient extraction::

      class Model(nnx.Module):
        def __call__(self, x):
          x2 = self.perturb('grad_of_x', x)
          return 3 * x2

      model = Model()
      forward = nnx.capture(lambda model, x: model(x), nnx.Perturbation) # Initialize perturbations
      _, perturbations = forward_capture(model, x)

      # Compute gradients with respect to perturbations
      loss = nnx.capture(forward, init=perturbations)
      grads, sowed = nnx.grad(loss, has_aux=True)(model, perturbations, x)
    """

    # Handle partial evaluation when first arg is a Variable type
    if isinstance(fn, type) and issubclass(fn, variableslib.Variable):
      # Partial application: return a function that waits for the actual fn
      all_var_types = (fn,) + var_types
      def partial_capture(actual_fn: tp.Callable[P, R] | Module) -> tp.Callable[P, tuple[R, State]]:
        return capture(actual_fn, *all_var_types, init=init, method_outputs=method_outputs)
      return partial_capture

    # Handle bound methods and callable Modules
    module_instance = None
    if inspect.ismethod(fn) and isinstance(fn.__self__, Module):
      module_instance = fn.__self__
    elif isinstance(fn, Module):
      module_instance = fn

    ft.wraps(fn)
    def wrapper(*fn_args, **kwargs):
      if module_instance is None:
        module = fn_args[0]
      else:
        module = module_instance

      # Extract initial values from state
      state_by_path = _collect_state_by_path(init) if init else {}

      # Initialize __captures__ as a tuple of Variables (one per type)
      for path, m in iter_modules(module):
        # Create initial dicts for each variable type
        initial_dicts = {}
        for var_type in var_types:
          initial_dicts[var_type] = {}

        # Populate from state if available
        if path in state_by_path:
          for name, var in state_by_path[path].items():
            var_type = type(var)
            if var_type not in initial_dicts:
              initial_dicts[var_type] = {}
            initial_dicts[var_type][name] = var.get_value()

        # Create the captures tuple
        captures_tuple = tuple(k(v) for (k,v) in initial_dicts.items())
        m.__captures__ = pytreelib.data(captures_tuple)

      # Wrap methods with capturing if required
      if method_outputs:
        for _, m in iter_modules(module):
          _add_capturing(type(m), method_outputs)

      try:
        result = fn(*fn_args, **kwargs)
      finally:

        # Undo method sowing modification
        for _, m in iter_modules(module):
          _remove_capturing(type(m))

      # Extract intermediates manually from __captures__
      interms = State({})
      _extract_captures(module, interms, set(var_types))
      if len(var_types) == 0:
          return result
      split_states = split_state(interms, *var_types)
      if len(var_types) == 1:
        return result, split_states
      else:
        return (result, *split_states)

    return wrapper

def _collect_state_by_path(state):
  """Build a mapping from module path to state Variables."""
  state_by_path = {}

  def collect(s, path_parts):
    if isinstance(s, MutableMapping):
      for key, value in s.items():
        if isinstance(value, variableslib.Variable):
          path_tuple = tuple(path_parts)
          if path_tuple not in state_by_path:
            state_by_path[path_tuple] = {}
          state_by_path[path_tuple][key] = value
        elif isinstance(value, MutableMapping):
          collect(value, path_parts + [key])

  collect(state, [])
  return state_by_path

def _navigate_to_path(state, path):
  """Navigate to a nested path in state, creating dicts as needed."""
  current = state
  for part in path:
    if part not in current:
      current[part] = State({})
    current = current[part]
  return current

def _extract_captures(module, state, var_types):
  """Extract intermediates from __captures__ tuple into state dict."""
  for path, mod in iter_modules(module):
    if hasattr(mod, '__captures__'):
      captures_tuple = mod.__captures__
      for var in captures_tuple:
        if not type(var) in var_types:
          continue
        current = _navigate_to_path(state, path)
        for key, value in var.items():
          current[key] = type(var)(value)
      delattr(mod, '__captures__')


def _add_capturing(cls, variable_type):
  """Adds capturing to methods of a Module.
  Does not instrument superclass methods."""
  for name, method in cls.__dict__.items():
    if callable(method) and (not name.startswith('_') or name == '__call__'):
      if not hasattr(method, '_does_capturing'):
        def closure(name, method): # Necessary to make 'name' immutable during iteration
          @ft.wraps(method)
          def wrapper(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            self.sow(variable_type, name, result)
            return result
          wrapper._does_capturing = True
          setattr(cls, name, wrapper)
        closure(name, method)
  return cls

def _remove_capturing(cls):
  """Remove capturing methods from a Module."""
  for name, method in cls.__dict__.items():
    if hasattr(method, '_does_capturing'):
      setattr(cls, name, method.__wrapped__)
  return cls
