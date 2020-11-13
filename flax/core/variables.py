# Copyright 2020 The Flax Authors.
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

"""A variable dict is a normal Python dictionary, which is a container for one
or more "variable collections", each of which are nested dictionaries whose
leaves are ``jax.numpy`` arrays.

The different variable collections share the same nested tree structure.

For example, consider the following variable dictionary::

  {
    "params": {
      "Conv1": { "weight": ..., "bias": ... },
      "BatchNorm1": { "scale": ..., "mean": ... },
      "Conv2": {...}
    },
    "batch_stats": {
      "BatchNorm1": { "moving_mean": ..., "moving_average": ...}
    }
  }

In this case, the ``"BatchNorm1"`` key lives in both the ``"params"`` and
```"batch_stats""`` collections. This reflects the fact that the submodule
named ``""BatchNorm1""`` has both trainable parameters (the ``"params"`` collection),
as well as other non-trainvable variables (the ``"batch_stats"`` collection)

TODO: Make "variable dict" design note, and link to it from here.
"""

import abc
from .frozen_dict import FrozenDict
from typing import Generic, TypeVar, Any, Dict

VariableCollection = Dict[str, Any]
VariableDict = Dict[str, VariableCollection]

T = TypeVar('T')
class Variable(Generic[T]):
  """A Variable object allows mutable access to a variable in a VariableDict.
  
  Variables are identified by a collection (e.g., "batch_stats") and a name 
  (e.g., "moving_mean"). The value property gives access to the variable's 
  content and can be assigned to for mutation.
  """

  def __init__(self, scope: 'Scope', collection: str, name: str):
    """Initializes a variable.

    Args:
      scope: The scope in which the variable is stored.
      collection: The collection of the variable (e.g., "params").
      name: The name of the variable (e.g., "dense").
    """
    self.scope = scope
    self.collection = collection
    self.name = name

  @property
  def value(self) -> T:
    """Returns the value of this Variable."""
    return self.scope.get_variable(self.collection, self.name)

  @value.setter
  def value(self, value: T):
    """Updates the value of this Variable."""
    self.scope.put_variable(self.collection, self.name, value)