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
from __future__ import annotations

import jax, jax.numpy as jnp
from flax.experimental import nnx

import typing as tp

#TODO: add tests and docstrings

class Metric(nnx.Module):
  def __init__(self):
    raise NotImplementedError('Must override `__init__()` method.')
  def empty(self):
    raise NotImplementedError('Must override `empty()` method.')
  def update(self):
    raise NotImplementedError('Must override `update()` method.')
  def compute(self):
    raise NotImplementedError('Must override `compute()` method.')

class Average(Metric):
  def __init__(self):
    self.total = nnx.Variable(jnp.array(0, dtype=jnp.float32))
    self.count = nnx.Variable(jnp.array(0, dtype=jnp.int32))
  def empty(self):
    self.total = self.total.replace(raw_value=jnp.array(0, dtype=jnp.float32))
    self.count = self.count.replace(raw_value=jnp.array(0, dtype=jnp.int32))
  def update(self, *, values: tp.Union[int, float, jax.Array], **_):
    self.total = self.total.replace(raw_value=self.total.value + (values if isinstance(values, (int, float)) else values.sum()))
    self.count = self.count.replace(raw_value=self.count.value + (1 if isinstance(values, (int, float)) else values.size))
  def compute(self):
    return self.total.value / self.count.value

class Accuracy(Average):
  def update(self, *, logits: jax.Array, labels: jax.Array, **_):
    super().update(values=(logits==labels))

class MultiMetric(Metric):
  '''MultiMetric class to store multiple metrics and update them in a single call.

  Example usage::

    >>> import jax.numpy as jnp
    >>> from flax.experimental import nnx

    >>> logits = jnp.array([1, 0, 0, 1, 0])
    >>> labels = jnp.array([1, 1, 0, 1, 0])
    >>> logits2 = jnp.array([0, 0, 0, 1, 1])
    >>> labels2 = jnp.array([0, 1, 1, 1, 1])

    >>> batch_loss = jnp.array([1, 2, 3, 4])
    >>> batch_loss2 = jnp.array([3, 2, 1, 0])

    >>> metrics = nnx.metrics.MultiMetric(
    ...   accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average()
    >>> )
    >>> metrics.compute()
    {'accuracy': Array(nan, dtype=float32), 'loss': Array(nan, dtype=float32)}
    >>> metrics.update(logits=logits, labels=labels, values=batch_loss)
    >>> metrics.compute()
    {'accuracy': Array(0.8, dtype=float32), 'loss': Array(2.5, dtype=float32)}
    >>> metrics.update(logits=logits2, labels=labels2, values=batch_loss2)
    >>> metrics.compute()
    {'accuracy': Array(0.7, dtype=float32), 'loss': Array(2., dtype=float32)}
    >>> metrics.empty()
    >>> metrics.compute()
    {'accuracy': Array(nan, dtype=float32), 'loss': Array(nan, dtype=float32)}
  '''
  def __init__(self, **metrics):
    # TODO: raise error if a kwarg is passed that is in ('empty', 'update', 'compute'), since these names are reserved for methods
    self._metric_names = []
    for metric_name, metric in metrics.items():
      self._metric_names.append(metric_name)
      vars(self)[metric_name] = metric
  def empty(self):
    for metric_name in self._metric_names:
      getattr(self, metric_name).empty()
  def update(self, **updates):
    # TODO: should we give the option of updating only some of the metrics and not all? e.g. if for some kwargs==None, don't do update
    # TODO: should we raise an error if a kwarg is passed into **updates that has no match with any underlying metric? e.g. user typo
    for metric_name in self._metric_names:
      getattr(self, metric_name).update(**updates)
  def compute(self):
    return {f'{metric_name}': getattr(self, metric_name).compute() for metric_name in self._metric_names}