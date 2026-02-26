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

import numpy as np

from flax import struct
from flax.nnx import filterlib, graph
from flax.nnx.pytreelib import Pytree
from flax.nnx.variablelib import Variable
import jax, jax.numpy as jnp


_MULTIMETRIC_RESERVED_NAMES = frozenset({
  'reset', 'update', 'compute', 'split',
  '_metric_names', '_expected_kwargs',
})


class MetricState(Variable):
  """Wrapper class for Metric Variables."""

  pass


class Metric(Pytree):
  """Base class for metrics. Any class that subclasses ``Metric`` should
  implement a ``compute``, ``reset`` and ``update`` method."""

  def __init__(self):
    raise NotImplementedError('Must override `__init__()` method.')

  def reset(self) -> None:
    """In-place reset the ``Metric``."""
    raise NotImplementedError('Must override `reset()` method.')

  def update(self, **kwargs) -> None:
    """In-place update the ``Metric``."""
    raise NotImplementedError('Must override `update()` method.')

  def compute(self):
    """Compute and return the value of the ``Metric``."""
    raise NotImplementedError('Must override `compute()` method.')

  def split(self, *filters: filterlib.Filter):
    return graph.split(self, *filters)


class Average(Metric):
  """Average metric.

  Example usage::

    >>> import jax.numpy as jnp
    >>> from flax import nnx

    >>> batch_loss = jnp.array([1, 2, 3, 4])
    >>> batch_loss2 = jnp.array([3, 2, 1, 0])

    >>> metrics = nnx.metrics.Average()
    >>> metrics.compute()
    Array(nan, dtype=float32)
    >>> metrics.update(values=batch_loss)
    >>> metrics.compute()
    Array(2.5, dtype=float32)
    >>> metrics.update(values=batch_loss2)
    >>> metrics.compute()
    Array(2., dtype=float32)
    >>> metrics.reset()
    >>> metrics.compute()
    Array(nan, dtype=float32)
  """

  def __init__(self, argname: str = 'values'):
    """Pass in a string denoting the key-word argument that :func:`update` will use to derive the new value.
    For example, constructing the metric as ``avg = Average('test')`` would allow you to make updates with
    ``avg.update(test=new_value)``.

    Args:
      argname: an optional string denoting the key-word argument that
        :func:`update` will use to derive the new value. Defaults to
        ``'values'``.
    """
    self.argname = argname
    self.total = MetricState(jnp.array(0, dtype=jnp.float32))
    self.count = MetricState(jnp.array(0, dtype=jnp.int32))

  def reset(self) -> None:
    """Reset this ``Metric``."""
    self.total[...] = jnp.array(0, dtype=jnp.float32)
    self.count[...] = jnp.array(0, dtype=jnp.int32)

  def update(self, **kwargs) -> None:
    """In-place update this ``Metric``. This method will use the value from
    ``kwargs[self.argname]`` to update the metric, where ``self.argname`` is
    defined on construction.

    Args:
      **kwargs: the key-word arguments that contains a ``self.argname``
        entry that maps to the value we want to use to update this metric.
    """
    if self.argname not in kwargs:
      raise TypeError(f"Expected keyword argument '{self.argname}'")
    values: tp.Union[int, float, jax.Array] = kwargs[self.argname]
    self.total[...] += (
      values if isinstance(values, (int, float)) else values.sum()
    )
    self.count[...] += 1 if isinstance(values, (int, float)) else values.size

  def compute(self) -> jax.Array:
    """Compute and return the average."""
    return self.total / self.count


@struct.dataclass
class Statistics:
  """Running statistics computed by the Welford algorithm.

  Attributes:
    mean: the running mean of the data.
    standard_error_of_mean: the standard error of the mean.
    standard_deviation: the population standard deviation
      (ddof=0) of the data.
  """

  mean: jnp.float32
  standard_error_of_mean: jnp.float32
  standard_deviation: jnp.float32


class Welford(Metric):
  """Uses Welford's algorithm to compute the mean and variance of a stream of data.

  Example usage::

    >>> import jax.numpy as jnp
    >>> from flax import nnx

    >>> batch_loss = jnp.array([1, 2, 3, 4])
    >>> batch_loss2 = jnp.array([3, 2, 1, 0])

    >>> metrics = nnx.metrics.Welford()
    >>> metrics.compute()
    Statistics(mean=Array(0., dtype=float32), standard_error_of_mean=Array(nan, dtype=float32), standard_deviation=Array(nan, dtype=float32))
    >>> metrics.update(values=batch_loss)
    >>> metrics.compute()
    Statistics(mean=Array(2.5, dtype=float32), standard_error_of_mean=Array(0.559017, dtype=float32), standard_deviation=Array(1.118034, dtype=float32))
    >>> metrics.update(values=batch_loss2)
    >>> metrics.compute()
    Statistics(mean=Array(2., dtype=float32), standard_error_of_mean=Array(0.43301272, dtype=float32), standard_deviation=Array(1.2247449, dtype=float32))
    >>> metrics.reset()
    >>> metrics.compute()
    Statistics(mean=Array(0., dtype=float32), standard_error_of_mean=Array(nan, dtype=float32), standard_deviation=Array(nan, dtype=float32))
  """

  def __init__(self, argname: str = 'values'):
    """Pass in a string denoting the key-word argument that :func:`update` will use to derive the new value.
    For example, constructing the metric as ``wf = Welford('test')`` would allow you to make updates with
    ``wf.update(test=new_value)``.

    Args:
      argname: an optional string denoting the key-word argument that
        :func:`update` will use to derive the new value. Defaults to
        ``'values'``.
    """
    self.argname = argname
    self.count = MetricState(jnp.array(0, dtype=jnp.int32))
    self.mean = MetricState(jnp.array(0, dtype=jnp.float32))
    self.m2 = MetricState(jnp.array(0, dtype=jnp.float32))

  def reset(self) -> None:
    """Reset this ``Metric``."""
    self.count[...] = jnp.array(0, dtype=jnp.uint32)
    self.mean[...] = jnp.array(0, dtype=jnp.float32)
    self.m2[...] = jnp.array(0, dtype=jnp.float32)

  def update(self, **kwargs) -> None:
    """In-place update this ``Metric``. This method will use the value from
    ``kwargs[self.argname]`` to update the metric, where ``self.argname`` is
    defined on construction.

    Args:
      **kwargs: the key-word arguments that contains a ``self.argname``
        entry that maps to the value we want to use to update this metric.
    """
    if self.argname not in kwargs:
      raise TypeError(f"Expected keyword argument '{self.argname}'")
    values: tp.Union[int, float, jax.Array] = kwargs[self.argname]
    count = 1 if isinstance(values, (int, float)) else values.size
    original_count = self.count[...]
    self.count[...] += count
    delta = (
      values if isinstance(values, (int, float)) else values.mean()
    ) - self.mean
    self.mean[...] += delta * count / self.count
    m2 = 0.0 if isinstance(values, (int, float)) else values.var() * count
    self.m2[...] += m2 + delta * delta * count * original_count / self.count

  def compute(self) -> Statistics:
    """Compute and return the mean and variance statistics in a
    ``Statistics`` dataclass object.
    """
    variance = self.m2 / self.count
    standard_deviation = variance**0.5
    sem = standard_deviation / (self.count**0.5)
    return Statistics(
      mean=self.mean[...],
      standard_error_of_mean=sem,
      standard_deviation=standard_deviation,
    )


class Accuracy(Average):
  """Accuracy metric. This metric subclasses :class:`Average`,
  and so they share the same ``reset`` and ``compute`` method
  implementations. Unlike :class:`Average`, no string needs to
  be passed to ``Accuracy`` during construction.

  Example usage::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> logits = jax.random.normal(jax.random.key(0), (5, 2))
    >>> labels = jnp.array([0, 1, 1, 1, 0])
    >>> logits2 = jax.random.normal(jax.random.key(1), (5, 2))
    >>> labels2 = jnp.array([0, 1, 1, 1, 1])

    >>> metrics = nnx.metrics.Accuracy()
    >>> metrics.compute()
    Array(nan, dtype=float32)
    >>> metrics.update(logits=logits, labels=labels)
    >>> metrics.compute()
    Array(0.6, dtype=float32)
    >>> metrics.update(logits=logits2, labels=labels2)
    >>> metrics.compute()
    Array(0.4, dtype=float32)
    >>> metrics.reset()
    >>> metrics.compute()
    Array(nan, dtype=float32)

    >>> logits3 = jax.random.normal(jax.random.key(2), (5,))
    >>> labels3 = jnp.array([0, 1, 0, 1, 1])
    >>> accuracy = nnx.metrics.Accuracy(threshold=0.5)
    >>> accuracy.update(logits=logits3, labels=labels3)
    >>> accuracy.compute()
    Array(0.8, dtype=float32)
  """

  def __init__(self, threshold: float | None = None, *args, **kwargs):
    """For binary classification, pass in a float denoting a threshold to determine if a
    prediction is positive. For example, constructing the metric as
    ``acc = Accuracy(threshold=0.5)`` would cause any logit greater than or equal to 0.5
    to be interpreted as a positive classification. For multi-class classification, do
    not pass in a threshold.

    Args:
      threshold: for binary classification, determines if a prediction is
        positive. Defaults to None.
    """
    if (threshold is not None) and (not isinstance(threshold, float)):
      raise TypeError(f'Expected threshold to be a float, got {type(threshold)}')

    self.threshold = threshold
    super().__init__(*args, **kwargs)

  def update(self, *, logits: jax.Array, labels: jax.Array, **_) -> None:  # type: ignore[override]
    """In-place update this ``Metric``.

    Args:
      logits: the outputted predicted activations. For multi-class
        classification, these values are argmax-ed (on the trailing
        dimension), before comparing them to the labels. For binary
        classification, these values are compared to the labels directly.
      labels: the ground truth integer labels.
    """
    if self.threshold is not None:  # Binary classification case
      if logits.ndim != labels.ndim:
        raise ValueError(
          'For binary classification, expected logits.ndim==labels.ndim, got '
          f'{logits.ndim} and {labels.ndim}'
        )
    elif logits.ndim != labels.ndim + 1:  # Multi-class classification case
      raise ValueError(
        'For multi-class classification, expected logits.ndim==labels.ndim+1, '
        f'got {logits.ndim} and {labels.ndim}'
      )

    if labels.dtype in (jnp.int64, np.int32, np.int64):
      labels = jnp.astype(labels, jnp.int32)
    elif labels.dtype != jnp.int32:
      raise ValueError(f'Expected labels.dtype==jnp.int32, got {labels.dtype}')

    if self.threshold is not None:  # Binary classification case
      super().update(values=((logits >= self.threshold) == (labels > 0)))
      return

    # Multi-class classification case
    super().update(values=(logits.argmax(axis=-1) == labels))


class MultiMetric(Metric):
  """MultiMetric class to store multiple metrics and update them in a single call.

  Example usage::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> metrics = nnx.MultiMetric(
    ...   accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average()
    ... )

    >>> metrics
    MultiMetric( # MetricState: 4 (16 B)
      accuracy=Accuracy( # MetricState: 2 (8 B)
        threshold=None,
        argname='values',
        total=MetricState( # 1 (4 B)
          value=Array(0., dtype=float32)
        ),
        count=MetricState( # 1 (4 B)
          value=Array(0, dtype=int32)
        )
      ),
      loss=Average( # MetricState: 2 (8 B)
        argname='values',
        total=MetricState( # 1 (4 B)
          value=Array(0., dtype=float32)
        ),
        count=MetricState( # 1 (4 B)
          value=Array(0, dtype=int32)
        )
      )
    )

    >>> metrics.accuracy
    Accuracy( # MetricState: 2 (8 B)
      threshold=None,
      argname='values',
      total=MetricState( # 1 (4 B)
        value=Array(0., dtype=float32)
      ),
      count=MetricState( # 1 (4 B)
        value=Array(0, dtype=int32)
      )
    )

    >>> metrics.loss
    Average( # MetricState: 2 (8 B)
      argname='values',
      total=MetricState( # 1 (4 B)
        value=Array(0., dtype=float32)
      ),
      count=MetricState( # 1 (4 B)
        value=Array(0, dtype=int32)
      )
    )

    >>> logits = jax.random.normal(jax.random.key(0), (5, 2))
    >>> labels = jnp.array([0, 1, 1, 1, 0])
    >>> logits2 = jax.random.normal(jax.random.key(1), (5, 2))
    >>> labels2 = jnp.array([0, 1, 1, 1, 1])

    >>> batch_loss = jnp.array([1, 2, 3, 4])
    >>> batch_loss2 = jnp.array([3, 2, 1, 0])

    >>> metrics.compute()
    {'accuracy': Array(nan, dtype=float32), 'loss': Array(nan, dtype=float32)}
    >>> metrics.update(logits=logits, labels=labels, values=batch_loss)
    >>> metrics.compute()
    {'accuracy': Array(0.6, dtype=float32), 'loss': Array(2.5, dtype=float32)}
    >>> metrics.update(logits=logits2, labels=labels2, values=batch_loss2)
    >>> metrics.compute()
    {'accuracy': Array(0.4, dtype=float32), 'loss': Array(2., dtype=float32)}
    >>> metrics.reset()
    >>> metrics.compute()
    {'accuracy': Array(nan, dtype=float32), 'loss': Array(nan, dtype=float32)}
  """

  def __init__(self, **metrics):
    """Pass in key-word arguments to the constructor, e.g.
    ``MultiMetric(keyword1=Average(), keyword2=Accuracy(), ...)``.

    Args:
      **metrics: the key-word arguments that will be used to access
        the corresponding ``Metric``.
    """
    # Validate metric names before any mutation.
    for name in metrics:
      if name in _MULTIMETRIC_RESERVED_NAMES:
        raise ValueError(
          f"Metric name '{name}' conflicts with a reserved "
          f'name. Reserved names: '
          f'{sorted(_MULTIMETRIC_RESERVED_NAMES)}'
        )

    self._metric_names: list[str] = []
    self._expected_kwargs: set[str] | None = set()
    for metric_name, metric in metrics.items():
      self._metric_names.append(metric_name)
      setattr(self, metric_name, metric)
      # Collect expected kwargs for validation in update().
      if self._expected_kwargs is None:
        continue
      sig = inspect.signature(metric.update)
      has_named_params = False
      has_var_keyword = False
      named_param_names: set[str] = set()
      for pname, param in sig.parameters.items():
        if pname == 'self':
          continue
        if param.kind in (
          param.POSITIONAL_OR_KEYWORD,
          param.KEYWORD_ONLY,
        ):
          named_param_names.add(pname)
          has_named_params = True
        elif param.kind == param.VAR_KEYWORD:
          has_var_keyword = True
      if has_named_params and has_var_keyword:
        # Metric declares specific params but also absorbs
        # extras (e.g. Accuracy's **_); can't validate
        # without false positives.
        self._expected_kwargs = None
      elif has_named_params:
        self._expected_kwargs.update(named_param_names)
      elif hasattr(metric, 'argname'):
        # Use argname convention (e.g. Average, Welford).
        self._expected_kwargs.add(metric.argname)
      elif has_var_keyword:
        # Pure **kwargs with no specific params; can't
        # validate.
        self._expected_kwargs = None

  def reset(self) -> None:
    """Reset all underlying ``Metric``'s."""
    for metric_name in self._metric_names:
      getattr(self, metric_name).reset()

  def update(self, **updates) -> None:
    """In-place update all underlying ``Metric``'s.

    All ``**updates`` are forwarded to each metric's
    ``update`` method.

    Args:
      **updates: keyword arguments forwarded to each
        underlying metric's ``update`` method.

    Raises:
      TypeError: if an unexpected keyword argument is
        provided and the expected set can be statically
        determined from the underlying metrics.
    """
    # TODO: should we give the option of updating only some of the metrics and not all? e.g. if for some kwargs==None, don't do update
    if self._expected_kwargs is not None:
      unexpected = set(updates) - self._expected_kwargs
      if unexpected:
        raise TypeError(
          f'Unexpected keyword argument(s): '
          f'{sorted(unexpected)}. '
          f'Expected: {sorted(self._expected_kwargs)}'
        )
    for metric_name in self._metric_names:
      getattr(self, metric_name).update(**updates)

  def compute(self) -> dict[str, tp.Any]:
    """Compute and return the value of all underlying ``Metric``'s. This method
    will return a dictionary, mapping strings (defined by the key-word arguments
    ``**metrics`` passed to the constructor) to the corresponding metric value.
    """
    return {
        f'{metric_name}': getattr(self, metric_name).compute()
        for metric_name in self._metric_names
    }
