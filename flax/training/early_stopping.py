# Copyright 2022 The Flax Authors.
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

"""Early stopping."""

import math
from flax import struct


class EarlyStopping(struct.PyTreeNode):
  """Early stopping to avoid overfitting during training.

  The following example stops training early if the difference between losses
  recorded in the current epoch and previous epoch is less than 1e-3
  consecutively for 2 times::

    early_stop = EarlyStopping(min_delta=1e-3, patience=2)
    for epoch in range(1, num_epochs+1):
      rng, input_rng = jax.random.split(rng)
      optimizer, train_metrics = train_epoch(
          optimizer, train_ds, config.batch_size, epoch, input_rng)
      _, early_stop = early_stop.update(train_metrics['loss'])
      if early_stop.should_stop:
        print('Met early stopping criteria, breaking...')
        break

  Attributes:
    min_delta: Minimum delta between updates to be considered an
        improvement.
    patience: Number of steps of no improvement before stopping.
    best_metric: Current best metric value.
    patience_count: Number of steps since last improving update.
    should_stop: Whether the training loop should stop to avoid
        overfitting.
  """
  min_delta: float = 0
  patience: int = 0
  best_metric: float = float('inf')
  patience_count: int = 0
  should_stop: bool = False

  def reset(self):
    return self.replace(best_metric=float('inf'),
                        patience_count=0,
                        should_stop=False)

  def update(self, metric):
    """Update the state based on metric.

    Returns:
      A pair (has_improved, early_stop), where `has_improved` is True when there
      was an improvement greater than `min_delta` from the previous
      `best_metric` and `early_stop` is the updated `EarlyStop` object.
    """

    if math.isinf(self.best_metric) or self.best_metric - metric > self.min_delta:
      return True, self.replace(best_metric=metric,
                                patience_count=0)
    else:
      should_stop = self.patience_count >= self.patience or self.should_stop
      return False, self.replace(patience_count=self.patience_count + 1,
                                 should_stop=should_stop)
