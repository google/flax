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

"""Early stopping iterator."""

class EarlyStopping:
  """Prevents overfitting by ending training early if validation loss
  does not improve.

  Attributes:
    patience: number of steps of no improvement before stopping.
    keep: number of past checkpoint files to keep.
  """
  def __init__(self, 
               steps=0,
               min_delta=0, 
               patience=0):
    self.max_steps = steps
    self.min_delta = min_delta
    self.patience = patience
    self.reset()

  def reset(self):
    self.count = 0
    self.patience_count = 0
    self.best_loss = None
    self.should_stop = False

  def __iter__(self):
    self.reset()
    return self

  def __next__(self):
    if self.count >= self.max_steps or self.should_stop:
      raise StopIteration
    
    iteration = self.count
    self.count += 1
    return iteration

  def update(self, metric):
    """Update iterator state.

    Args:
      metric: int or float: metric (i.e. validation loss) to determine 
          improvement.
    """
    if self.best_loss is None or \
      self.best_loss - metric > self.min_delta:
      self.patience_count = 0
      self.best_loss = metric
    else:
      self.patience_count += 1
      if self.patience_count > self.patience:
          self.should_stop = True