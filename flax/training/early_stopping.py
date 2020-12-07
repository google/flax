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

"""Early stopping for model training."""

import numpy as np
from flax.training import checkpoints

class EarlyStopping:
  """Prevents overfitting by ending training early if validation loss
  does not improve.

  Attributes:
    ckpt_dir: str: path to store checkpoint files in.
    prefix: str: checkpoint file name prefix.
    patience: number of steps of no improvement before stopping.
    keep: number of past checkpoint files to keep.
  """
  def __init__(self, ckpt_dir, prefix='checkpoint_', min_delta=0, patience=0, keep=1):
    self.ckpt_dir = ckpt_dir
    self.min_delta = min_delta
    self.patience = patience
    self.prefix = prefix
    self.keep = keep

    self.count = 0
    self.best_loss = np.Inf
    self._should_stop = False

  @property
  def stop(self):
      return self.should_stop

  def save_checkpoint(self,
                      target, 
                      step,
                      metric):
    """Save a checkpoint of the model.

    Args:
      target: serializable flax object, usually a flax optimizer.
      step: int or float: training step number or other metric number.
      metric: int or float: metric (i.e. validation loss) to determine 
          improvement. 
    Returns:
        Filename of saved checkpoint.
    """
    output = None
    if self.best_loss - val_loss > self.min_delta:
        output = checkpoints.save_checkpoint(self.ckpt_dir,
                                             target=target,
                                             step=step,
                                             prefix=self.prefix,
                                             keep=self.keep)
        self.count = 0
        self.best_loss = val_loss
    else:
        self.count += 1
        if self.count >= self.patience:
            self.should_stop = True
    
    return output